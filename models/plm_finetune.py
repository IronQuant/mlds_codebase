import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, random_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)


def _load_model(model_name, device):
    # fp32 explicitly: some checkpoints (deberta-v3) declare fp16 and newer
    # transformers honors it -- half precision NaNs deberta and breaks the
    # recipe's stated fp32
    return AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=3, torch_dtype=torch.float32
    ).to(device)


def _make_examples(tok, df, max_len):
    # "Inflation pressures have eased."  (dovish)
    #   -> ['[CLS]', 'inflation', 'pressures', 'have', 'eased', '.', '[SEP]']
    #   -> {'input_ids':      [101, 14200, 15399, 2031, 10987, 1012, 102],
    #       'attention_mask': [1, 1, 1, 1, 1, 1, 1],
    #       'labels':         0}
    #
    # no padding here -- the collator pads each batch to its own longest sequence.
    # padded to 10, where 0 = [PAD] and mask 0 = ignore:
    #      {'input_ids':      [101, 14200, 15399, 2031, 10987, 1012, 102, 0, 0, 0],
    #       'attention_mask': [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    #       'labels':         0}
    enc = tok(df["sentence"].to_list(), truncation=True, max_length=max_len)
    return [
        {"input_ids": ids, "attention_mask": mask, "labels": lab}
        for ids, mask, lab in zip(
            enc["input_ids"], enc["attention_mask"], df["label"].to_list()
        )
    ]


def _evaluate(model, dl, device, class_w=None):
    """Return (cross_entropy, accuracy, weighted_f1, macro_f1) over a dataloader.

    class_w weights the reported CE to match the weighted training objective
    (so the printed loss tracks what we optimise); None gives plain CE.
    """
    model.eval()
    ce_sum, correct, n, yp, yt = 0.0, 0, 0, [], []
    with torch.no_grad():
        for batch in dl:
            labels = batch["labels"]
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            ce = F.cross_entropy(out.logits, batch["labels"], weight=class_w)
            ce_sum += ce.item() * labels.size(0)
            preds = out.logits.argmax(1).cpu()
            correct += (preds == labels).sum().item()
            n += labels.size(0)
            yp += preds.tolist()
            yt += labels.tolist()
    return (
        ce_sum / n,
        correct / n,
        f1_score(yt, yp, average="weighted"),
        f1_score(yt, yp, average="macro"),
    )


def finetune(
    train_df,
    model_name="bert-base-uncased",
    lr=1e-5,
    batch_size=8,
    max_len=256,
    max_epochs=100,
    patience=7,
    val_frac=0.2,
    val_df=None,
    seed=0,
    test_df=None,
    save_dir=None,
    device="cpu",
    verbose=False,
):
    """Fine-tune `model_name` on train_df.

    Returns (model, tokenizer, metrics).

    metrics is a dict of best val metrics:
    {val_ce, val_acc, val_f1, val_macro_f1}.

    If test_df is given, the BEST-MACRO checkpoint is scored on it and
    {test_ce, test_acc, test_f1, test_macro_f1} are added.

    If save_dir is given, the model (best-macro checkpoint) + tokenizer are
    written there.

    If val_df is given, it is used as the validation set verbatim and no split
    is taken from train_df -- needed when train_df mixes corpora and early
    stopping must track the target distribution only.

    From Shah: the 80/20 within-train validation split and the lr/batch grid.
    Everything else is ours -- the paper specifies no early stopping, optimizer,
    max_len, epoch count, or checkpoint policy for the PLMs, so AdamW, max_len
    256, class-weighted cross-entropy, restoring the best-macro checkpoint, and
    the macro-F1 patience-7 rule (min-delta 0) are our choices and must be
    reported as such.
    """

    tok = AutoTokenizer.from_pretrained(model_name)

    # Convert the training DataFrame into a list of examples
    examples = _make_examples(tok, train_df, max_len)

    # seed right before model init + split
    torch.manual_seed(seed)
    np.random.seed(seed)

    # load the actual model
    model = _load_model(model_name, device)

    if val_df is not None:
        train_ds, val_ds = examples, _make_examples(tok, val_df, max_len)
    else:
        # split the examples into training and validation sets
        n_val = int(len(examples) * val_frac)
        train_ds, val_ds = random_split(examples, [len(examples) - n_val, n_val])

    # a batch must be one rectangular tensor, so rows are padded to equal length.
    # pad per batch (to its own longest, often ~30 tokens) rather than to max_len=256
    # -- far less wasted compute, and attention_mask makes the result identical.
    collate = DataCollatorWithPadding(tok)
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate
    )
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    # inverse-frequency class weights: neutral is ~49%, so unweighted CE lets it
    # dominate the gradient and collapse the model to the majority class. Weighting
    # the training loss counters that. Reported acc/F1 stay plain; reported CE is
    # weighted too, so the printed loss tracks the objective (see _evaluate).
    counts = torch.bincount(
        torch.tensor(train_df["label"].to_list()), minlength=3
    ).float()
    class_w = (counts.sum() / (3 * counts)).to(device)

    # patience and checkpoint both key on macro-F1, the metric we select and report
    # on. min-delta 0: any new max counts (strict >), so the checkpoint is the true
    # best epoch and a slow, sub-0.01-per-epoch climb is never cut off mid-ascent.
    # best_metrics holds (ce, acc, weighted, macro) at that same epoch.
    best_macro, best_state, best_metrics = float("-inf"), None, None
    es_count = 0
    n_epochs = 0
    for epoch in range(max_epochs):
        if es_count >= patience:
            break
        es_count += 1
        n_epochs = epoch + 1
        t0 = time.perf_counter()

        # ---- train ----
        model.train()
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad()
            out = model(**batch)
            # weighted CE from logits, not the model's built-in unweighted out.loss
            loss = F.cross_entropy(out.logits, batch["labels"], weight=class_w)
            loss.backward()
            opt.step()

        # ---- validate: macro-F1 alone drives patience and the checkpoint ----
        ce, acc, f1, mf1 = _evaluate(model, val_dl, device, class_w)
        # strict > (min-delta 0): equal mF1 must NOT reset, or a frozen run (e.g.
        # lr=1e-7 stuck at one value) would never stop and hit max_epochs.
        if mf1 > best_macro:
            best_macro, es_count = mf1, 0
            best_metrics = (ce, acc, f1, mf1)
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        if verbose:
            dt = time.perf_counter() - t0
            print(
                f"    epoch {epoch:>2}: val CE={ce:.4f}  acc={acc:.4f}  wF1={f1:.4f}  mF1={mf1:.4f}  es={es_count}  {dt:.1f}s"
            )

    # restore the best-macro checkpoint, so val_macro_f1, the test score, and any
    # later predict() all describe the same model the grid selected on -- not the
    # last, possibly overfit, epoch
    if best_state is not None:
        model.load_state_dict(best_state)

    b_ce, b_acc, b_wf1, b_mf1 = best_metrics
    metrics = {
        "val_ce": b_ce,
        "val_acc": b_acc,
        "val_f1": float(b_wf1),
        "val_macro_f1": float(b_mf1),
        "epochs": n_epochs,
    }

    if test_df is not None:
        test_dl = DataLoader(
            _make_examples(tok, test_df, max_len),
            batch_size=batch_size,
            collate_fn=collate,
        )
        t_ce, t_acc, t_f1, t_mf1 = _evaluate(model, test_dl, device, class_w)
        metrics.update(
            test_ce=t_ce, test_acc=t_acc, test_f1=t_f1, test_macro_f1=t_mf1
        )

    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(save_dir)
        tok.save_pretrained(save_dir)
        if verbose:
            print(f"    saved -> {save_dir}")

    return model, tok, metrics


def predict(model, tok, test_df, max_len=256, batch_size=16, device="cpu"):
    """Integer label predictions (0/1/2) for test_df."""
    model.eval()
    enc = tok(
        test_df["sentence"].to_list(),
        truncation=True,
        max_length=max_len,
        padding=True,
        return_tensors="pt",
    )
    preds = []
    with torch.no_grad():
        for i in range(0, enc["input_ids"].size(0), batch_size):
            logits = model(
                input_ids=enc["input_ids"][i : i + batch_size].to(device),
                attention_mask=enc["attention_mask"][i : i + batch_size].to(device),
            ).logits
            preds += logits.argmax(1).cpu().tolist()
    return preds
