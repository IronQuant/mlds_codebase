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
    """
    Load a pre-trained HuggingFace transformer model for sequence classification.

    Args:
        model_name: The name of the pre-trained model to load.
        device: The device to load the model onto (e.g., "cpu" or "cuda").
    Returns:
        The loaded model on the specified device.
    """
    return AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=3, torch_dtype=torch.float32
    ).to(device)


def _encode(tok, df, max_len):
    """
    Convert a frame of sentences/labels into a list of dicts
    suitable for a HuggingFace DataLoader.

    Args:
        tok: The tokenizer to use.
        df: A DataFrame with "sentence" and "label" columns.
        max_len: The maximum sequence length.

    Returns:
        A list of dicts with keys "input_ids", "attention_mask", and "labels".
    """

    # "Inflation pressures have eased."  (dovish)
    #   -> ['[CLS]', 'inflation', 'pressures', 'have', 'eased', '.', '[SEP]']
    #   -> {'input_ids':      [101, 14200, 15399, 2031, 10987, 1012, 102],
    #       'attention_mask': [1, 1, 1, 1, 1, 1, 1],
    #       'labels':         0}
    #
    # no padding here -- the collator pads each batch to its own longest sequence.

    enc = tok(df["sentence"].to_list(), truncation=True, max_length=max_len)

    return [
        {"input_ids": ids, "attention_mask": mask, "labels": lab}
        for ids, mask, lab in zip(
            enc["input_ids"], enc["attention_mask"], df["label"].to_list()
        )
    ]


def _evaluate(model, dl, device, class_w=None):
    """
    Evaluate a model on a dataloader.

    Args:
        model: The model to evaluate.
        dl: The dataloader.
        device: The device to run evaluation on.
        class_w: Optional class weights for the cross-entropy loss.

    Returns:
        A tuple (cross_entropy, accuracy, weighted_f1, macro_f1).
    """
    model.eval()  # flip model to eval mode (no dropout, etc.)

    # ce_sum: total CE loss,
    # correct: number of correct predictions,
    # n: total number of samples,
    # yp: predicted labels,
    # yt: true labels

    ce_sum, correct, n, yp, yt = 0.0, 0, 0, [], []
    with torch.no_grad():
        for batch in dl:
            # true labels for this batch
            labels = batch["labels"]

            # move everything to the correct device
            batch = {k: v.to(device) for k, v in batch.items()}

            # run the model and get logits
            out = model(**batch)

            # compute CE Loss
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


def _make_loaders(tok, train_df, max_len, batch_size, val_frac):
    """Split the encoded rows and wrap each half in a padding DataLoader."""
    # Tokenise the training frame into rows the DataLoader can batch
    encoded = _encode(tok, train_df, max_len)

    # Train / Validation Split of the Encoded Data
    # Store in Torch Dataset objects
    n_val = int(len(encoded) * val_frac)
    train_ds, val_ds = random_split(encoded, [len(encoded) - n_val, n_val])

    # create the Torch Dataloaders
    collate = DataCollatorWithPadding(tok)
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate
    )
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    return train_dl, val_dl


def _class_weights(train_df, device):
    """Inverse-frequency weights, so 49% neutral cannot dominate the gradient."""
    # Setup inverse class weights for weighted CE loss
    counts = torch.bincount(
        torch.tensor(train_df["label"].to_list()), minlength=3
    ).float()
    return (counts.sum() / (3 * counts)).to(device)


def _train_epoch(model, train_dl, opt, class_w, device):
    """One pass over the training data."""
    model.train()
    for batch in train_dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        opt.zero_grad()
        out = model(**batch)
        # weighted CE from logits, not the model's built-in unweighted out.loss
        loss = F.cross_entropy(out.logits, batch["labels"], weight=class_w)
        loss.backward()
        opt.step()


def _score(model, tok, df, max_len, batch_size, class_w, device, prefix):
    """Score a frame with the current weights, keys prefixed e.g. test_macro_f1."""
    dl = DataLoader(
        _encode(tok, df, max_len),
        batch_size=batch_size,
        collate_fn=DataCollatorWithPadding(tok),
    )
    ce, acc, f1, mf1 = _evaluate(model, dl, device, class_w)
    return {
        f"{prefix}_ce": ce,
        f"{prefix}_acc": acc,
        f"{prefix}_f1": f1,
        f"{prefix}_macro_f1": mf1,
    }


def finetune(
    train_df,
    model_name="bert-base-uncased",
    lr=1e-5,
    batch_size=8,
    max_len=256,
    max_epochs=100,
    patience=7,
    val_frac=0.2,
    seed=0,
    test_df=None,
    save_dir=None,
    device="cpu",
    verbose=False,
):
    """
    Fine-tune `model_name` on train_df.

    Returns (model, tokenizer, metrics).

    metrics is a dict of best validation metrics:
    {val_ce, val_acc, val_f1, val_macro_f1}.

    If test_df is given, the BEST-MACRO checkpoint is scored on it and
    {test_ce, test_acc, test_f1, test_macro_f1} are added.

    If save_dir is given, the model (best-macro checkpoint) + tokenizer are
    written there.

    Args:
        train_df: A pandas DataFrame containing training sentences and labels.
        model_name: Name of the HuggingFace transformer model.
        lr: Learning rate for the optimizer.
        batch_size: Batch size for training and evaluation.
        max_len: Maximum token length for the tokenizer.
        max_epochs: Maximum number of epochs to train.
        patience: Number of epochs to wait for improvement before early stopping.
        val_frac: Fraction of training data to use for validation.
        seed: Random seed for reproducibility.
        test_df: Optional pandas DataFrame containing test sentences and labels.
        save_dir: Optional directory to save the best model and tokenizer.
        device: Device to run the model on (e.g., "cpu" or "cuda").
        verbose: If True, print progress messages.
    Returns:
        A tuple (model, tokenizer, metrics) where:
            - model: The fine-tuned HuggingFace model.
            - tokenizer: The tokenizer corresponding to the model.
            - metrics: A dictionary containing validation and test metrics.

    """

    # Load the tokenizer from model_name
    tok = AutoTokenizer.from_pretrained(model_name)

    # set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # load the actual model
    model = _load_model(model_name, device)

    # create the training and validation DataLoaders
    train_dl, val_dl = _make_loaders(tok, train_df, max_len, batch_size, val_frac)

    # setup optimizer qand class weights for weighted CE loss
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    class_w = _class_weights(train_df, device)

    best_macro, best_state, best_metrics = float("-inf"), None, None
    es_count = 0
    n_epochs = 0
    for epoch in range(max_epochs):
        if es_count >= patience:
            break
        es_count += 1
        n_epochs = epoch + 1
        t0 = time.perf_counter()

        # train
        _train_epoch(model, train_dl, opt, class_w, device)

        # validate
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

    # restore the best-macro checkpoint, so val_macro_f1
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
        metrics.update(
            _score(model, tok, test_df, max_len, batch_size, class_w, device, "test")
        )

    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(save_dir)
        tok.save_pretrained(save_dir)
        if verbose:
            print(f"    saved -> {save_dir}")

    return model, tok, metrics
