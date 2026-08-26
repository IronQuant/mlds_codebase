"""Domain-adaptive pretraining (DAPT): continued MLM on unlabeled FOMC text.

Gururangan et al. (2020). The pool is FOMC policy communication only (the
register argument vs CB-LMs' mixed speeches + research prose). The pools are
decontaminated upstream in data/data_unlabelled.ipynb, so no labeled sentence
reaches this module.
"""

import time

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)


def dapt(
    sentences,
    model_name="roberta-large",
    lr=1e-5,
    batch_size=32,
    max_len=128,
    epochs=1,
    mlm_probability=0.15,
    seed=0,
    save_dir=None,
    device="cuda",
    verbose=False,
):
    """Continue masked-language-model pretraining on `sentences`.

    Standard MLM: 15% of tokens masked, model reconstructs them. One pass over
    a few hundred thousand sentences is the Gururangan et al. regime; loss is
    reported per epoch so under/overtraining is visible.

    Returns the save_dir path (or the in-memory model if save_dir is None) --
    fine-tune afterwards by passing save_dir as model_name to finetune().
    """
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)

    if verbose:
        print(f"    tokenizing {len(sentences):,} sentences...", flush=True)
    enc = tok(sentences, truncation=True, max_length=max_len)
    examples = [
        {"input_ids": ids, "attention_mask": mask}
        for ids, mask in zip(enc["input_ids"], enc["attention_mask"])
    ]
    collate = DataCollatorForLanguageModeling(tok, mlm_probability=mlm_probability)
    dl = DataLoader(examples, batch_size=batch_size, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    if verbose:
        print(f"    training: {len(dl):,} steps/epoch x {epochs} epoch(s)", flush=True)
    model.train()
    for epoch in range(epochs):
        total, n, t0 = 0.0, 0, time.time()
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            opt.step()
            total += loss.item()
            n += 1
            if verbose and n % 100 == 0:
                rate = n / (time.time() - t0)
                eta = (len(dl) - n) / rate / 60
                print(
                    f"    step {n}/{len(dl)}: mlm loss {total / n:.4f}"
                    f" | {rate:.1f} it/s, eta {eta:.0f} min",
                    flush=True,
                )
        if verbose:
            print(f"    epoch {epoch}: mlm loss {total / n:.4f}", flush=True)

    if save_dir is not None:
        model.save_pretrained(save_dir)
        tok.save_pretrained(save_dir)
        if verbose:
            print(f"    saved -> {save_dir}", flush=True)
        return save_dir
    return model
