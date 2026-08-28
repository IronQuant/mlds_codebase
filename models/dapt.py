"""
Domain-adaptive pretraining (DAPT): continued MLM on unlabeled FOMC text.
"""

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
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
    """
    Continue masked-language-model pretraining on `sentences`.

    Args:
        sentences: List of sentence strings to continue pretraining on.
        model_name: Name of the HuggingFace transformer model.
        lr: Learning rate for the optimizer.
        batch_size: Batch size for training.
        max_len: Maximum token length for the tokenizer.
        epochs: Number of epochs to train.
        mlm_probability: Probability of masking tokens for MLM.
        seed: Random seed for reproducibility.
        save_dir: Optional directory to save the fine-tuned model and tokenizer.
        device: Device to run the model on (e.g., "cpu" or "cuda").
        verbose: If True, print progress messages.

    Returns:
        The save_dir path (or the in-memory model if save_dir is None) --
        fine-tune afterwards by passing save_dir as model_name to finetune().
    """

    torch.manual_seed(seed)

    # Load tokenizer and model for MLM (bolting on  the MLM head)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)

    if verbose:
        print(f"    tokenizing {len(sentences):,} sentences...", flush=True)

    enc = tok(sentences, truncation=True, max_length=max_len)
    encoded = [
        {"input_ids": ids, "attention_mask": mask}
        for ids, mask in zip(enc["input_ids"], enc["attention_mask"])
    ]

    collate = DataCollatorForLanguageModeling(tok, mlm_probability=mlm_probability)

    dl = DataLoader(encoded, batch_size=batch_size, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    # Sort out the linear scheduler
    total_steps = len(dl) * epochs
    sched = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=int(0.06 * total_steps), num_training_steps=total_steps
    )

    if verbose:
        print(f"    training: {len(dl):,} steps/epoch x {epochs} epoch(s)", flush=True)

    # Pre-training
    model.train()
    for epoch in range(epochs):
        total, n = 0.0, 0
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
            n += 1
        if verbose:
            print(f"    epoch {epoch}: mlm loss {total / n:.4f}", flush=True)

    if save_dir is not None:
        model.save_pretrained(save_dir)
        tok.save_pretrained(save_dir)
        if verbose:
            print(f"    saved -> {save_dir}", flush=True)
        return save_dir
    return model
