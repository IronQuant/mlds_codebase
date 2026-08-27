import os

import polars as pl

from config import SHAH_PLM, SHAH_SEEDS
from data.loader_twd_labelled import load_splits
from models.plm_finetune import finetune
from results import save_result

LEARNING_RATES = (2e-5, 1e-5, 5e-6)
BATCH_SIZES = (32, 16, 8)

COLUMNS = [
    "model",
    "seed",
    "lr",
    "batch_size",
    "epochs",
    "val_ce",
    "val_acc",
    "val_f1",
    "val_macro_f1",
]


def grid_search(
    out_path,
    models,
    seeds=SHAH_SEEDS,
    learning_rates=LEARNING_RATES,
    batch_sizes=BATCH_SIZES,
    device="cuda",
):
    """
    
    Search lr x batch per model, recording validation metrics only.

    Winners are chosen by mean val_macro_f1 across seeds (see winners()); val_f1
    (weighted) is kept alongside only for comparison with Shah's Table 5.

    Args:
        out_path: Path to the CSV file where results will be saved.
        models: List of model names to grid search over.
        seeds: List of random seeds for reproducibility.
        learning_rates: List of learning rates to try.
        batch_sizes: List of batch sizes to try.
        device: Device to run the model on (e.g., "cpu" or "cuda").

    Returns:
        A polars DataFrame containing the grid search results with columns:
            - model: The name of the model.
            - seed: The random seed used for this run.
            - lr: The learning rate used for this run.
            - batch_size: The batch size used for this run.
            - epochs: The number of epochs trained before early stopping.
            - val_ce: Validation cross-entropy loss.
            - val_acc: Validation accuracy.
            - val_f1: Validation weighted F1 score.
            - val_macro_f1: Validation macro F1 score.
    """
    rows, done = [], set()
    if os.path.exists(out_path):
        rows = pl.read_csv(out_path).to_dicts()
        done = {(r["model"], r["seed"], r["lr"], r["batch_size"]) for r in rows}
        print(f"resuming -- {len(done)} configs already done", flush=True)

    total = len(models) * len(seeds) * len(batch_sizes) * len(learning_rates)
    for name in models:
        cfg = SHAH_PLM[name]
        for seed in seeds:
            train, _ = load_splits("benchmark", seed=seed)
            for batch_size in batch_sizes:
                for lr in learning_rates:
                    if (name, seed, lr, batch_size) in done:
                        continue
                    print(
                        f"[{len(rows) + 1}/{total}] {name} seed={seed} "
                        f"lr={lr} batch={batch_size}",
                        flush=True,
                    )
                    _, _, m = finetune(
                        train,
                        model_name=cfg["model_name"],
                        lr=lr,
                        batch_size=batch_size,
                        max_len=cfg.get("max_len", 256),
                        seed=seed,
                        device=device,
                        verbose=True,
                    )
                    rows.append(
                        {
                            "model": name,
                            "seed": seed,
                            "lr": lr,
                            "batch_size": batch_size,
                            "epochs": m["epochs"],
                            "val_ce": round(m["val_ce"], 4),
                            "val_acc": round(m["val_acc"], 4),
                            "val_f1": round(m["val_f1"], 4),
                            "val_macro_f1": round(m["val_macro_f1"], 4),
                        }
                    )
                    # append after every config -- a disconnect loses one run.
                    save_result(out_path, **{c: rows[-1][c] for c in COLUMNS})
                    print(
                        f"    epochs={m['epochs']}  "
                        f"val_macro_f1={m['val_macro_f1']:.4f}  saved"
                    )
    return pl.DataFrame(rows).select(COLUMNS)


def winners(grid_df):
    """
    Best (lr, batch) per model by mean validation macro-F1 across seeds.
    
    Args:
        grid_df: A polars DataFrame containing the grid search results with columns:

    Returns:
        A polars DataFrame containing the best (lr, batch) configuration for each model
            - model: The name of the model.
            - lr: The learning rate of the best configuration.
            - batch_size: The batch size of the best configuration.
            - mean_val_macro_f1: The mean validation macro-F1 score across seeds.
            - std_val_macro_f1: The standard deviation of the validation macro-F1 score across seeds.
            - mean_epochs: The mean number of epochs trained before early stopping across seeds.
            - n: The number of seeds used for averaging.
    """
    return (
        grid_df.group_by(["model", "lr", "batch_size"])
        .agg(
            pl.col("val_macro_f1").mean().alias("mean_val_macro_f1"),
            pl.col("val_macro_f1").std(ddof=0).alias("std_val_macro_f1"),
            pl.col("epochs").mean().alias("mean_epochs"),
            pl.len().alias("n"),
        )
        .sort("mean_val_macro_f1", descending=True)
        .group_by("model", maintain_order=True)
        .first()
    )
