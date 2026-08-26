import os

import polars as pl

from config import SHAH_PLM, SHAH_SEEDS
from data.twd.loader import load
from models.plm_finetune import finetune
from results import save_result


# refined grid around the stable zone: Shah's decade-spaced grid had 1e-4 (collapse),
# 1e-7 (never learns), 1e-6 (trails), and batch 4 (slow/spiky). Centre tightly on the
# 1e-5 winner, keep batch 8/16/32. 3 x 3 = 9 configs (was 16).
LEARNING_RATES = (2e-5, 1e-5, 5e-6)
BATCH_SIZES = (32, 16, 8)

COLUMNS = [
    "model", "seed", "lr", "batch_size", "epochs",
    "val_ce", "val_acc", "val_f1", "val_macro_f1",
]


def grid_search(
    out_path,
    models,
    seeds=SHAH_SEEDS,
    learning_rates=LEARNING_RATES,
    batch_sizes=BATCH_SIZES,
    device="cuda",
    on_save=None,
):
    """Search lr x batch per model, recording validation metrics only.

    Winners are chosen by mean val_macro_f1 across seeds (see winners()); val_f1
    (weighted) is kept alongside only for comparison with Shah's Table 5.

    No test_df is passed to finetune, so hyperparameter selection never sees the
    test set. Appends one row to out_path after every config and skips whatever
    is already there, so a disconnect costs one run.

    out_path should live on Drive (config.RESULTS_DIR), not in the git clone --
    the row is appended in place, so nothing has to be pushed and no second
    writer can clobber rows it never read.

    on_save, if given, is called with the row dict just appended.

    Returns a polars DataFrame of all rows, resumed ones included.
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
            train, _ = load("benchmark", seed=seed)  # test deliberately unused
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
                    # append, never rewrite: rewriting the whole file lets a second
                    # writer clobber rows it never saw.
                    save_result(out_path, **{c: rows[-1][c] for c in COLUMNS})
                    print(
                        f"    epochs={m['epochs']}  "
                        f"val_macro_f1={m['val_macro_f1']:.4f}  saved"
                    )
                    if on_save is not None:
                        on_save(rows[-1])
    return pl.DataFrame(rows).select(COLUMNS)


def winners(grid_df):
    """Best (lr, batch) per model by mean validation macro-F1 across seeds."""
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
