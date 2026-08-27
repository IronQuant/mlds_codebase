import csv
from pathlib import Path

from sklearn.metrics import f1_score

from data.loader_twd_labelled import load_splits


def save_result(path, **row):
    """
    Append one row to a results CSV. Header written on first call.
    Columns come from the keys of the first call against a fresh file, so the
    grid and the results CSVs can carry different schemas.

    Args:
        path: Path to the CSV file.
        **row: Key-value pairs representing the row to append.
    """
    path = Path(path)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row)
        if new:
            w.writeheader()
        w.writerow(row)


def already_done(path, force=False, **keys):
    """
    True if a row matching `keys` is already in the CSV.
    force=True always return False.

    Args:
        path: Path to the CSV file.
        force: If True, always return False.
        **keys: Key-value pairs to match against existing rows.
    Returns:
        True if a row matching `keys` is already in the CSV, False otherwise.
    """
    path = Path(path)
    if force or not path.exists():
        return False
    with open(path, newline="") as f:
        return any(
            all(r.get(k) == str(v) for k, v in keys.items()) for r in csv.DictReader(f)
        )


def run_seeds(out, model, predict, seeds, corpus, split, force=False):
    """Score `predict` on every seed and append one results row per seed.

    Every untrained-at-call-time model shares this shape: load the split, get
    predictions, score, save. Only fine-tuning differs, because it reports epochs
    and reads its metrics straight out of finetune().

    Args:
        out: Path to the results CSV.
        model: Row label, e.g. "bow" or "frozen:roberta-large".
        predict: Callable (train, test, seed) -> list of integer labels.
        seeds: Seeds to run.
        corpus: Row label for the corpus, e.g. "twd".
        split: Dataset name passed to load_splits.
        force: If True, rerun and append even when a row already exists.
    """
    for seed in seeds:
        if already_done(out, force=force, model=model, corpus=corpus, seed=seed):
            print(f"{model} seed {seed}: already done, skipping")
            continue

        train, test = load_splits(split, seed=seed)
        pred = predict(train, test, seed)
        true = test["label"].to_list()

        save_result(
            out,
            model=model,
            corpus=corpus,
            seed=seed,
            epochs="",  # no training
            weighted_f1=round(f1_score(true, pred, average="weighted"), 4),
            macro_f1=round(f1_score(true, pred, average="macro"), 4),
        )
        print(f"{model} seed {seed}: saved")
