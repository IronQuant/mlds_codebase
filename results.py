import csv
from pathlib import Path


def save_result(path, **row):
    """Append one row to a results CSV. Header written on first call."""
    path = Path(path)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row)
        if new:
            w.writeheader()
        w.writerow(row)


def save_preds(path, preds, true, **keys):
    """Append one row per prediction. LLM preds can't be regenerated for free."""
    path = Path(path)
    new = not path.exists()
    fields = [*keys, "idx", "true", "pred"]
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        for i, (t, p) in enumerate(zip(true, preds)):
            w.writerow({**keys, "idx": i, "true": t, "pred": p})


def already_done(path, force=False, **keys):
    """True if a row matching `keys` is already in the CSV. force=True always False."""
    path = Path(path)
    if force or not path.exists():
        return False
    with open(path, newline="") as f:
        return any(
            all(r.get(k) == str(v) for k, v in keys.items()) for r in csv.DictReader(f)
        )
