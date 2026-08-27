import csv
from pathlib import Path


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
