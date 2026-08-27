"""Acquire the Shah et al. (2023) labelled benchmark splits.

Fetched from the gtfintechlab/fomc-hawkish-dovish GitHub repo. HuggingFace
carries only seed 944601, so GitHub is the only source for all three published
splits. fetch_splits() returns all six splits stacked; load_splits() slices one seed
or the chronological partition out of them. Nothing is stored.
"""

import io
import urllib.request

import polars as pl

from config import SHAH_SEEDS as SEEDS

RAW = "https://raw.githubusercontent.com/gtfintechlab/fomc-hawkish-dovish/main/training_data/test-and-training"
# upstream carries a lab-manual- prefix on every file
URL = {
    "train": RAW + "/training_data/lab-manual-split-combine-train-{seed}.xlsx",
    "test": RAW + "/test_data/lab-manual-split-combine-test-{seed}.xlsx",
}


def _read(split, seed):
    with urllib.request.urlopen(URL[split].format(seed=seed)) as r:
        return pl.read_excel(io.BytesIO(r.read()))


def fetch_splits(verbose=True):
    """Return all six published splits stacked, with seed and split columns.

    2,480 sentences x 3 seeds = 7,440 rows. The seeds are three partitions of
    one identical set, so this repeats each sentence three times. Storing one
    row per sentence is not possible: no column is unique (2,480 rows, 1,070
    distinct orig_index), and a duplicated sentence can be train under one seed
    and test under another, so a per-sentence split label has no meaning.
    """
    frames = []
    for seed in SEEDS:
        for split in ("train", "test"):
            df = _read(split, seed)
            frames.append(
                df.with_columns(
                    pl.lit(seed).alias("seed"), pl.lit(split).alias("split")
                )
            )
            if verbose:
                print(f"  seed {seed} {split}: {len(df)} rows")
    return pl.concat(frames)


def load_splits(dataset="benchmark", seed=944601, cutoff=2019):
    """
    Load the labelled benchmark from data/twd/labelled/sentences.csv.

    Args:
        dataset:
            "benchmark" gives the published train/test split for one seed.
            "chrono" gives the same sentences partitioned by year instead.
        seed: Which published split to use. Must be one of SHAH_SEEDS.
            Used by "benchmark" only, ignored by "chrono".
        cutoff: Last training year for "chrono". Ignored by "benchmark".

    Returns:
        A (train, test) tuple of polars DataFrames.

    Raises:
        ValueError: If seed is not one of SHAH_SEEDS, or dataset is unknown.
    """
    df = fetch_splits(verbose=False)

    if dataset == "chrono":
        # the three seeds are three partitions of one set, so which we read is arbitrary
        one = df.filter(pl.col("seed") == SEEDS[0]).drop("seed", "split")
        train = one.filter(pl.col("year") <= cutoff)
        test = one.filter(pl.col("year") > cutoff)
        print(
            f"Chrono <={cutoff} | Train: {train.shape[0]} rows  |  "
            f"Test: {test.shape[0]} rows  |  Total: {one.shape[0]}"
        )
        return train, test

    if dataset == "benchmark":
        if seed not in SEEDS:
            raise ValueError(f"seed must be one of {SEEDS} - got {seed}")
        one = df.filter(pl.col("seed") == seed).drop("seed")
        train = one.filter(pl.col("split") == "train").drop("split")
        test = one.filter(pl.col("split") == "test").drop("split")
        print(
            f"Seed {seed} | Train: {train.shape[0]} rows  |  "
            f"Test: {test.shape[0]} rows  |  Total: {train.shape[0] + test.shape[0]}"
        )
        return train, test

    raise ValueError(f'dataset must be "benchmark" or "chrono" - got {dataset!r}')
