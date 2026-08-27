"""
Acquire the Shah et al. (2023) labelled benchmark splits.
GitHub is the only source for all three published
splits. Fetch_splits() returns all six splits stacked 
Load_splits() slices one seed into train and test frames
(or the chronological partition out of them) 
Nothing is stored.
"""

import io
import urllib.request

import polars as pl

from config import SHAH_SEEDS as SEEDS

RAW = "https://raw.githubusercontent.com/gtfintechlab/fomc-hawkish-dovish/main/training_data/test-and-training"

URL = {
    "train": RAW + "/training_data/lab-manual-split-combine-train-{seed}.xlsx",
    "test": RAW + "/test_data/lab-manual-split-combine-test-{seed}.xlsx",
}


def _read(split, seed):
    """
    Read one of the six published splits from GitHub.
    Six = 3 seeds x 2 splits (train/test)
    
    Args:
        split: "train" or "test".
        seed: One of SHAH_SEEDS.
    Returns:
        A polars DataFrame with columns:
            - orig_index: The row number in the original Excel file.
            - sentence: The raw sentence text.
            - label: "hawkish" or "dovish".
            - year: The year of the FOMC meeting from which the sentence was drawn.
    """

    with urllib.request.urlopen(URL[split].format(seed=seed)) as r:
        return pl.read_excel(io.BytesIO(r.read()))


def fetch_splits(verbose=True):
    """
    Return all six published splits stacked, with seed and split columns.
    2,480 sentences x 3 seeds = 7,440 rows. 
    
    Args:
        verbose: Print row counts for each seed/split.
    Returns:
        A polars DataFrame with columns:
            - orig_index: The row number in the original Excel file.
            - sentence: The raw sentence text.
            - label: "hawkish" or "dovish".
            - year: The year of the FOMC meeting from which the sentence was drawn.
            - seed: Which of the three published splits this row belongs to.
            - split: "train" or "test".

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
