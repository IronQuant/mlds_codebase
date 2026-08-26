import sys

sys.path.insert(0, "..")

from config import (
    SHAH_BENCHMARK_DIR as BENCHMARK_DIR,
    SHAH_SEEDS as SEEDS,
)

import polars as pl


def load(dataset="benchmark", seed=944601, cutoff=2019):
    """
    Load the Shah et al. (2023) labelled FOMC data.

    Args:
        dataset:
            "benchmark" gives the 2,480 human-labelled sentences in their
                published train/test split, one version per seed.
            "chrono" gives the same 2,480 sentences partitioned by year
                instead of at random
            "full" gives the 38,169-sentence RoBERTa pseudo-labelled corpus,
                which has no official split.
        seed: Which of the three published splits to read. Must be one of
            SHAH_SEEDS. Used by "benchmark" only, ignored by the others.
        cutoff: Last training year for "chrono". Train is year <= cutoff,
            test is everything after. Ignored by the others.

    Returns:
        For "benchmark" and "chrono", a (train, test) tuple of polars
        DataFrames. For "full", a single polars DataFrame.

    Raises:
        ValueError: If seed is not one of SHAH_SEEDS.
    """
    if dataset == "chrono":
        df = pl.concat(
            [
                # note: the seed is irrelevant here, chrono split is deterministic
                pl.read_excel(BENCHMARK_DIR / f"split-combine-train-{SEEDS[0]}.xlsx"),
                pl.read_excel(BENCHMARK_DIR / f"split-combine-test-{SEEDS[0]}.xlsx"),
            ]
        )
        train = df.filter(pl.col("year") <= cutoff)
        test = df.filter(pl.col("year") > cutoff)

        print(
            f"Chrono <={cutoff} | Train: {train.shape[0]} rows  |  "
            f"Test: {test.shape[0]} rows  |  Total: {df.shape[0]}"
        )
        return train, test

    if dataset == "benchmark":
        if seed not in SEEDS:
            raise ValueError(f"seed must be one of {SEEDS} - got {seed}")
        train = pl.read_excel(BENCHMARK_DIR / f"split-combine-train-{seed}.xlsx")
        test = pl.read_excel(BENCHMARK_DIR / f"split-combine-test-{seed}.xlsx")
        print(
            f"Seed {seed} | Train: {train.shape[0]} rows  |  "
            f"Test: {test.shape[0]} rows  |  Total: {train.shape[0] + test.shape[0]}"
        )
        return train, test


if __name__ == "__main__":
    # quick sanity check
    for seed in SEEDS:
        load("benchmark", seed)
