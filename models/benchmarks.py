import numpy as np

from config import NAME2INT


def random_guess(train_df, test_df, seed=0):
    """Uniform random over the three classes. The chance floor."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, len(NAME2INT), len(test_df)).tolist()


def always_neutral(train_df, test_df, seed=0):
    """Predict the majority class every time."""
    return [NAME2INT["neutral"]] * len(test_df)
