import numpy as np

from config import NAME2INT


def random_guess(test_df, seed=0):
    """
    Uniform random over the three classes. The chance floor.

    Args:
        test_df: The test DataFrame, used to determine how many predictions to make.
        seed: Random seed for reproducibility.
    Returns:
        A list of random integers in [0, 1, 2], one for each row in test_df.

    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, len(NAME2INT), len(test_df)).tolist()


def always_neutral(test_df):
    """
    Predict the majority class every time.
    Args:
        test_df: The test DataFrame, used to determine how many predictions to make.

    Returns:
        A list of integers, all set to the majority class
    """
    return [NAME2INT["neutral"]] * len(test_df)
