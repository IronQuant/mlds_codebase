from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from models.preprocessing import clean


def bow(
    train_df, test_df, ngram_range=(1, 1), min_df=2, class_weight="balanced", seed=0
):
    """
    Bag-of-words counts + logistic regression. Returns test predictions.

    Args:
        train_df: The training DataFrame, with columns "sentence" and "label".
        test_df: The test DataFrame, with column "sentence".
        ngram_range: The ngram range to use in the CountVectorizer.
        min_df: The minimum document frequency to use in the CountVectorizer.
        class_weight: Class weight to use in the LogisticRegression.
        seed: Random seed for reproducibility.

    Returns:
        A list of predictions for the test set.
    """

    vec = CountVectorizer(
        tokenizer=clean,
        lowercase=False,
        token_pattern=None,
        ngram_range=ngram_range,
        min_df=min_df,
    )

    X = vec.fit_transform(train_df["sentence"].to_list())
    Xt = vec.transform(test_df["sentence"].to_list())

    clf = LogisticRegression(
        max_iter=2000, class_weight=class_weight, random_state=seed
    )
    clf.fit(X, train_df["label"].to_list())
    return clf.predict(Xt).tolist()
