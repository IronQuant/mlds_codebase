from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from models.preprocessing import clean


def bow(train_df, test_df, ngram_range=(1, 1), min_df=2, class_weight="balanced", seed=0):
    """
    Bag-of-words counts + logistic regression. Returns test predictions.
    Returns a list of predictions for the test set.
    """
    # clean() does the tokenising
    vec = CountVectorizer(
        tokenizer=clean,
        lowercase=False,
        token_pattern=None,
        ngram_range=ngram_range,
        min_df=min_df,
    )

    # fit on train only
    X = vec.fit_transform(train_df["sentence"].to_list())
    Xt = vec.transform(test_df["sentence"].to_list())

    clf = LogisticRegression(
        max_iter=2000, class_weight=class_weight, random_state=seed
    )
    clf.fit(X, train_df["label"].to_list())
    return clf.predict(Xt).tolist()
