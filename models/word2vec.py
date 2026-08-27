import numpy as np
from sklearn.linear_model import LogisticRegression
from models.preprocessing import clean

_KV = None


def _vectors(name="word2vec-google-news-300"):
    """
    Load a pre-trained word2vec model.
    First call downloads the model (~1.6GB) and caches it for future calls. 

    Args:
        name: The name of the pre-trained word2vec model to load.

    Returns:
        The loaded word2vec KeyedVectors instance.
    """

    global _KV
    if _KV is None:
        import gensim.downloader as api

        _KV = api.load(name)
    return _KV


def _embed(sentences, kv):
    """
    Embed a list of sentences using the provided word2vec KeyedVectors instance.
    Args:
        sentences: A list of sentences to embed.
        kv: The word2vec KeyedVectors instance.

    Returns:
        A numpy array of shape (len(sentences), kv.vector_size) containing the mean-pooled word2vec vectors.
    """


    out = np.zeros((len(sentences), kv.vector_size), dtype=np.float32)
    for i, s in enumerate(sentences):
        vecs = [kv[t] for t in clean(s) if t in kv]
        if vecs:
            out[i] = np.mean(vecs, axis=0)
    return out


def word2vec(train_df, test_df, class_weight="balanced", seed=0):
    """
    Mean-pooled word2vec vectors + logistic regression.
    Returns a list of predictions for the test set.

    Args:
        train_df: The training DataFrame, with columns "sentence" and "label".
        test_df: The test DataFrame, with column "sentence".
        class_weight: Class weight to use in the LogisticRegression.
        seed: Random seed for reproducibility.

    Returns:
        A list of predictions for the test set.

    """
    kv = _vectors()
    X = _embed(train_df["sentence"].to_list(), kv)
    Xt = _embed(test_df["sentence"].to_list(), kv)

    clf = LogisticRegression(
        max_iter=2000, class_weight=class_weight, random_state=seed
    )
    clf.fit(X, train_df["label"].to_list())
    return clf.predict(Xt).tolist()
