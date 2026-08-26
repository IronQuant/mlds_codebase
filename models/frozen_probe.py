import torch
from sklearn.linear_model import LogisticRegression
from transformers import AutoModel, AutoTokenizer


def _encode(model, tok, sentences, max_len, batch_size, device):
    vecs = []
    # no gradients, weights never change -- the "frozen" part
    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            # sentence -> ~7 token ids
            enc = tok(
                sentences[i : i + batch_size],
                truncation=True,
                max_length=max_len,
                padding=True,
                return_tensors="pt",
            ).to(device)

            # one 768-dim vector per token, each word in context
            hidden = model(**enc).last_hidden_state

            # average tokens into one vector per sentence, padding excluded
            mask = enc["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1)
            vecs.append(pooled.cpu())
    return torch.cat(vecs).numpy()


def probe(
    train_df,
    test_df,
    model_name="roberta-base",
    max_len=256,
    batch_size=32,
    device="cpu",
    class_weight="balanced",
    seed=0,
):
    """Mean-pooled frozen encoder + logistic regression. Returns test predictions."""
    # encoder only -- no classification head
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    # 1984 x 768 train matrix, 496 x 768 test
    X = _encode(model, tok, train_df["sentence"].to_list(), max_len, batch_size, device)
    Xt = _encode(model, tok, test_df["sentence"].to_list(), max_len, batch_size, device)

    # only this learns: 768 features -> 3 classes
    clf = LogisticRegression(
        max_iter=2000, class_weight=class_weight, random_state=seed
    )
    clf.fit(X, train_df["label"].to_list())
    return clf.predict(Xt).tolist()
