"""Acquire the WCB 380k unlabelled sentence corpus (Shah et al. 2025)."""

import pandas as pd
from huggingface_hub import hf_hub_download

from utils.corpus_helpers import dedup_key, normalize

REPO = "gtfintechlab/WCB_380k_sentences"
FILE = "data/train-00000-of-00001.parquet"


def fetch_sentences():
    """
    Return 380,200 sentences across 25 banks, the Fed included.
    Dowloaded from the HuggingFace Hub, where it is stored as a single Parquet file.

    Returns:
        A pandas DataFrame with columns:
            - sentence: The raw sentence text.
            - bank: The bank from which the sentence was drawn.
            - meeting_date: The date of the meeting from which the sentence was drawn.
    """
    df = pd.read_parquet(hf_hub_download(REPO, FILE, repo_type="dataset"))
    df = df[["sentence", "bank", "release_date"]].rename(
        columns={"release_date": "meeting_date"}
    )
    # the Fed's own text belongs to the TWD pool, not the non-US increment
    df = df[~df["bank"].isin(["federal_reserve_system", "fomc"])].copy()
    df["sentence"] = df["sentence"].astype(str).map(normalize)
    df["key"] = df["sentence"].map(dedup_key)
    return df.drop_duplicates("key")
