"""Acquire the WCB 380k unlabelled sentence corpus (Shah et al. 2025)."""

import pandas as pd
from huggingface_hub import hf_hub_download

REPO = "gtfintechlab/WCB_380k_sentences"
FILE = "data/train-00000-of-00001.parquet"


def fetch_sentences():
    """Return 380,200 sentences across 25 banks, the Fed included.

    The source carries no doc_type; WCB is minutes or the local equivalent
    throughout, so only the bank and release date are kept alongside the text.
    """
    df = pd.read_parquet(hf_hub_download(REPO, FILE, repo_type="dataset"))
    return df[["sentence", "bank", "release_date"]].rename(
        columns={"release_date": "meeting_date"}
    )
