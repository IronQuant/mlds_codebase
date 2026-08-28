"""Acquire the WCB annotated sentences (Shah et al. 2025)."""

import pandas as pd
from huggingface_hub import hf_hub_download

from config import NAME2INT
from utils.corpus_helpers import dedup_key, normalize

REPO = "gtfintechlab/all_annotated_sentences_25000"
SEED_DIR = "5768"


def fetch_annotated(verbose=True):
    """
    Return the 23,182 labelled non-US sentences, deduplicated.
    Downloaded from the HuggingFace Hub
    We further drop FOMC sentences, irrelevant sentences, and dupes

    Args:
        verbose: Print row counts before and after filtering.
    Returns:
        A pandas DataFrame with columns:
            - bank_name: The bank from which the sentence was drawn.
            - sentence: The raw sentence text.
            - label: "hawkish" or "dovish".
            - label_int: 0 for dovish, 1 for hawkish, 2 for neutral.
    """
    df = pd.concat(
        [
            pd.read_parquet(
                hf_hub_download(
                    REPO, f"{SEED_DIR}/{sp}-00000-of-00001.parquet", repo_type="dataset"
                )
            )
            for sp in ("train", "val", "test")
        ],
        ignore_index=True,
    )
    if verbose:
        print(
            "raw rows:",
            len(df),
            "| stance labels:",
            df["stance_label"].value_counts().to_dict(),
        )

    # Drop "Irrelevant" Sentences
    df = df[df["stance_label"].isin(NAME2INT)]

    # Drop FOMC Sentences
    df = df[df["bank_name"] != "fomc"].copy()
    df["sentence"] = df["sentences"]
    df["label"] = df["stance_label"]
    df["label_int"] = df["label"].map(NAME2INT)

    # Drop duplicates
    key = df["sentence"].map(normalize).map(dedup_key)
    df = df[~key.duplicated()]
    if verbose:
        print("after dropping irrelevant, fomc and duplicates:", len(df))
    return df[["bank_name", "sentence", "label", "label_int"]].reset_index(drop=True)
