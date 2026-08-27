"""Build the DAPT pretraining pools.

The per-source loaders each clean their own corpus. Two steps cannot live there
because they span sources, and both are what the leakage argument rests on:

  cross-source dedupe   a sentence in both corpora is kept once, in TWD
  decontamination       every labelled sentence, train and test alike, is
                        removed from both pools, so no split decision can leak
                        into pretraining
"""

import pandas as pd

from data.loader_twd_labelled import load_splits
from data.loader_twd_unlabelled import fetch_documents
from data.loader_wcb_labelled import fetch_annotated
from data.loader_wcb_unlabelled import fetch_sentences
from utils.corpus_helpers import dedup_key, normalize


def _clean(df):
    """Normalise the text and attach a punctuation-insensitive dedup key."""
    df = df.copy()
    df["sentence"] = df["sentence"].astype(str).map(normalize)
    df["key"] = df["sentence"].map(dedup_key)
    return df


def build_pools(verbose=True):
    """Return (fomc, global) pools as DataFrames. 164,688 and 439,414 rows."""
    twd = _clean(fetch_documents())
    if verbose:
        print("twd raw:", len(twd), twd["doc_type"].value_counts().to_dict())
    twd = twd.drop_duplicates("key")

    wcb = fetch_sentences()
    wcb = wcb[~wcb["bank"].isin(["federal_reserve_system", "fomc"])]
    wcb = _clean(wcb).drop_duplicates("key")
    if verbose:
        print("wcb after fed drop and within-source dedupe:", len(wcb))

    # cross-source: a sentence in both corpora belongs to the FOMC pool
    cross = wcb["key"].isin(set(twd["key"]))
    wcb = wcb[~cross]
    if verbose:
        print("cross-source duplicates dropped from wcb:", int(cross.sum()))

    # decontamination against every labelled sentence in both corpora
    train, test = load_splits("benchmark", seed=5768)
    labelled = (
        train["sentence"].to_list()
        + test["sentence"].to_list()
        + fetch_annotated(verbose=False)["sentence"].to_list()
    )
    keys = set(pd.Series(labelled, dtype=str).map(normalize).map(dedup_key))
    if verbose:
        print("labelled key set:", len(keys))
        for name, df in [("twd", twd), ("wcb", wcb)]:
            print(
                f"  {name}: {int(df['key'].isin(keys).sum())} contaminated rows removed"
            )
    twd = twd[~twd["key"].isin(keys)].drop(columns="key")
    wcb = wcb[~wcb["key"].isin(keys)].drop(columns="key")

    glob = pd.concat([twd.assign(bank="fomc"), wcb], ignore_index=True)
    if verbose:
        print(f"fomc pool: {len(twd):,} | global pool: {len(glob):,}")
    return twd, glob
