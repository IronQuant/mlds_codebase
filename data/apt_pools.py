"""

Build the DAPT pretraining pools.
The per-source loaders each clean their own corpus.

Here, we do two additional steps:
  cross-source dedupe:  A sentence in both corpora is kept once, in TWD
  decontamination:      Every labelled sentence, train and test alike, is
                        removed from both pools, so no split decision can leak
                        into pretraining
"""

import pandas as pd

from data.loader_twd_filtered import fetch_filtered
from data.loader_twd_labelled import load_splits
from data.loader_twd_unlabelled import fetch_documents
from data.loader_wcb_labelled import fetch_annotated
from data.loader_wcb_unlabelled import fetch_sentences
from utils.corpus_helpers import dedup_key, normalize


def build_pools(verbose=True):
    """
    Return (fomc, global) pools as DataFrames.

    We download and clean the TWD and WCB corpora
    We deduplicate across corpora, keeping duplicates in TWD
    We decontaminate both corpora against every labelled sentence in both corpora

    Args:
        verbose: If True, print details in processing

    Returns:
        fomc: FOMC sentences only, from the TWD corpus
        global: fomc plus the non-US WCB sentences, concatenated

    """
    # each loader returns its own corpus already normalised and deduplicated
    twd, wcb = fetch_documents(), fetch_sentences()
    if verbose:
        print("twd:", len(twd), "| wcb:", len(wcb))

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


def build_tapt_pool(seed, verbose=True):
    """
    Return the TAPT pool for one seed: the filtered corpus minus that seed's test split.

    Curated-TAPT (Gururangan et al. 2020, 5.1) pretrains on the pool the labelled
    set was sampled from. Train stays in, since TAPT trains on the task's own
    training data by design; only test is held out.

    Args:
        seed: Which published split defines the held-out test set.
        verbose: If True, print the counts.

    Returns:
        A pandas DataFrame of sentences, with the dedup key dropped.
    """
    df = fetch_filtered()
    _, test = load_splits("benchmark", seed=seed)
    keys = set(pd.Series(test["sentence"].to_list()).map(normalize).map(dedup_key))
    clean = df[~df["key"].isin(keys)]
    if verbose:
        print(f"tapt pool seed {seed}: {len(df):,} -> {len(clean):,}")
    return clean.drop(columns="key")
