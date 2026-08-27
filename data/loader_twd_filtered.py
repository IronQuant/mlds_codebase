"""
Acquire the keyword-filtered TWD sentence corpus.
This is the 38,169-sentence pool Shah et al. sampled the 2,480 labelled
sentences from, so it is the task distribution rather than the broad domain.
Curated-TAPT (Gururangan et al. 2020, section 5.1) pretrains on exactly this.
Already sentence-level in the repo, so no splitting is needed.
"""

import glob
import io
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import pandas as pd

from utils.corpus_helpers import dedup_key, normalize

TARBALL = (
    "https://github.com/gtfintechlab/fomc-hawkish-dovish/archive/refs/heads/main.tar.gz"
)
PREFIX = "fomc-hawkish-dovish-main/data/filtered_data/"
# speech_labeled holds the 201 title-filtered speeches, matching the paper.
# the _labeled dirs also carry RoBERTa pseudo-labels; only `sentence` is read.
SUBDIRS = {
    "meeting_minutes": "meeting_minutes",
    "press_conference": "press_conference",
    "speech": "speech_labeled",
}


def _download_filtered(RAW):
    """
    Download and extract the filtered_data tree from the TDW repo.

    Args:
        RAW: Path to a temporary directory in which to extract the files.

    Returns:
        None. The filtered_data tree is extracted to RAW.
    """
    print("downloading TDW repo tarball (~61MB)...", flush=True)
    buf = io.BytesIO(urllib.request.urlopen(TARBALL).read())
    with tarfile.open(fileobj=buf, mode="r:gz") as tf:
        for m in tf.getmembers():
            if PREFIX in m.name and m.name.endswith(".csv"):
                m.name = m.name.split(PREFIX, 1)[1]
                tf.extract(m, RAW)
    print("extracted ->", RAW)


def fetch_filtered():
    """
    Return the filtered TWD corpus, normalised and deduplicated.

    38,169 raw sentences: 20,618 meeting minutes, 5,086 press conference,
    12,465 speech, matching Shah et al. Table 3 post-filter counts.

    Returns:
        A pandas DataFrame with columns:
            - sentence: The normalised sentence text.
            - doc_type: "meeting_minutes", "press_conference", or "speech".
            - key: Punctuation-insensitive dedup key.
    """
    tmp = tempfile.TemporaryDirectory()
    RAW = Path(tmp.name)
    _download_filtered(RAW)

    rows = []
    for doc_type, sub in SUBDIRS.items():
        files = glob.glob(str(RAW / sub / "*.csv"))
        before = len(rows)
        for f in files:
            for s in pd.read_csv(f)["sentence"].dropna().astype(str):
                rows.append((s, doc_type))
        print(f"  {doc_type}: {len(files)} docs -> {len(rows) - before:,} sentences")

    df = pd.DataFrame(rows, columns=["sentence", "doc_type"])
    df["sentence"] = df["sentence"].map(normalize)
    df["key"] = df["sentence"].map(dedup_key)
    return df.drop_duplicates("key")
