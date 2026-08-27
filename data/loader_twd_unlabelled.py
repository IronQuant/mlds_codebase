"""Acquire the unfiltered TWD sentence corpus.

Shah et al. release whole documents, not sentences, so the ~172k-sentence corpus
described in the paper has to be rebuilt by sentence-splitting the raw documents
in the gtfintechlab/fomc-hawkish-dovish GitHub repo. NLTK's boundary decisions
differ slightly from theirs, which is why this yields 179,638 rather than 172,454.
"""

import glob
import io
import os
import re
import tarfile
import tempfile
import urllib.request

from pathlib import Path

import ftfy
import nltk
import pandas as pd

from utils.corpus_helpers import is_junk, rejoin_hyphens

TARBALL = (
    "https://github.com/gtfintechlab/fomc-hawkish-dovish/archive/refs/heads/main.tar.gz"
)
PREFIX = "fomc-hawkish-dovish-main/data/raw_data/"


def _download_raw(RAW):
    """Download and extract the raw_data tree from the TDW repo."""
    RAW.mkdir(parents=True, exist_ok=True)
    print("downloading TDW repo tarball (~61MB)...", flush=True)
    buf = io.BytesIO(urllib.request.urlopen(TARBALL).read())
    with tarfile.open(fileobj=buf, mode="r:gz") as tf:
        for m in tf.getmembers():
            if PREFIX in m.name and m.name.endswith((".txt", ".csv")):
                m.name = m.name.split(PREFIX, 1)[1]
                tf.extract(m, RAW)
    print("extracted ->", RAW)


def _date(fname):
    m = re.search(r"(\d{8})", os.path.basename(fname))
    return int(m.group(1)) if m else pd.NA


def _keep(s, doc_type, date, rows):
    """Drop lines that are not language: PDF headers, tables, web chrome, fragments."""
    s = rejoin_hyphens(s)
    web_nav = "Toggle Button" in s or "Main Menu" in s or "Skip to main content" in s
    if is_junk(s) or web_nav or len(s.split()) < 4:
        return
    rows.append((s, doc_type, date))


def fetch_documents():
    """Return a DataFrame of (sentence, doc_type, meeting_date). 179,638 rows."""
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    from nltk.tokenize import sent_tokenize

    tmp = tempfile.TemporaryDirectory()
    RAW = Path(tmp.name)
    _download_raw(RAW)
    rows, stats = [], {}
    # minutes and speeches are whole documents; press conferences ship sentence-level
    for dt, sub in [
        ("meeting_minutes", "meeting_minutes"),
        ("speech", "speech/text/all"),
    ]:
        files = glob.glob(str(RAW / sub / "*.txt"))
        before = len(rows)
        for f in files:
            text = ftfy.fix_text(open(f, encoding="utf-8", errors="ignore").read())
            for s in sent_tokenize(text):
                _keep(s, dt, _date(f), rows)
        stats[dt] = (len(files), len(rows) - before)

    pc = glob.glob(str(RAW / "press_conference" / "csv" / "all" / "*.csv"))
    before = len(rows)
    for f in pc:
        for s in pd.read_csv(f)["sentence"].dropna().astype(str):
            _keep(ftfy.fix_text(s), "press_conference", _date(f), rows)
    stats["press_conference"] = (len(pc), len(rows) - before)

    for k, (nf, ns) in stats.items():
        print(f"  {k}: {nf} docs -> {ns:,} sentences")
    return pd.DataFrame(rows, columns=["sentence", "doc_type", "meeting_date"])
