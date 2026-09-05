# FOMC monetary policy stance classification

Code for an MSc thesis on classifying the policy stance of Federal Open Market
Committee (FOMC) sentences as hawkish, dovish or neutral.

The project rebuilds performance one capability at a time, from a rule-based
dictionary up to a fine-tuned RoBERTa-large, and then asks two further
questions. Does continued pretraining on unlabelled central bank text help? Can
labels borrowed from other central banks stand in for an institution's own?

Everything reported in the thesis can be reproduced from this repository.

## Setup
Python 3.11.
```bash
python -m venv .venv
source .venv/bin/activate        
pip install -r requirements.txt
```

`torch` is deliberately unpinned to a build. Pip installs a CPU wheel on
Windows and a CUDA wheel on Colab, and the encoder experiments need the GPU one.

The zero-shot LLM experiments need an Anthropic API key in the environment as
`ANTHROPIC_API_KEY`. Nothing else does, so you can run the rest without one.

## Data

No data files are committed. Every loader in `data/` downloads its corpus from
the HuggingFace Hub on first use, so you need an internet connection the first
time you run a notebook.

Two datasets are used, both published by other authors.

- **Trillion Dollar Words** (Shah et al., 2023) is the labelled benchmark.
  2,480 FOMC sentences with three published train/test partitions.
- **World Central Banks** (Shah et al., 2025) supplies labelled sentences from
  25 central banks, and the unlabelled text used for continued pretraining.

Both datasets are published by Shah et al. under a CC BY-NC 4.0 licence and are
downloaded from their original sources at runtime. No data files are
redistributed with this code, which is shared under CC BY 4.0.

The 100 masked-idiom items are from Gambacorta et al. (2024) and the rule-based
dictionary from Gorodnichenko et al. (2023). Both are reproduced with
attribution for replication.

## Where results go

`config.py` writes to Google Drive if it finds it mounted, and otherwise to the
repository root. Results accumulate in `results.csv`, one row per model, corpus
and seed. Rerunning a notebook skips any row already present, so an interrupted
run can be restarted without repeating finished work. Delete the rows or set
`FORCE = True` to rerun something deliberately.

Five result files are committed, so every number in the thesis can be checked
without running anything: `results.csv`, `grid.csv`, `idioms.csv`,
`idioms_flang.csv` and `mlm_loss.csv`.

## What is where

| Path | What it holds |
|---|---|
| `config.py` | Paths, label mapping, seeds, and the winning hyperparameters |
| `data/` | One loader per corpus, plus the adaptation pool construction |
| `models/` | One module per model family, each exposing a fit-and-score function |
| `experiments/` | The notebooks that produce the results |
| `utils/` | Corpus normalisation and the results-file helpers |

## Run order

The experiment notebooks are independent of one another and can be run in any
order.

1. `experiments/grid.ipynb` searches learning rate and batch size for the six
   encoders. The winners are already recorded in `config.py`, so this is only
   needed to reproduce the search itself.
2. `experiments/experiments.ipynb` runs the main comparison: baselines,
   bag-of-words, word2vec, frozen probes, fine-tuned encoders and the zero-shot
   LLMs.
3. `experiments/apt.ipynb` runs continued pretraining and evaluates the adapted
   encoders.
4. `experiments/data_curve.ipynb` refits RoBERTa-large on subsets of the
   training partitions.
5. `experiments/wcb_transfer.ipynb` runs the cross-bank label transfer arms.
6. `experiments/flang_idioms.ipynb` runs the masked-idiom probe on the FLANG
   checkpoints.

## Hardware and runtime

The encoder work was run on a single A100. On that hardware the model trains at
roughly 105 sentences per second, so a fine-tuning run on the full 1,984
training sentences takes a few minutes and a run on the 23,000-sentence
cross-bank pool takes closer to forty. Continued pretraining is the long one, at
roughly an hour per arm.

Everything runs on CPU too, and the baselines, bag-of-words, word2vec and the
zero-shot LLM calls are comfortable there. The encoder experiments are not.

Adapted encoder checkpoints are written to `models/` and are gitignored. Each is
about 1.4GB, which is past what GitHub accepts, so they are not distributed.
Rerunning `apt.ipynb` recreates them.

## Licence

Code in this repository is shared under a Creative Commons Attribution 4.0
International (CC BY 4.0) licence. See `LICENSE`. You are free to use and adapt
it, including commercially, provided you give credit.

The datasets are not covered by that licence and remain under their own terms.
Both Trillion Dollar Words and World Central Banks are released by Shah et al.
under CC BY-NC 4.0, which is non-commercial. No dataset files are redistributed
here. The loaders fetch them from the original sources.
