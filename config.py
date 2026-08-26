from pathlib import Path

# Directories
_ROOT = Path(__file__).parent
DATA_DIR = _ROOT / "data"

# integer label mapping for hawkish/dovish/neutral classification
NAME2INT = {"dovish": 0, "hawkish": 1, "neutral": 2}
INT2NAME = {0: "dovish", 1: "hawkish", 2: "neutral"}

# result CSVs live on Google Drive: Colab writes them there, and locally Drive for
# Desktop exposes the same folder at G:. Falls back to the repo if neither exists.
for _cand in (Path("/content/drive/MyDrive/thesis"), Path("G:/My Drive/thesis")):
    if _cand.is_dir():
        RESULTS_DIR = _cand
        break
else:
    RESULTS_DIR = _ROOT

# Shah et al. (2023) -- Trillion Dollar Words
SHAH_DATA_DIR = DATA_DIR / "twd"
# per-seed split-combine train/test xlsx
SHAH_BENCHMARK_DIR = SHAH_DATA_DIR / "raw"
# 944601 == the HuggingFace split
SHAH_SEEDS = (5768, 78516, 944601)

# HF model id + lr/batch + Shah Table 5 Combined-S target (mean, std ddof=1).
# lr/batch = winners of the fp32 + class-weighted 9-config grid (mean val
# macro-F1 over 3 seeds, winners() on grid.csv, 2026-07-29).
SHAH_PLM = {
    "bert-base-uncased": dict(
        model_name="bert-base-uncased", lr=2e-5, batch_size=8, target=(0.6360, 0.0225)
    ),
    "bert-large-uncased": dict(
        model_name="bert-large-uncased", lr=1e-5, batch_size=8, target=(0.6619, 0.0123)
    ),
    "roberta-base": dict(
        model_name="roberta-base", lr=1e-5, batch_size=32, target=(0.6981, 0.0097)
    ),
    "roberta-large": dict(
        model_name="roberta-large", lr=1e-5, batch_size=16, target=(0.7113, 0.0106)
    ),
    "flang-bert": dict(
        model_name="SALT-NLP/FLANG-BERT", lr=2e-5, batch_size=16, target=(0.6443, 0.0117)
    ),
    "flang-roberta": dict(
        model_name="SALT-NLP/FLANG-RoBERTa",
        lr=2e-5,
        batch_size=8,
        max_len=128,
        target=(0.6348, 0.0021),
    ),
}
