from pathlib import Path

# Directories
_ROOT = Path(__file__).parent
DATA_DIR = _ROOT / "data"

# integer label mapping for hawkish/dovish/neutral classification
NAME2INT = {"dovish": 0, "hawkish": 1, "neutral": 2}
INT2NAME = {0: "dovish", 1: "hawkish", 2: "neutral"}

# Shah et al. (2023) — Trillion Dollar Words
SHAH_DATA_DIR = DATA_DIR / "twd"
SHAH_BENCHMARK_DIR = SHAH_DATA_DIR / "raw"
SHAH_SEEDS = (5768, 78516, 944601)


# # codebase/ -- this file lives in codebase/final/, so go up one
# _ROOT = Path(__file__).parent
# CODEBASE = _ROOT
# print (CODEBASE)

# # raw data sources — one subdir per dataset
# DATA_DIR = _ROOT / "data"

# # result CSVs live only on Google Drive: Colab writes them there, and locally
# # Drive for Desktop exposes the same folder at G: -- one copy, no downloading.
# # Falls back to the repo only if neither mount exists.
# for _cand in (Path("/content/drive/MyDrive/thesis"), Path(r"G:\My Drive\thesis")):
#     if _cand.is_dir():
#         RESULTS_DIR = _cand
#         break
# else:
#     RESULTS_DIR = _ROOT
# print("results ->", RESULTS_DIR)

# # integer label mapping for hawkish/dovish/neutral classification
# LABELS = {0: "dovish", 1: "hawkish", 2: "neutral"}

# # NAME2INT and INT2NAME are the inverse mappings
# NAME2INT = {"dovish": 0, "hawkish": 1, "neutral": 2}
# INT2NAME = {0: "dovish", 1: "hawkish", 2: "neutral"}

# # uniform labeled-corpus schema: column order shared by every source after finalise()
# UNIFORM_COLS = [
#     "sentence",
#     "label",
#     "label_int",
#     "central_bank",
#     "doc_type",
#     "source",
#     "year",
# ]


# # Shah et al. (2023) — Trillion Dollar Words
# # code (notebooks + utils) lives under papers/; all data lives under data/shah_raw/
# SHAH_DIR = _ROOT / "papers" / "trilliondollarwords"
# SHAH_DATA_DIR = DATA_DIR / "twd"
# # per-seed split-combine train/test xlsx
# SHAH_BENCHMARK_DIR = SHAH_DATA_DIR / "raw"
# # 38,169 RoBERTa pseudo-labelled sentences
# SHAH_FULL_CORPUS = SHAH_DATA_DIR / "full_corpus.csv"
# # 944601 == the HuggingFace split
# SHAH_SEEDS = (5768, 78516, 944601)
# # where fine-tuned model checkpoints are saved (one subfolder per model/seed)
# SHAH_MODELS_DIR = SHAH_DIR / "saved_models"

# # The 6-model ablation: 2x2 architecture (BERT vs RoBERTa) x size (base vs large),
# # plus the matched FLANG pair (each base arch further pretrained on the same
# # financial corpus). finbert + modernbert-large dropped (see comments below).
# #
# # HF model id + lr/batch + Shah Table 5 Combined-S target (mean, std ddof=1).
# # lr/batch = winners of the fp32 + class-weighted 9-config grid (mean val
# # macro-F1 over 3 seeds, winners() on grid.csv, 2026-07-29).
# SHAH_PLM = {
#     "bert-base-uncased": dict(
#         model_name="bert-base-uncased", lr=2e-5, batch_size=8, target=(0.6360, 0.0225)
#     ),
#     "bert-large-uncased": dict(
#         model_name="bert-large-uncased", lr=1e-5, batch_size=8, target=(0.6619, 0.0123)
#     ),
#     # DROPPED from the ablation: redundant with flang-bert (both bert-base-sized
#     # financial models) and a different pretraining lineage, which confounds the
#     # domain comparison. FLANG is the consistent treatment across both archs.
#     # "finbert": dict(
#     #     model_name="yiyanghkust/finbert-pretrain",
#     #     lr=1e-5,
#     #     batch_size=16,
#     #     target=(0.6304, 0.0217),
#     # ),
#     "roberta-base": dict(
#         model_name="roberta-base", lr=1e-5, batch_size=32, target=(0.6981, 0.0097)
#     ),
#     # b=16 winner (0.736) sits between collapse-prone cells; b=16 also the
#     # reliable choice on chrono (batch=32 collapsed there)
#     "roberta-large": dict(
#         model_name="roberta-large", lr=1e-5, batch_size=16, target=(0.7113, 0.0106)
#     ),
#     # promoted from the exploratory grid 29 Jul: 0.726 (0.009) val macro-F1 at
#     # 1e-5/b16, second only to roberta-large with half the spread. No Shah
#     # target -- not in their table. b=32 OOMs a 40GB A100 (disentangled
#     # attention), hence the 2x3 grid; needs sentencepiece + protobuf.
#     "deberta-v3-large": dict(
#         model_name="microsoft/deberta-v3-large", lr=1e-5, batch_size=16
#     ),
#     "flang-bert": dict(
#         model_name="SALT-NLP/FLANG-BERT", lr=2e-5, batch_size=16, target=(0.6443, 0.0117)
#     ),
#     # max_position_embeddings is 130, so anything past 128 tokens overruns the
#     # position table and trips a CUDA device-side assert. Other models use 256.
#     "flang-roberta": dict(
#         model_name="SALT-NLP/FLANG-RoBERTa",
#         lr=2e-5,
#         batch_size=8,
#         max_len=128,
#         target=(0.6348, 0.0021),
#     ),
#     # DROPPED from the ablation: tests "newest architecture", tangential to the
#     # size x architecture x domain story, and never beat RoBERTa (0.652 < 0.707).
#     # "modernbert-large": dict(
#     #     model_name="answerdotai/ModernBERT-large", lr=1e-5, batch_size=4
#     # ),
# }
