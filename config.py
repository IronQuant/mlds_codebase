from pathlib import Path

# -----------------------------------------------------------------------------------
# RESULTS Paths
# -----------------------------------------------------------------------------------
# Results directory, use google drive if avail, otherwise use local results dir
_ROOT = Path(__file__).parent
for _cand in (Path("/content/drive/MyDrive/thesis"), Path("G:/My Drive/thesis")):
    if _cand.is_dir():
        RESULTS_DIR = _cand
        break
else:
    RESULTS_DIR = _ROOT


# -----------------------------------------------------------------------------------
# Mappings and Seeds
# -----------------------------------------------------------------------------------
# integer label mapping for hawkish/dovish/neutral classification
NAME2INT = {"dovish": 0, "hawkish": 1, "neutral": 2}
INT2NAME = {0: "dovish", 1: "hawkish", 2: "neutral"}

# Random seeds used for each experiment
# These are the seeds from Shah et al. (2023)
SHAH_SEEDS = (5768, 78516, 944601)


# -----------------------------------------------------------------------------------
# Winning Configurations
# -----------------------------------------------------------------------------------
SHAH_PLM = {
    "bert-base-uncased": dict(model_name="bert-base-uncased", lr=2e-5, batch_size=8),
    "bert-large-uncased": dict(model_name="bert-large-uncased", lr=1e-5, batch_size=8),
    "roberta-base": dict(model_name="roberta-base", lr=1e-5, batch_size=32),
    "roberta-large": dict(model_name="roberta-large", lr=1e-5, batch_size=16),
    "flang-bert": dict(model_name="SALT-NLP/FLANG-BERT", lr=2e-5, batch_size=16),
    "flang-roberta": dict(
        model_name="SALT-NLP/FLANG-RoBERTa",
        lr=2e-5,
        batch_size=8,
        max_len=128,
    ),
}
