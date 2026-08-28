from pathlib import Path

# -----------------------------------------------------------------------------------
# RESULTS Paths
# -----------------------------------------------------------------------------------
# Results directory, use google drive if avail, otherwise use local results dir

# Parent Directory
_ROOT = Path(__file__).parent

# Check for Google Drive or local results directory
for _cand in (Path("/content/drive/MyDrive/thesis"), Path("G:/My Drive/thesis")):
    if _cand.is_dir():
        RESULTS_DIR = _cand
        break
else:
    RESULTS_DIR = _ROOT


# -----------------------------------------------------------------------------------
# Mappings and Seeds
# -----------------------------------------------------------------------------------
NAME2INT = {"dovish": 0, "hawkish": 1, "neutral": 2}
INT2NAME = {0: "dovish", 1: "hawkish", 2: "neutral"}

# These are the seeds from Shah et al. (2023)
SHAH_SEEDS = (5768, 78516, 944601)


# -----------------------------------------------------------------------------------
# Winning Configurations (result from hyperparameter search)
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

# -----------------------------------------------------------------------------------
# Adaptation configuration (continued MLM)
# -----------------------------------------------------------------------------------
# Gururangan et al. (2020) Table 13. They use two settings: an effective batch of
# 2048 at lr 5e-4 for DAPT, and 256 at 1e-4 for TAPT on tasks under 5K examples,
# both reached through gradient accumulation. We apply the TAPT setting to every
# arm, so the arms stay comparable to each other rather than each matching a
# different external configuration. max_len matches our fine-tuning; theirs is
# unstated. Everything else follows the paper.
APT = dict(
    lr=1e-4,
    batch_size=32,
    accum_steps=8,
    max_len=256,
    mlm_probability=0.15,
    warmup=0.06,
    adam_eps=1e-6,
    adam_betas=(0.9, 0.98),
)

# adaptation budget per arm. DAPT is one pass over a large pool; TAPT is 100
APT_EPOCHS = {
    "dapt": 1,
    "tapt": 100,
    "curated-tapt": 12,
}


# -----------------------------------------------------------------------------------
# Idioms
# -----------------------------------------------------------------------------------

IDIOMS = [
    ("Accommodative [MASK] policy", "monetary"),
    ("Asset [MASK] program", "purchase"),
    ("Balance [MASK] payments", "of"),
    ("Bank [MASK] International Settlements", "for"),
    ("Basel [MASK] on Banking Supervision", "committee"),
    ("Bretton [MASK] system", "woods"),
    ("Capital [MASK] ratio", "adequacy"),
    ("Central [MASK] (CCP)", "counterparties"),
    ("Central [MASK] balance sheet", "bank"),
    ("Central [MASK] digital currency", "bank"),
    ("Collateralized [MASK] obligation", "debt"),
    ("Committee [MASK] Payments and Market Infrastructure", "on"),
    ("Commodity [MASK] index", "price"),
    ("Contractionary [MASK] policy", "monetary"),
    ("Core [MASK] price index", "consumer"),
    ("Countercyclical [MASK] buffer", "capital"),
    ("Credit [MASK] swap", "default"),
    ("Cross-Currency [MASK] Swaps", "basis"),
    ("Currency [MASK] of reserves", "composition"),
    ("Decentralized [MASK] (DeFi)", "finance"),
    ("Distributed [MASK] technology", "ledger"),
    ("Domestic [MASK] important bank", "systemically"),
    ("Effective [MASK] bound", "lower"),
    ("Effective [MASK] funds rate", "federal"),
    ("Effective [MASK] rate", "exchange"),
    ("Efficient [MASK] hypothesis", "market"),
    ("Emerging [MASK] and developing economies", "market"),
    ("Emerging [MASK] economies", "market"),
    ("European [MASK] Bank", "central"),
    ("Exchange [MASK] pass-through", "rate"),
    ("Exchange [MASK] regime", "rate"),
    ("Expansionary [MASK] policy", "monetary"),
    ("Financial [MASK] board", "stability"),
    ("Fixed [MASK] rate", "exchange"),
    ("Floating [MASK] rate", "exchange"),
    ("Foreign [MASK] intervention", "exchange"),
    ("Foreign [MASK] investment", "direct"),
    ("Foreign [MASK] reserves", "exchange"),
    ("Global [MASK] important banks", "systemically"),
    ("Gross [MASK] debt", "external"),
    ("Gross [MASK] product", "domestic"),
    ("Interbank [MASK] rate", "offered"),
    ("Interest [MASK] on deposit facility", "rate"),
    ("Interest [MASK] on excess reserves", "rate"),
    ("Interest [MASK] parity", "rate"),
    ("Interest [MASK] risk", "rate"),
    ("Interest [MASK] swap", "rate"),
    ("Interest [MASK] targeting", "rate"),
    ("International [MASK] Fund", "monetary"),
    ("International [MASK] of Insurance Supervisors", "association"),
    ("International [MASK] position", "investment"),
    ("Inverted [MASK] curve", "yield"),
    ("Labor [MASK] participation rate", "force"),
    ("Lender [MASK] last resort", "of"),
    ("Liquidity [MASK] test", "stress"),
    ("London [MASK] Offered Rate", "interbank"),
    ("Long-term [MASK] operation", "refinancing"),
    ("Long-term [MASK] rates", "interest"),
    ("Macroprudential [MASK] measures", "policy"),
    ("Main [MASK] operation", "refinancing"),
    ("Marginal [MASK] facility", "lending"),
    ("Monetary [MASK] committee", "policy"),
    ("Monetary [MASK] framework", "policy"),
    ("Monetary [MASK] stance", "policy"),
    ("Monetary [MASK] transmission", "policy"),
    ("Natural [MASK] of interest", "rate"),
    ("Natural [MASK] of unemployment", "rate"),
    ("Negative [MASK] rates", "interest"),
    ("Net [MASK] debt", "external"),
    ("Neutral [MASK] policy", "monetary"),
    ("Nominal [MASK] rate", "exchange"),
    ("Non-accelerating [MASK] rate of unemployment", "inflation"),
    ("Non-bank [MASK] institution", "financial"),
    ("Non-bank [MASK] intermediation", "credit"),
    ("Open [MASK] operations", "market"),
    ("Overnight [MASK] facility", "deposit"),
    ("Overnight [MASK] swap", "index"),
    ("Pegged [MASK] rate", "exchange"),
    ("Producer [MASK] index", "price"),
    ("Purchasing [MASK] parity", "power"),
    ("Real [MASK] exchange rate", "effective"),
    ("Real [MASK] rate", "exchange"),
    ("Safe [MASK] assets", "haven"),
    ("Secured [MASK] financing rate", "overnight"),
    ("Short-term [MASK] rates", "interest"),
    ("Sovereign [MASK] crisis", "debt"),
    ("Sovereign [MASK] fund", "wealth"),
    ("Special [MASK] Rights", "drawing"),
    ("Special [MASK] vehicle", "purpose"),
    ("Systemically [MASK] financial institution", "important"),
    ("Targeted [MASK] refinancing operations", "longer"),
    ("Terms [MASK] trade", "of"),
    ("Tier [MASK] capital", "1"),
    ("Tight [MASK] policy", "monetary"),
    ("Too [MASK] to fail", "big"),
    ("Unconventional [MASK] policy", "monetary"),
    ("Value [MASK] Risk", "at"),
    ("Velocity [MASK] money", "of"),
    ("Yield [MASK] control", "curve"),
    ("Zero [MASK] bound", "lower"),
]
