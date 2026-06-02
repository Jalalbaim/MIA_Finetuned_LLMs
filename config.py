"""
hyperparameters root
"""

from pathlib import Path


SEEDS = [0, 1, 2] 


ROOT          = Path(__file__).parent.resolve()
DATA_DIR      = ROOT / "raw_data" # Raw Enronmail data (if needed)
CKPT_DIR      = ROOT / "checkpoints"
RESULTS_DIR   = ROOT / "results"
FIGURES_DIR   = ROOT / "figures"
LOG_DIR       = ROOT / "logs"

# raw pool written by the data pipeline (Step 1)
POOL_FILE     = DATA_DIR / "pool.jsonl"

# split files written by membership assignment (Step 2)
# filenames are parameterised by N and seed at runtime, e.g.:
#   members_N2000_seed0.jsonl
#   nonmembers_seed0.jsonl   (always 2000, fixed across N values)
#   eval_N2000_seed0.jsonl
SPLIT_TEMPLATE = DATA_DIR / "{split}_N{n}_seed{seed}.jsonl"

# pretrained reference model saved once (Step 3)
PRETRAINED_CKPT = CKPT_DIR / "gpt_neo_pretrained"

#  Model

MODEL_NAME = "EleutherAI/gpt-neo-125m"   # HuggingFace model id
MAX_SEQ_LEN = 1024                         # truncation length (tokens)
MIN_SEQ_LEN = 50                           # minimum body length kept in pool

#  Pool / corpus

POOL_SIZE       = 10_000   # total sequences drawn from Enron
N_NONMEMBERS    = 2_000    # fixed non-member set size across all sweeps
N_EVAL          = None     # derived: POOL_SIZE - N_members - N_NONMEMBERS
                           # (set dynamically per experiment; see split logic)

#  Corpus-size sweep (RQ3 secondary axis) 

CORPUS_SIZES = [500, 2_000, 6_000]   # N_members values; N=6000 replaces N=8000

#  Fine-tuning hyperparameters

FINETUNE = dict(
    optimiser    = "AdamW",
    learning_rate= 5e-5,
    batch_size   = 8,
    weight_decay = 0.01,       # AdamW default; not varied
    warmup_steps = 0,          # no warmup (small dataset regime)
)

# Epoch sweep — primary axis for R1 → R2 regime characterisation (RQ3)
EPOCH_SWEEP = [1, 5, 10, 20]

# Default N for the epoch sweep and DP experiments
N_DEFAULT = 2_000

#  Attack signals 

# Fraction of lowest-probability tokens used for Min-K% signal
MIN_K_FRACTION = 0.20    # 20 % as specified in the protocol

# Primary signal for bound-validity analysis (RQ1, RQ2)
PRIMARY_SIGNAL = "ref"   # SRef = log P_ft(x) - log P_pre(x)

# All signals computed at every checkpoint
ALL_SIGNALS = ["loss", "ref", "zlib", "mink"]

#  KL estimator 

# Number of sequences from E used to estimate KL
# None = use all available sequences in E
KL_EVAL_SIZE = None

#  Differential privacy (RQ4)

DP = dict(
    epsilon_values  = [1, 4, 8],      # privacy budgets
    delta_fn        = lambda n: 1/n,  # delta = 1/N (depends on corpus size)
    max_grad_norm   = 1.0,            # gradient clipping norm
    accounting      = "rdp",          # RDP composition (Opacus default)
    epochs          = 3,              # fixed; chosen post-hoc (see protocol §1.3)
    n               = N_DEFAULT,      # N=2000 for DP experiments
)

#  Metrics

# FPR thresholds for TPR@FPR reporting
FPR_THRESHOLDS = [0.01, 0.001]    # 1% and 0.1%

# Number of histogram bins for empirical TV estimation
TV_N_BINS = 500

#  Pilot / decision gate (Week 1)
R2_KL_THRESHOLD = 2.0

PILOT_EPOCHS = [1, 5, 20]

#  Figures 

FIG_DPI    = 150
FIG_FORMAT = "png"

FIGURE_PATHS = {
    1: FIGURES_DIR / "fig1_tv_pinsker_bh_vs_kl_epoch_sweep.png",
    2: FIGURES_DIR / "fig2_pinsker_vs_bh_regime.png",
    3: FIGURES_DIR / "fig3_token_vs_sequence_bounds.png",
    4: FIGURES_DIR / "fig4_dp_epsilon_vs_kl_adv.png",
}

#Results CSV columns

RESULTS_COLUMNS = [
    "seed", "n_members", "epochs", "signal",
    "auroc", "tpr_at_fpr_1pct", "tpr_at_fpr_01pct",
    "adv", "tv_empirical",
    "kl_seq", "kl_tok",
    "pinsker_seq", "bh_seq",
    "pinsker_tok", "bh_tok",
    "dp_epsilon",        # None for non-DP runs
    "perplexity",        # None for non-DP runs
]