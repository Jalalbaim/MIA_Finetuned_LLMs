"""
hyperparameters root
"""

from pathlib import Path


SEEDS = [0, 1, 2] 


ROOT          = Path(__file__).parent.resolve()
DATA_DIR      = ROOT / "raw_data" # Raw Enronmail data (if needed)
#CKPT_DIR      = Path("/kaggle/input/datasets/mohamedjalalbaim/gpt-neo-125m-finetuned-enron/checkpoints")
CKPT_DIR      = ROOT / "checkpoints"  # Local checkpoint directory for fine-tuning runs

# Writable location for checkpoints produced by the current run (CKPT_DIR
# above is a read-only Kaggle input dataset mount).
CKPT_OUT_DIR  = ROOT / "checkpoints"
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
#PRETRAINED_CKPT = Path("/kaggle/input/datasets/mohamedjalalbaim/gpt-neo-125m-finetuned-enron/checkpoints/gpt_neo_pretrained")
PRETRAINED_CKPT = CKPT_DIR / "gpt_neo_pretrained"
#  Model

MODEL_NAME = "EleutherAI/gpt-neo-125m"   # HuggingFace model id
#MODEL_NAME = "EleutherAI/gpt-neo-1.3B"
#MODEL_NAME = "EleutherAI/gpt-neo-2.7B"
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
    optimiser        = "AdamW",
    learning_rate    = 5e-5,
    batch_size       = 1,        # actual mini-batch per GPU step (1 for 1.3B model)
    grad_accum_steps = 8,        # effective batch = batch_size × grad_accum_steps = 8
    weight_decay     = 0.01,
    warmup_steps     = 0,
)

# Epoch sweep — primary axis for R1 → R2 regime characterisation (RQ3)
#EPOCH_SWEEP = [1, 2, 3, 5, 10, 15, 20]
EPOCH_SWEEP = [1, 3, 5, 10]

# Default N for the epoch sweep and DP experiments
N_DEFAULT = 2_000

#  LoRA fine-tuning (additive variant, RQ3 secondary axis)

LORA = dict(
    ranks          = [4, 16, 64],
    alpha          = 16,    # scaling; kept fixed across ranks for a clean rank sweep
    dropout        = 0.0,   # no dropout -- deterministic divergence study
    target_modules = None,  # None -> resolved in finetune/train_lora.py for GPT-Neo attention
    n              = 6000,
    seed           = 0,
)

# LoRA updates only a small adapter, so it tolerates (and typically needs) a
# higher learning rate than full fine-tuning (FINETUNE["learning_rate"] = 5e-5).
LORA_LEARNING_RATE = 3e-4

#  MiCA fine-tuning (Minor Component Adaptation, Rüdiger & Raschka arXiv:2604.01694)

MICA = dict(
    ranks          = [4, 16, 64],   # matched to LORA["ranks"] for fair comparison
    alpha          = 16,            # matched to LORA["alpha"]
    dropout        = 0.0,
    target_modules = None,          # None -> resolved to ["q_proj","v_proj"] in finetune/train_mica.py
    n              = 6000,
    seed           = 0,
)

# MiCA's paper uses a higher LR than LoRA (2e-3 vs 5e-4 in their ablation setup).
# Source: Rüdiger & Raschka, arXiv:2604.01694, Table 2.
MICA_LEARNING_RATE = 2e-3

# NOTE on trainable-param budget vs LoRA:
# At equal rank r, MiCA trains only A (shape r × d_in) while LoRA trains both
# A (r × d_in) and B (d_out × r).  MiCA therefore has ~half the trainable
# parameters of LoRA at the same rank.  train_mica.py prints both counts so
# the comparison caveat is visible rather than hidden.

#  Attack signals 

# Fraction of lowest-probability tokens used for Min-K% signal
MIN_K_FRACTION = 0.20    # 20 % as specified in the protocol

# Primary signal for bound-validity analysis (RQ1, RQ2)
PRIMARY_SIGNAL = "ref"   # SRef = log P_ft(x) - log P_pre(x)

# All signals computed at every checkpoint
ALL_SIGNALS = ["loss", "ref", "zlib", "mink"]

#  Rare-token attack-utility experiment (attacks/rare_token_attack.py)
# Restricts the s_ref-style log-ratio attack to a K-fraction subset of positions
# selected by various rules (rare/common under P_pre, random, rare under P_ft)
# and measures attack utility (AUROC / TPR@FPR) -- NOT a bound-tightness test.

RARE_ATTACK = dict(
    k_fractions  = [0.05, 0.10, 0.20, 0.50, 1.00],  # fraction of positions kept per arm
    random_seeds = [0, 1, 2, 3, 4],                  # >=5 draws for the `random` arm baseline
)

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
    "kl_seq",
    "pinsker_seq", "bh_seq",
    "dp_epsilon",        # None for non-DP runs
    "perplexity",        # None for non-DP runs
    "lora_rank",         # None for non-LoRA runs
    "mica_rank",         # None for non-MiCA runs
]