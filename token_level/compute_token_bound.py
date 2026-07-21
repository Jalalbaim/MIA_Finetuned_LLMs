"""
Token-level Bretagnolle-Huber (BH) bound estimator.

Empirically estimates and validates the token-level BH bound on MIA leakage
by ancestral-sampling sequences from the fine-tuned model P_ft itself (NOT
from the held-out member/eval corpus -- sampling from the training corpus
estimates a different, signed quantity that can go negative; see the
docstring on estimate_config() for why).

Does not touch the existing sequence-level pipeline (kl_estimators/compute_kl.py,
metrics/compute_metrics.py) -- this is a new, independent script that only reads
metrics_all.csv to join the existing 'adv' column for comparison.

Usage:
    python token_level/compute_token_bound.py --pilot
    python token_level/compute_token_bound.py --full
    python token_level/compute_token_bound.py --seed 0 --n 2000 --epoch 1
    python token_level/compute_token_bound.py --seed 0 --n 2000 --epoch 1 --m 4 --T 16 --gen_batch 4 --fwd_batch 4
    token_level/compute_token_bound.py --pilot --m 200 --T 128 --gen_batch 10 --fwd_batch 4
"""

import argparse
import math
import re
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from config import CKPT_DIR, CKPT_OUT_DIR, RESULTS_DIR

TOKEN_LEVEL_DIR = _ROOT / "token_level"

M_SAMPLES_DEFAULT = 200
SEQ_LEN_CAP_DEFAULT = 128

# Sub-batch sizes to bound peak memory on CPU-only runs; lower these if you
# hit an OOM, raise them if you have GPU headroom to spare.
GEN_BATCH_DEFAULT = 10
FWD_BATCH_DEFAULT = 4

_PLAIN_FT_RE = re.compile(r"^gpt_neo_ft_N(?P<n>\d+)_seed(?P<seed>\d+)_epoch(?P<epoch>\d+)$")


# Checkpoint / model loading

def resolve_ckpt_root() -> Path:
    """CKPT_DIR is a hardcoded Kaggle mount path in config.py; it only exists
    when this runs on Kaggle. Locally, checkpoints live under CKPT_OUT_DIR
    instead. Pick whichever actually exists on disk."""
    if CKPT_DIR.exists():
        return CKPT_DIR
    if CKPT_OUT_DIR.exists():
        return CKPT_OUT_DIR
    raise FileNotFoundError(
        f"Neither CKPT_DIR ({CKPT_DIR}) nor CKPT_OUT_DIR ({CKPT_OUT_DIR}) exists."
    )


def load_model(path: Path, device: torch.device) -> AutoModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(str(path)).to(device)
    model.eval()
    return model


def discover_full_finetune_grid(ckpt_root: Path) -> list[dict]:
    """Scan ckpt_root for plain full-fine-tuning checkpoints (excludes
    dp/lora/mica-prefixed dirs and gpt_neo_pretrained itself)."""
    found = []
    for p in sorted(ckpt_root.iterdir()):
        if not p.is_dir():
            continue
        m = _PLAIN_FT_RE.match(p.name)
        if m:
            found.append({
                "n": int(m["n"]), "seed": int(m["seed"]), "epoch": int(m["epoch"]),
            })
    found.sort(key=lambda c: (c["seed"], c["n"], c["epoch"]))
    return found


# Ancestral sampling from P_ft

@torch.no_grad()
def generate_sequences(
    model_ft: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    m: int,
    T: int,
    device: torch.device,
    gen_seed: int,
    gen_batch: int = GEN_BATCH_DEFAULT,
) -> torch.Tensor:
    """Ancestral-sample m sequences of exactly T tokens from P_ft, temperature
    1.0, no top-k/top-p/greedy truncation. min_new_tokens=T suppresses the EOS
    logit until length T is reached so every sample has the same fixed length
    (this is the only deviation from "pure" per-step softmax -- required
    because we need fixed T, not variable-length EOS-terminated samples)."""
    bos_id = tokenizer.eos_token_id
    torch.manual_seed(gen_seed)

    chunks = []
    remaining = m
    while remaining > 0:
        b = min(gen_batch, remaining)
        input_ids = torch.full((b, 1), bos_id, dtype=torch.long, device=device)
        out = model_ft.generate(
            input_ids=input_ids,
            do_sample=True,
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            min_new_tokens=T,
            max_new_tokens=T,
            pad_token_id=tokenizer.pad_token_id,
        )
        chunks.append(out)
        remaining -= b

    return torch.cat(chunks, dim=0)  # (m, 1+T)


# Exact per-position KL (teacher forcing, full vocab)

@torch.no_grad()
def per_position_kl_and_logprobs(
    seqs: torch.Tensor,
    model_ft: AutoModelForCausalLM,
    model_pre: AutoModelForCausalLM,
    device: torch.device,
    fwd_batch: int = FWD_BATCH_DEFAULT,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One forward pass per model over the generated sequences (teacher
    forcing, no generation). Returns per-position (m, T) tensors:
      kl_t     = KL(P_ft(.|x_<t) || P_pre(.|x_<t)), exact vocab sum
      lp_ft_t  = log P_ft(x_t | x_<t)   for the realized token
      lp_pre_t = log P_pre(x_t | x_<t)  for the realized token
    """
    m, L = seqs.shape
    T = L - 1
    kl_t_all = torch.zeros(m, T)
    lp_ft_all = torch.zeros(m, T)
    lp_pre_all = torch.zeros(m, T)

    for start in range(0, m, fwd_batch):
        end = min(start + fwd_batch, m)
        batch = seqs[start:end].to(device)  # (b, L)

        logits_ft = model_ft(input_ids=batch).logits   # (b, L, V)
        logits_pre = model_pre(input_ids=batch).logits

        shifted_ft = logits_ft[:, :-1, :].float()   # (b, T, V) predicts batch[:,1:]
        shifted_pre = logits_pre[:, :-1, :].float()
        labels = batch[:, 1:]                        # (b, T)

        log_p = F.log_softmax(shifted_ft, dim=-1)
        log_q = F.log_softmax(shifted_pre, dim=-1)
        p = log_p.exp()

        kl = (p * (log_p - log_q)).nan_to_num(0.0).sum(dim=-1)          # (b, T)
        lp_ft = log_p.gather(-1, labels.unsqueeze(-1)).squeeze(-1)      # (b, T)
        lp_pre = log_q.gather(-1, labels.unsqueeze(-1)).squeeze(-1)     # (b, T)

        kl_t_all[start:end] = kl.cpu()
        lp_ft_all[start:end] = lp_ft.cpu()
        lp_pre_all[start:end] = lp_pre.cpu()

    return kl_t_all, lp_ft_all, lp_pre_all


# Per-configuration estimator

def estimate_config(
    seed: int,
    n_members: int,
    epoch: int,
    m: int,
    T: int,
    gen_seed: int,
    ckpt_root: Path,
    tokenizer: AutoTokenizer,
    model_pre: AutoModelForCausalLM,
    device: torch.device,
    per_position_out: Path | None = None,
) -> dict | None:
    """KL(P_ft || P_pre) is an expectation UNDER P_ft, so samples MUST come
    from ancestral sampling of the fine-tuned model itself -- not from the
    member/eval corpus. Sampling real corpus text instead would estimate
    E_{x~corpus}[log P_ft(x) - log P_pre(x)], a signed difference of two
    divergences (KL(corpus||P_ft) terms) that can go negative and is not
    the quantity the token-level BH theorem bounds."""
    ckpt_path = ckpt_root / f"gpt_neo_ft_N{n_members}_seed{seed}_epoch{epoch}"
    if not ckpt_path.exists():
        print(f"  [skip] Checkpoint not found: {ckpt_path}")
        return None

    print(f"\n{'='*70}")
    print(f"  Config: seed={seed}  n_members={n_members}  epoch={epoch}")
    print(f"  Checkpoint: {ckpt_path.name}")
    t0 = time.time()

    print("  Loading fine-tuned checkpoint ...")
    model_ft = load_model(ckpt_path, device)

    print(f"  Ancestral-sampling m={m} sequences, T={T} tokens, temperature=1.0 ...")
    seqs = generate_sequences(model_ft, tokenizer, m, T, device, gen_seed)

    print("  Teacher-forcing both models over generated samples (exact per-position KL) ...")
    kl_t_all, lp_ft_all, lp_pre_all = per_position_kl_and_logprobs(seqs, model_ft, model_pre, device)

    # KL is an exact vocab sum -> mathematically >= 0. Clamp only for float
    # noise; anything beyond noise means a real bug, not something to hide.
    kl_t_clamped = kl_t_all.clamp(min=0.0)
    neg_mag = (kl_t_all - kl_t_clamped).abs().max().item()
    if neg_mag > 1e-4:
        print(f"  WARNING: KL_t negative beyond float noise (max |neg| = {neg_mag:.6f}); clamped to 0.")

    bh_t_all = torch.sqrt(1.0 - torch.exp(-kl_t_clamped))  # (m, T), per-position BH transform

    # Token-level estimator: sum over positions THEN average over samples
    # (transform-then-average, per spec -- summing after transform, never
    # transform the summed KL).
    kl_tok_per_sample = kl_t_clamped.sum(dim=1)   # (m,)
    bh_tok_per_sample = bh_t_all.sum(dim=1)       # (m,)
    kl_tok_hat = kl_tok_per_sample.mean().item()
    bh_tok_hat = bh_tok_per_sample.mean().item()

    # Sequence-level estimator, from the SAME samples: sum of realized-token
    # log-ratios over the whole sequence, then average, THEN transform once.
    kl_seq_per_sample = lp_ft_all.sum(dim=1) - lp_pre_all.sum(dim=1)  # (m,)
    kl_seq_hat_raw = kl_seq_per_sample.mean().item()
    kl_seq_hat = kl_seq_hat_raw
    if kl_seq_hat < 0:
        print(f"  WARNING: kl_seq_hat is negative ({kl_seq_hat_raw:.4f}) -- estimator noise, clipping to 0.")
        kl_seq_hat = 0.0

    bh_seq_hat = math.sqrt(1.0 - math.exp(-kl_seq_hat))
    pinsker_seq_hat = math.sqrt(kl_seq_hat / 2.0)

    elapsed = time.time() - t0

    # Sanity checks (printed for every config, never skipped)
    print(f"\n  --- Sanity checks (seed={seed} n={n_members} epoch={epoch}) ---")

    ratio = kl_tok_hat / kl_seq_hat_raw if kl_seq_hat_raw not in (0, 0.0) else float("nan")
    ratio_ok = (not math.isnan(ratio)) and (0.8 <= ratio <= 1.2)
    print(f"    kl_tok_hat={kl_tok_hat:.4f}  kl_seq_hat(raw)={kl_seq_hat_raw:.4f}  ratio={ratio:.4f}"
          f"  [{'OK' if ratio_ok else 'FLAG: >20% off — same estimand should roughly agree'}]")

    for name, val in (("bh_seq_hat", bh_seq_hat), ("bh_tok_hat", bh_tok_hat)):
        in_range = 0.0 <= val < 1.0
        print(f"    {name}={val:.4f}  in [0,1): {'PASS' if in_range else 'FAIL — STOP, this indicates a bug'}")
        if not in_range:
            raise RuntimeError(f"{name}={val} outside [0,1) — unclipped negative KL or overflow, not silently clamping.")

    direction_ok = bh_seq_hat <= bh_tok_hat
    print(f"    bh_seq_hat <= bh_tok_hat: {bh_seq_hat:.4f} <= {bh_tok_hat:.4f}  "
          f"[{'expected direction' if direction_ok else 'violated — note but not necessarily a bug'}]")

    print(f"    kl_seq_hat={kl_seq_hat:.4f}  bh_seq_hat={bh_seq_hat:.4f}  "
          f"pinsker_seq_hat={pinsker_seq_hat:.4f}  kl_tok_hat={kl_tok_hat:.4f}  bh_tok_hat={bh_tok_hat:.4f}")
    print(f"    elapsed: {elapsed:.1f}s")

    if per_position_out is not None:
        rows = []
        for i in range(m):
            for t in range(T):
                rows.append({
                    "sample_index": i,
                    "position_t": t + 1,
                    "kl_t": kl_t_clamped[i, t].item(),
                    "bh_t": bh_t_all[i, t].item(),
                })
        pd.DataFrame(rows).to_csv(per_position_out, index=False)
        print(f"    Saved per-position long-format file -> {per_position_out}")

    del model_ft
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "seed": seed,
        "n_members": n_members,
        "epochs": epoch,
        "m_samples": m,
        "seq_len_cap": T,
        "kl_seq_hat": kl_seq_hat,
        "bh_seq_hat": bh_seq_hat,
        "pinsker_seq_hat": pinsker_seq_hat,
        "kl_tok_hat": kl_tok_hat,
        "bh_tok_hat": bh_tok_hat,
    }


# adv join + final adv <= bound check

def join_adv_existing(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    metrics_path = RESULTS_DIR / "metrics_all.csv"
    if not metrics_path.exists():
        print(f"  [warn] {metrics_path} not found -- adv_ref_existing left as NaN.")
        df["adv_ref_existing"] = float("nan")
        return df

    metrics = pd.read_csv(metrics_path)
    ref = metrics[
        (metrics["signal"] == "ref")
        & metrics["lora_rank"].isna()
        & metrics["mica_rank"].isna()
        & metrics["dp_epsilon"].isna()
    ][["seed", "n_members", "epochs", "adv"]].rename(columns={"adv": "adv_ref_existing"})

    df = df.merge(ref, on=["seed", "n_members", "epochs"], how="left")

    print("\n  --- adv_ref_existing <= bh_seq_hat and bh_tok_hat ---")
    for _, r in df.iterrows():
        if pd.isna(r["adv_ref_existing"]):
            print(f"    seed={r['seed']} n={r['n_members']} epoch={r['epochs']}: "
                  f"no matching adv row in metrics_all.csv (signal='ref')")
            continue
        ok_seq = r["adv_ref_existing"] <= r["bh_seq_hat"] + 1e-6
        ok_tok = r["adv_ref_existing"] <= r["bh_tok_hat"] + 1e-6
        status = "PASS" if (ok_seq and ok_tok) else "FAIL"
        print(f"    [{status}] seed={r['seed']} n={r['n_members']} epoch={r['epochs']}: "
              f"adv={r['adv_ref_existing']:.4f}  bh_seq_hat={r['bh_seq_hat']:.4f}  "
              f"bh_tok_hat={r['bh_tok_hat']:.4f}"
              + ("" if status == "PASS" else "  <-- VIOLATION"))

    return df


# Entry point

def main() -> None:
    parser = argparse.ArgumentParser(description="Token-level BH bound estimator (full fine-tuning only).")
    parser.add_argument("--pilot", action="store_true",
                        help="Run the two-config pilot: N2000 seed0 epoch1 and epoch20.")
    parser.add_argument("--full", action="store_true",
                        help="Run every full-fine-tuning checkpoint found on disk.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--m", type=int, default=M_SAMPLES_DEFAULT)
    parser.add_argument("--T", type=int, default=SEQ_LEN_CAP_DEFAULT)
    parser.add_argument("--gen_seed", type=int, default=0)
    parser.add_argument("--gen_batch", type=int, default=GEN_BATCH_DEFAULT)
    parser.add_argument("--fwd_batch", type=int, default=FWD_BATCH_DEFAULT)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_root = resolve_ckpt_root()
    print(f"Device      : {device}")
    print(f"Ckpt root   : {ckpt_root}")
    print(f"m, T        : {args.m}, {args.T}")

    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_root / "gpt_neo_pretrained"))
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading pretrained reference model ...")
    model_pre = load_model(ckpt_root / "gpt_neo_pretrained", device)

    if args.pilot:
        configs = [
            {"seed": 0, "n": 2000, "epoch": 1},
            {"seed": 0, "n": 2000, "epoch": 20},
        ]
    elif args.full:
        configs = discover_full_finetune_grid(ckpt_root)
        print(f"Discovered {len(configs)} full-fine-tuning checkpoint(s) on disk.")
    else:
        if args.seed is None or args.n is None or args.epoch is None:
            parser.error("Specify --pilot, --full, or all of --seed/--n/--epoch.")
        configs = [{"seed": args.seed, "n": args.n, "epoch": args.epoch}]

    TOKEN_LEVEL_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for cfg in configs:
        per_pos_out = None
        if cfg["n"] == 2000 and cfg["seed"] == 0 and cfg["epoch"] == 20:
            per_pos_out = TOKEN_LEVEL_DIR / "token_level_bounds_per_position.csv"

        result = estimate_config(
            seed=cfg["seed"], n_members=cfg["n"], epoch=cfg["epoch"],
            m=args.m, T=args.T, gen_seed=args.gen_seed,
            ckpt_root=ckpt_root, tokenizer=tokenizer, model_pre=model_pre, device=device,
            per_position_out=per_pos_out,
        )
        if result is not None:
            rows.append(result)

    if not rows:
        print("No results produced -- nothing to write.")
        return

    df = join_adv_existing(rows)
    out_cols = ["seed", "n_members", "epochs", "m_samples", "seq_len_cap",
                "kl_seq_hat", "bh_seq_hat", "pinsker_seq_hat",
                "kl_tok_hat", "bh_tok_hat", "adv_ref_existing"]
    df = df[out_cols]

    out_path = TOKEN_LEVEL_DIR / "token_level_bounds.csv"
    if out_path.exists():
        existing = pd.read_csv(out_path)
        key_cols = ["seed", "n_members", "epochs"]
        existing = existing[~existing.set_index(key_cols).index.isin(df.set_index(key_cols).index)]
        df = pd.concat([existing, df], ignore_index=True)
    df = df.sort_values(["seed", "n_members", "epochs"]).reset_index(drop=True)
    df.to_csv(out_path, index=False)
    print(f"\nWrote {len(df)} row(s) -> {out_path}")


if __name__ == "__main__":
    main()
