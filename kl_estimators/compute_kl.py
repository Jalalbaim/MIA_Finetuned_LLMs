"""

  KL̂_seq — sequence-level Monte Carlo estimator (mean of per-sequence log-ratios)
  KL̂_tok — exact token-level KL (averaged over E, exact over the full vocabulary)

Usage:
    python kl_estimators/compute_kl.py
    python kl_estimators/compute_kl.py --seed 0 --n 2000
    python kl_estimators/compute_kl.py --seed 0 --n 2000 --epoch 5
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "data"))

from config import (
    CKPT_DIR,
    CORPUS_SIZES,
    EPOCH_SWEEP,
    KL_EVAL_SIZE,
    MAX_SEQ_LEN,
    R2_KL_THRESHOLD,
    RESULTS_DIR,
    SEEDS,
)
from membership_assignment import load_split


@torch.no_grad()
def get_full_logprobs(
    text: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    
    input_ids = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LEN,
    )["input_ids"].to(device)                         # (1, seq_len)

    out_logits = model(input_ids=input_ids).logits[0]  # (seq_len, V)

    shifted_logits = out_logits[:-1].float()           # (T, V), float32
    shift_labels   = input_ids[0, 1:]                  # (T,)

    log_probs = F.log_softmax(shifted_logits, dim=-1)  # (T, V)
    token_log_probs = log_probs[
        torch.arange(len(shift_labels), device=device), shift_labels
    ]                                                  # (T,)

    # sum, not mean — KL estimator requires full log-likelihood
    return token_log_probs.sum().item(), token_log_probs, shifted_logits


# Estimator 1 — Sequence-level KL

def compute_kl_seq(
    eval_sequences: list[dict],
    model_ft: AutoModelForCausalLM,
    model_pre: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
) -> tuple[float, list[float], float, float]:

    contributions: list[float] = []
    lp_ft_list:    list[float] = []
    lp_pre_list:   list[float] = []

    for rec in eval_sequences:
        text = rec["text"]
        lp_ft,  _, _ = get_full_logprobs(text, model_ft,  tokenizer, device)
        lp_pre, _, _ = get_full_logprobs(text, model_pre, tokenizer, device)
        contributions.append(lp_ft - lp_pre)
        lp_ft_list.append(lp_ft)
        lp_pre_list.append(lp_pre)

    kl_seq      = sum(contributions) / len(contributions)
    mean_lp_ft  = sum(lp_ft_list)   / len(lp_ft_list)
    mean_lp_pre = sum(lp_pre_list)  / len(lp_pre_list)

    if kl_seq < 0:
        print(
            f"\n{'!'*60}\n"
            f"  WARNING: KL_seq is negative ({kl_seq:.4f}) — possible distribution\n"
            f"  shift in E or insufficient memorisation. Check eval set construction.\n"
            f"{'!'*60}\n"
        )

    return kl_seq, contributions, mean_lp_ft, mean_lp_pre


# Estimator 2 — token-level KL


def compute_kl_tok(
    eval_sequences: list[dict],
    model_ft: AutoModelForCausalLM,
    model_pre: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
) -> float:
    
    kl_values: list[float] = []

    for rec in eval_sequences:
        text = rec["text"]
        _, _, logits_ft  = get_full_logprobs(text, model_ft,  tokenizer, device)
        _, _, logits_pre = get_full_logprobs(text, model_pre, tokenizer, device)

        log_p = F.log_softmax(logits_ft,  dim=-1)  # (T, V)
        log_q = F.log_softmax(logits_pre, dim=-1)  # (T, V)
        p     = log_p.exp()                         # (T, V)

        # KL(p ∥ q) = Σ p·(log p − log q); nan_to_num handles 0·(−∞) → 0
        kl_x = (p * (log_p - log_q)).nan_to_num(0.0).sum().item()
        kl_values.append(kl_x)

    return sum(kl_values) / len(kl_values)



# Bound computation


def compute_bounds(kl_seq: float, kl_tok: float) -> dict:
    return {
        "pinsker_seq": math.sqrt(max(kl_seq, 0.0) / 2.0),
        "bh_seq":      math.sqrt(1.0 - math.exp(-max(kl_seq, 0.0))),
        "pinsker_tok": math.sqrt(max(kl_tok, 0.0) / 2.0),
        "bh_tok":      math.sqrt(1.0 - math.exp(-max(kl_tok, 0.0))),
    }



# Per-checkpoint driver


def _load_model(ckpt_path: Path, device: torch.device) -> AutoModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(str(ckpt_path)).to(device)
    model.eval()
    return model


def compute_kl_for_checkpoint(
    seed: int,
    n_members: int,
    epoch: int,
    device: torch.device,
    model_pre: AutoModelForCausalLM | None = None,
    tokenizer: AutoTokenizer | None = None,
) -> dict | None:

    ckpt_name = f"gpt_neo_ft_N{n_members}_seed{seed}_epoch{epoch}"
    ckpt_path = CKPT_DIR / ckpt_name

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"kl_N{n_members}_seed{seed}_epoch{epoch}.json"

    if out_path.exists():
        print(f"  [skip] {out_path.name} already exists.")
        with out_path.open(encoding="utf-8") as fh:
            return json.load(fh)

    if not ckpt_path.exists():
        print(f"  [skip] Checkpoint not found: {ckpt_path}")
        return None

    print(f"\n{'='*60}")
    print(f"  Checkpoint : {ckpt_name}")
    print(f"  Output     : {out_path.name}")

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(str(CKPT_DIR / "gpt_neo_pretrained"))
        tokenizer.pad_token = tokenizer.eos_token

    if model_pre is None:
        print("  Loading pretrained reference ...")
        model_pre = _load_model(CKPT_DIR / "gpt_neo_pretrained", device)

    print("  Loading fine-tuned model ...")
    model_ft = _load_model(ckpt_path, device)

    _, _, eval_set = load_split(seed, n_members)
    eval_sequences = eval_set if KL_EVAL_SIZE is None else eval_set[:KL_EVAL_SIZE]
    n_eval = len(eval_sequences)
    print(f"  Eval set E : {n_eval:,} sequences")

    if n_eval == 0:
        print("  ERROR: eval set is empty — cannot estimate KL. Skipping.")
        del model_ft
        return None

    t0 = time.time()

    # Estimator 1 — sequence-level
    print("  Estimator 1 — sequence-level ...")
    kl_seq, _, mean_lp_ft, mean_lp_pre = compute_kl_seq(
        eval_sequences, model_ft, model_pre, tokenizer, device
    )

    # Estimator 2 — token-level
    print("  Estimator 2 — token-level ...")
    kl_tok = compute_kl_tok(
        eval_sequences, model_ft, model_pre, tokenizer, device
    )

    print("Bounds ...")
    bounds      = compute_bounds(kl_seq, kl_tok)
    pinsker_seq = bounds["pinsker_seq"]
    bh_seq      = bounds["bh_seq"]
    elapsed     = time.time() - t0

    # --- Sanity checks ---
    checks = {
        "kl_seq > 0":                       kl_seq > 0,
        "kl_tok > kl_seq":                  kl_tok > kl_seq,
        "bh_seq <= 1.0":                    bh_seq <= 1.0,
        "mean_seq_lp_ft > mean_seq_lp_pre": mean_lp_ft > mean_lp_pre,
    }
    # R2 regime: KL exceeds threshold where Pinsker becomes vacuous (kl_seq > 2 → pinsker > 1)
    regime  = "R2 regime" if kl_seq > R2_KL_THRESHOLD else "R1 regime"
    vacuous = pinsker_seq > 1.0

    print(
        f"\n  epoch={epoch} | kl_seq={kl_seq:.4f} | kl_tok={kl_tok:.4f} | "
        f"pinsker_seq={pinsker_seq:.4f} | bh_seq={bh_seq:.4f} | "
        f"vacuous={'YES' if vacuous else 'NO'}"
    )
    print(f"  {regime}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print("  Sanity checks:")
    for name, passed in checks.items():
        print(f"    {'PASS' if passed else 'FAIL'}  {name}")

    result = {
        "seed":                  seed,
        "n_members":             n_members,
        "epoch":                 epoch,
        "kl_seq":                kl_seq,
        "kl_tok":                kl_tok,
        "pinsker_seq":           pinsker_seq,
        "bh_seq":                bh_seq,
        "pinsker_tok":           bounds["pinsker_tok"],
        "bh_tok":                bounds["bh_tok"],
        "n_eval_sequences":      n_eval,
        "mean_seq_logprob_ft":   mean_lp_ft,
        "mean_seq_logprob_pre":  mean_lp_pre,
    }

    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    del model_ft
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate KL(P_ft ∥ P_pre) for fine-tuned GPT-Neo checkpoints."
    )
    parser.add_argument("--seed",  type=int, default=None,
                        help="Single seed to run (default: all SEEDS from config)")
    parser.add_argument("--n",     type=int, default=None,
                        help="Single N_members to run (default: all CORPUS_SIZES)")
    parser.add_argument("--epoch", type=int, default=None,
                        help="Single epoch to run (default: all EPOCH_SWEEP)")
    args = parser.parse_args()

    seeds  = [args.seed]  if args.seed  is not None else SEEDS
    ns     = [args.n]     if args.n     is not None else CORPUS_SIZES
    epochs = [args.epoch] if args.epoch is not None else EPOCH_SWEEP

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device    : {device}")
    print(f"Seeds     : {seeds}")
    print(f"N values  : {ns}")
    print(f"Epochs    : {epochs}")

    t_wall = time.time()

    for seed in seeds:
        for n in ns:
            print(f"\n{'#'*60}")
            print(f"  seed={seed}  N={n:,}  — loading shared pretrained reference ...")

            tokenizer = AutoTokenizer.from_pretrained(str(CKPT_DIR / "gpt_neo_pretrained"))
            tokenizer.pad_token = tokenizer.eos_token
            model_pre = _load_model(CKPT_DIR / "gpt_neo_pretrained", device)

            epoch_results: list[dict] = []

            for epoch in epochs:
                result = compute_kl_for_checkpoint(
                    seed=seed,
                    n_members=n,
                    epoch=epoch,
                    device=device,
                    model_pre=model_pre,
                    tokenizer=tokenizer,
                )
                if result is not None:
                    epoch_results.append(result)

            del model_pre
            if device.type == "cuda":
                torch.cuda.empty_cache()

            if not epoch_results:
                continue

            epoch_results.sort(key=lambda r: r["epoch"])
            print(f"\n  Summary — seed={seed}  N={n:,}")
            hdr = (
                f"  {'epoch':>5} | {'kl_seq':>7} | {'kl_tok':>7} | "
                f"{'pinsker_seq':>11} | {'bh_seq':>6} | vacuous"
            )
            sep = (
                "  " + "-"*5 + "+" + "-"*8 + "+" + "-"*8 + "+"
                + "-"*13 + "+" + "-"*8 + "+" + "-"*9
            )
            print(hdr)
            print(sep)
            for r in epoch_results:
                vacuous_str = "YES  ←" if r["pinsker_seq"] > 1 else "NO"
                print(
                    f"  {r['epoch']:>5} | {r['kl_seq']:>7.3f} | {r['kl_tok']:>7.3f} | "
                    f"  {r['pinsker_seq']:>9.3f}  | {r['bh_seq']:>6.3f} | {vacuous_str}"
                )

    total = time.time() - t_wall
    print(f"\n{'='*60}")
    print(f"Total wall time: {total:.1f}s  ({total/60:.1f} min)")


if __name__ == "__main__":
    main()
