"""
TV-route upper confidence bound on the advantage of telling P from Q
from ONE generated continuation.

    TV(P, Q) = E_{x~P}[(1 - Q(x)/P(x))_+] = E_{x~Q}[(1 - P(x)/Q(x))_+]

P = fine-tuned checkpoint, Q = reference (default: gpt_neo_pretrained).

Protocol (per prompt):
  1. Ancestral-sample n continuations of exactly T tokens from P and n from Q:
     temperature 1, no top-k / top-p, EOS logit masked at every step.
  2. Score every continuation under BOTH models with the SAME EOS-masked law
     (teacher forcing): y = log P(x) - log Q(x).
  3. z = (1 - e^{-y})_+ for P-samples, z = (1 - e^{y})_+ for Q-samples.
     All 2n terms are independent, in [0, 1], with mean TV(P, Q).
  4. Upper confidence bound: Chernoff / kl-inverse (Hoeffding 1963, Thm 1;
     valid for independent [0,1] variables with a common mean), at level
     alpha / K (Bonferroni over K prompts). Hoeffding UCB reported alongside.

Laws actually bounded: P_T(. | prompt), Q_T(. | prompt) = length-T
continuations with EOS forbidden. Bound covers an attacker who sees one
continuation per prompt. For k i.i.d. continuations: 1 - (1 - TV_up)^k.
Coverage is over the sampling randomness only (checkpoints fixed).

Usage:
  python tv_route/compute_tv_bound.py --p gpt_neo_ft_N2000_seed0_epoch5
  python tv_route/compute_tv_bound.py --p "gpt_neo_ft_N2000_seed0_epoch*" --n 1000 --T 64
  python tv_route/compute_tv_bound.py --p A --q B              # null control
  python tv_route/compute_tv_bound.py --p A --prompts prompts.jsonl
  python tv_route/compute_tv_bound.py --p A --n 20 --T 8       # smoke test
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from config import CKPT_DIR, CKPT_OUT_DIR, RESULTS_DIR  # noqa: E402

TV_DIR = _ROOT / "tv_route"
CACHE_DIR = TV_DIR / "cache"
OUT_CSV = RESULTS_DIR / "tv_bounds.csv"


# ----------------------------------------------------------------------------
# Confidence bounds for the mean of independent [0, 1] variables
# ----------------------------------------------------------------------------

def kl_bern(p: float, q: float) -> float:
    """kl(p || q) for Bernoulli laws, natural log."""
    eps = 1e-15
    q = min(max(q, eps), 1 - eps)
    out = 0.0
    if p > 0:
        out += p * math.log(p / q)
    if p < 1:
        out += (1 - p) * math.log((1 - p) / (1 - q))
    return out


def kl_inverse_ucb(mean: float, n: int, alpha: float, iters: int = 100) -> float:
    """sup{mu in [mean, 1] : kl(mean || mu) <= log(1/alpha) / n}.

    Valid (1 - alpha) upper confidence bound on the common mean of n
    independent [0,1] variables (Hoeffding 1963, Thm 1, lower tail).
    Returns the upper end of the bisection bracket (conservative)."""
    if mean >= 1.0:
        return 1.0
    c = math.log(1.0 / alpha) / n
    lo, hi = mean, 1.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if kl_bern(mean, mid) <= c:
            lo = mid
        else:
            hi = mid
    return hi


def hoeffding_ucb(mean: float, n: int, alpha: float) -> float:
    return min(1.0, mean + math.sqrt(math.log(1.0 / alpha) / (2 * n)))


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------

def resolve_ckpt_root() -> Path:
    """CKPT_DIR is a Kaggle mount in config.py; fall back to local checkpoints."""
    for root in (CKPT_DIR, CKPT_OUT_DIR):
        if Path(root).exists():
            return Path(root)
    raise FileNotFoundError(f"Neither {CKPT_DIR} nor {CKPT_OUT_DIR} exists.")


def load_model(path: Path, device: torch.device) -> AutoModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(str(path))
    return model.to(device).eval()


@torch.no_grad()
def sample_continuations(model, prefix: torch.Tensor, n: int, T: int, eos_id: int,
                         device: torch.device, batch: int, gen: torch.Generator) -> torch.Tensor:
    """n continuations of exactly T tokens, temperature 1, EOS masked.
    Own sampling loop (not HF generate) so the sampling law is exactly the
    law used in score_continuations."""
    out = []
    for s in range(0, n, batch):
        b = min(batch, n - s)
        cur = prefix.unsqueeze(0).repeat(b, 1).to(device)
        past, toks = None, []
        for _ in range(T):
            o = model(input_ids=cur, past_key_values=past, use_cache=True)
            past = o.past_key_values
            logits = o.logits[:, -1, :].float()
            logits[:, eos_id] = -float("inf")
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1, generator=gen)
            toks.append(nxt)
            cur = nxt
        out.append(torch.cat(toks, dim=1).cpu())
    return torch.cat(out, dim=0)  # (n, T)


@torch.no_grad()
def score_continuations(model, prefix: torch.Tensor, conts: torch.Tensor, eos_id: int,
                        device: torch.device, batch: int) -> np.ndarray:
    """log P(continuation | prefix) under the EOS-masked law, float64 sum."""
    L0 = prefix.numel()
    res = []
    for s in range(0, conts.shape[0], batch):
        c = conts[s:s + batch].to(device)
        full = torch.cat([prefix.unsqueeze(0).repeat(c.shape[0], 1).to(device), c], dim=1)
        logits = model(input_ids=full).logits[:, L0 - 1:-1, :].float()  # predicts c
        logits[..., eos_id] = -float("inf")
        lp = F.log_softmax(logits, dim=-1).gather(-1, c.unsqueeze(-1)).squeeze(-1)
        res.append(lp.double().sum(dim=1).cpu())
    return torch.cat(res).numpy()


# ----------------------------------------------------------------------------
# Prompts
# ----------------------------------------------------------------------------

def load_prompts(path: str | None, tokenizer, max_prompt_tokens: int) -> list[tuple[str, torch.Tensor]]:
    """Each prefix = [EOS] + prompt tokens. No file -> single unconditional prompt."""
    eos = tokenizer.eos_token_id
    if path is None:
        return [("<unconditional>", torch.tensor([eos], dtype=torch.long))]
    prompts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            text = json.loads(line)["prompt"] if line.startswith("{") else line
            ids = tokenizer(text, add_special_tokens=False)["input_ids"][:max_prompt_tokens]
            prompts.append((text, torch.tensor([eos] + ids, dtype=torch.long)))
    return prompts


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p", nargs="+", required=True,
                    help="P checkpoint dir name(s) under the checkpoint root; globs allowed")
    ap.add_argument("--q", default="gpt_neo_pretrained", help="Q checkpoint dir name")
    ap.add_argument("--n", type=int, default=1000, help="samples per model per prompt")
    ap.add_argument("--T", type=int, default=64, help="continuation length (tokens)")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--prompts", default=None, help="jsonl ({'prompt': ...}) or one prompt per line")
    ap.add_argument("--max_prompt_tokens", type=int, default=64)
    ap.add_argument("--gen_seed", type=int, default=12345)
    ap.add_argument("--gen_batch", type=int, default=32)
    ap.add_argument("--fwd_batch", type=int, default=16)
    ap.add_argument("--out", default=str(OUT_CSV))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = resolve_ckpt_root()
    q_path = root / args.q
    p_paths = sorted({p for pat in args.p for p in root.glob(pat) if p.is_dir()})
    if not p_paths:
        raise FileNotFoundError(f"No checkpoint matches {args.p} under {root}")

    tokenizer = AutoTokenizer.from_pretrained(str(q_path))
    eos = tokenizer.eos_token_id
    prompts = load_prompts(args.prompts, tokenizer, args.max_prompt_tokens)
    K = len(prompts)
    alpha_k = args.alpha / K
    N = 2 * args.n
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"device={device}  root={root}\nQ={args.q}  |P|={len(p_paths)}  K={K}  n={args.n}  "
          f"T={args.T}  alpha={args.alpha}  (alpha/K={alpha_k:.3g})")

    model_q = load_model(q_path, device)

    # Q-samples and log Q(Q-samples) depend only on (Q, prompt): compute once.
    q_cache = []
    for k, (_, prefix) in enumerate(prompts):
        gen = torch.Generator(device=device).manual_seed(args.gen_seed + 1_000_003 * k + 1)
        xq = sample_continuations(model_q, prefix, args.n, args.T, eos, device, args.gen_batch, gen)
        lq_xq = score_continuations(model_q, prefix, xq, eos, device, args.fwd_batch)
        q_cache.append((xq, lq_xq))

    rows = []
    for p_path in p_paths:
        t0 = time.time()
        model_p = load_model(p_path, device)
        per_prompt = []
        for k, (ptext, prefix) in enumerate(prompts):
            gen = torch.Generator(device=device).manual_seed(args.gen_seed + 1_000_003 * k)
            xp = sample_continuations(model_p, prefix, args.n, args.T, eos, device, args.gen_batch, gen)
            lp_xp = score_continuations(model_p, prefix, xp, eos, device, args.fwd_batch)
            lq_xp = score_continuations(model_q, prefix, xp, eos, device, args.fwd_batch)
            xq, lq_xq = q_cache[k]
            lp_xq = score_continuations(model_p, prefix, xq, eos, device, args.fwd_batch)

            y_p = lp_xp - lq_xp          # log P/Q on P-samples
            y_q = lp_xq - lq_xq          # log P/Q on Q-samples
            z_p = np.clip(-np.expm1(-y_p), 0.0, 1.0)   # (1 - Q/P)_+
            z_q = np.clip(-np.expm1(y_q), 0.0, 1.0)    # (1 - P/Q)_+
            tv_hat = float(np.concatenate([z_p, z_q]).mean())

            row = dict(
                p_ckpt=p_path.name, q_ckpt=args.q, prompt_idx=k, prompt=ptext[:80],
                n_per_model=args.n, T=args.T, alpha=args.alpha, K=K,
                tv_hat=tv_hat,
                tv_up_kl=kl_inverse_ucb(tv_hat, N, alpha_k),
                tv_up_hoeffding=hoeffding_ucb(tv_hat, N, alpha_k),
                # diagnostics (not bounds):
                tv_hat_from_P=float(z_p.mean()),   # both unbiased for TV;
                tv_hat_from_Q=float(z_q.mean()),   # a large gap flags a law mismatch
                kl_hat_PQ=float(y_p.mean()),       # plug-in KL(P||Q), NOT a bound
                mean_Q_over_P=float(np.exp(-y_p).mean()),  # should be ~1 (heavy-tailed)
            )
            per_prompt.append(row)
            np.savez_compressed(CACHE_DIR / f"{p_path.name}__vs__{q_path.name}__k{k}__n{args.n}_T{args.T}.npz",
                                y_p=y_p, y_q=y_q)

        worst = max(per_prompt, key=lambda r: r["tv_up_kl"])
        summary = dict(worst, prompt_idx=-1, prompt=f"MAX over {K} prompts")
        rows.extend(per_prompt + [summary])
        print(f"{p_path.name:45s} TV_hat={worst['tv_hat']:.4f}  TV_up(kl)={worst['tv_up_kl']:.4f}  "
              f"TV_up(Hoeff)={worst['tv_up_hoeffding']:.4f}  "
              f"[P:{worst['tv_hat_from_P']:.4f} Q:{worst['tv_hat_from_Q']:.4f}]  {time.time() - t0:.0f}s")
        del model_p
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, mode="a", header=not out.exists(), index=False)
    print(f"\nAppended {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
