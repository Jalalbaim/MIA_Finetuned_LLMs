"""
Build one small, human-readable illustrative example of token-level KL / bound
localization, for a qualitative claim in the paper: the token-level bound
localizes leakage risk to specific tokens (rather than only ranking whole
documents), and degrades gracefully at those tokens instead of saturating the
way the whole-sequence bound does.

Uses the deepest, most memorization-prone configuration on disk
(n_members=6000, seed=0, epoch=20) and the SAME generation / KL conventions as
token_level/compute_token_bound.py (temperature=1.0 ancestral sampling, no
top-k/top-p, exact per-position KL via teacher-forced log_softmax) -- those
functions are imported and reused directly rather than reimplemented, so this
script cannot silently drift from the already-validated pipeline.

Two stages, run separately so the candidate selection can be reviewed before
the final per-example CSVs are written:

    python token_level/build_illustrative_example.py --stage candidates
    python token_level/build_illustrative_example.py --stage finalize

Stage "candidates": generates C candidates, computes per-token KL/bounds,
decodes tokens, cross-checks the top-5 highest-KL_t positions per candidate
against the real members_N6000_seed0.jsonl training text, writes
candidates_summary.csv, previews the selection + window-aggregate numbers
(to catch a saturated window BEFORE committing to it), and caches everything
to illustrative_example_cache.pt so stage "finalize" reruns no model calls.

Stage "finalize": loads the cache, re-derives the same selection
deterministically, and writes selected_example_tokens.csv and
selected_window_aggregate.csv.
"""

import argparse
import math
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer

# Windows consoles default to cp1252, which can't render raw BPE pieces like
# 'Ġ' (U+0120, GPT2's space marker) -- reconfigure so printing never crashes;
# CSV files are written with explicit encoding="utf-8" regardless (see to_csv
# calls below) so file content is unaffected by this.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "data"))

from config import DATA_DIR  # noqa: E402
from membership_assignment import load_split  # noqa: E402
from token_level.compute_token_bound import (  # noqa: E402
    resolve_ckpt_root,
    load_model,
    generate_sequences,
    per_position_kl_and_logprobs,
)

TOKEN_LEVEL_DIR = _ROOT / "token_level"


def _cache_path(epoch: int) -> Path:
    return TOKEN_LEVEL_DIR / f"illustrative_example_cache_epoch{epoch}.pt"


# Target configuration -- n_members/seed fixed by the brief; epoch is
# selectable (see main_scan): epoch=20 turned out to saturate the window-level
# bound unconditionally (see scan results), so a shallower epoch is used for
# the final example instead, chosen empirically rather than assumed.
N_MEMBERS = 6000
SEED = 0
DEFAULT_EPOCH = 20

# Generation
DEFAULT_C_CANDIDATES = 10
T_GEN = 64
GEN_SEED = 0

# Selection
DEFAULT_WINDOW_LEN = 32
TOP_K_CHECK = 5
NGRAM_SPANS = (12, 8, 5)     # largest first; first hit = longest match
MIN_VERIFIED_NGRAM = 5
FRONT_LOAD_CUTOFF = 15       # positions 1..15 = known positional-artifact zone
CROSSOVER_KL = 1.5936        # Pinsker/BH crossover (Pinsker tighter below, BH above)
NEIGHBOR_RADIUS = 5
SATURATION_THRESHOLD = 0.9   # min_window_seq >= this counts as "saturated" for our purposes


# Corpus cross-check


def ngram_ids_centered(seq_ids: list[int], center_1idx: int, width: int, T: int):
    """Return (ids, start_1idx, end_1idx_inclusive) for a `width`-token window
    centered on 1-indexed position `center_1idx`, clipped to [1, T]."""
    c0 = center_1idx - 1
    half = width // 2
    start = max(0, c0 - half)
    end = min(T, start + width)
    if end - start < width:
        start = max(0, end - width)
    return seq_ids[start:end], start + 1, end


def cross_check_candidate(seq_ids_1d, kl_row, tokenizer, corpus_texts, T):
    """For the top-K highest-KL_t positions of one candidate, test n-gram
    windows (largest span first) for an exact substring match against the
    real training corpus. Returns {position_1idx: {"does_match", "best_len"}}."""
    k = min(TOP_K_CHECK, T)
    topk = torch.topk(kl_row, k=k)
    positions = (topk.indices + 1).tolist()
    out = {}
    for pos in positions:
        best_len = 0
        for width in NGRAM_SPANS:
            ids, s, e = ngram_ids_centered(seq_ids_1d, pos, width, T)
            if not ids:
                continue
            text = tokenizer.decode(ids)
            if any(text in doc for doc in corpus_texts):
                best_len = e - s
                break
        out[pos] = {"does_match": best_len > 0, "best_len": best_len}
    return out


# Candidate summary (candidates_summary.csv)


def build_candidate_summary(cache) -> pd.DataFrame:
    kl_t = cache["kl_t"]
    C, T = kl_t.shape
    rows = []
    for i in range(C):
        vals = kl_t[i]
        max_val, max_idx = torch.max(vals, dim=0)
        max_kl_t = max_val.item()
        position_of_max_kl_t = max_idx.item() + 1
        background_median = vals.median().item()
        ratio = max_kl_t / background_median if background_median > 0 else float("nan")

        cc = cache["cross_check"].get(i, {})
        best_len, best_pos = 0, None
        for pos, info in cc.items():
            if info["best_len"] > best_len:
                best_len, best_pos = info["best_len"], pos

        rows.append({
            "candidate_id": i,
            "max_kl_t": max_kl_t,
            "position_of_max_kl_t": position_of_max_kl_t,
            "best_ngram_match_length": best_len,
            "best_ngram_match_position": best_pos,
            "background_median_kl_t": background_median,
            "spike_to_background_ratio": ratio,
            "selected": False,
        })
    return pd.DataFrame(rows)


# Selection (Step 5)


def _neighbor_fraction_below_crossover(kl_row, pos_1idx, T, radius=NEIGHBOR_RADIUS):
    c = pos_1idx - 1
    lo, hi = max(0, c - radius), min(T, c + radius + 1)
    idxs = [j for j in range(lo, hi) if j != c]
    if not idxs:
        return 0.0
    vals = kl_row[idxs]
    return (vals < CROSSOVER_KL).float().mean().item()


def _best_window_containing(spike_pos_1idx, T, W, front_cutoff, keep_in_front):
    starts = list(range(1, T - W + 2))
    feasible = [s for s in starts if s <= spike_pos_1idx <= s + W - 1]
    if not feasible:
        feasible = starts  # not expected for T=64, W=32 (every position is coverable)

    def overlap(s):
        e = s + W - 1
        return max(0, min(e, front_cutoff) - s + 1)

    def center_dist(s):
        e = s + W - 1
        return abs((s + e) / 2.0 - spike_pos_1idx)

    if keep_in_front:
        feasible.sort(key=center_dist)
    else:
        feasible.sort(key=lambda s: (overlap(s), center_dist(s)))
    best_start = feasible[0]
    return best_start, best_start + W - 1


def select_candidate_and_window(cache) -> dict:
    kl_t = cache["kl_t"]
    C, T = kl_t.shape
    W = cache["meta"]["window_len"]
    front_cutoff = cache["meta"]["front_load_cutoff"]
    lines = []

    # Criterion (a): verified corpus match at an isolated (low-KL-neighborhood) position
    candidates_a = []
    for i in range(C):
        cc = cache["cross_check"].get(i, {})
        for pos, info in cc.items():
            if info["does_match"] and info["best_len"] >= MIN_VERIFIED_NGRAM:
                frac = _neighbor_fraction_below_crossover(kl_t[i], pos, T)
                if frac > 0.5:
                    candidates_a.append((i, pos, info["best_len"], frac))

    if candidates_a:
        candidates_a.sort(key=lambda x: (-x[2], -x[3]))
        chosen_i, spike_pos, match_len, frac = candidates_a[0]
        mode, has_verified = "a", True
        lines.append(
            f"Criterion (a) SATISFIED: {len(candidates_a)} (candidate, position) pair(s) across "
            f"all candidates had a verified corpus match (n-gram length >= {MIN_VERIFIED_NGRAM}) "
            f"at a position where >50% of neighbors (+/-{NEIGHBOR_RADIUS} positions) sit below "
            f"the KL={CROSSOVER_KL} Pinsker/BH crossover."
        )
        lines.append("All qualifying (candidate, position) pairs, ranked by match length then neighbor purity:")
        for i, pos, mlen, fr in candidates_a:
            lines.append(f"    candidate {i}  position {pos}  match_len={mlen}  neighbor_purity={fr:.0%}")
        lines.append(f"Chosen: candidate {chosen_i}, position {spike_pos} (top of the ranking above).")
    else:
        best = None
        for i in range(C):
            vals = kl_t[i]
            if T <= front_cutoff:
                continue
            tail = vals[front_cutoff:]
            local_max, local_idx = torch.max(tail, dim=0)
            spike_pos_i = front_cutoff + local_idx.item() + 1
            median_val = vals.median().item()
            gap = local_max.item() - median_val
            if best is None or gap > best[3]:
                best = (i, spike_pos_i, local_max.item(), gap, median_val)
        chosen_i, spike_pos, spike_val, gap, median_val = best
        mode, has_verified = "b", False
        lines.append(
            f"Criterion (a) NOT satisfied by any candidate (no verified match >= "
            f"{MIN_VERIFIED_NGRAM} tokens with an isolated/low-KL neighborhood)."
        )
        lines.append(
            f"Falling back to criterion (b): for each candidate, find its highest KL_t among "
            f"positions > {front_cutoff} (excluding the known front-loaded positional-artifact "
            f"zone, positions 1-{front_cutoff}, from being treated as 'the' spike), then rank "
            f"candidates by the gap between that post-front-load max and the candidate's whole-"
            f"sequence background median KL_t."
        )
        lines.append(
            f"Chosen: candidate {chosen_i} -- post-position-{front_cutoff} max KL_t="
            f"{spike_val:.3f} at position {spike_pos}, background median={median_val:.3f}, "
            f"gap={gap:.3f}."
        )
        lines.append(
            "NOTE: this example is NOT a verified corpus match -- only a high-divergence "
            "signature. Paper text should say 'token signature consistent with memorization', "
            "not 'verified memorized fragment'."
        )

    keep_in_front = has_verified and spike_pos <= front_cutoff
    if keep_in_front:
        lines.append(
            f"Spike position {spike_pos} falls within the front-loaded zone (<= {front_cutoff}), "
            f"but it IS a verified corpus match, so per instructions it is kept there instead of "
            f"being discarded for positional-artifact concerns; the window is centered on it as "
            f"well as the sequence boundary allows."
        )
    window_start, window_end = _best_window_containing(spike_pos, T, W, front_cutoff, keep_in_front)
    lines.append(
        f"Window search over all {T - W + 1} contiguous {W}-token windows of candidate {chosen_i}: "
        f"selected [{window_start}, {window_end}] "
        + ("(centered on the in-front-zone verified spike)."
           if keep_in_front else
           f"(minimizing overlap with positions 1-{front_cutoff} while containing the identified "
           f"spike at position {spike_pos}, tie-broken by centering the spike in the window).")
    )

    return {
        "candidate_id": chosen_i,
        "spike_position": spike_pos,
        "mode": mode,
        "has_verified_match": has_verified,
        "window_start": window_start,
        "window_end": window_end,
        "reasoning": "\n".join(lines),
    }


# Window aggregate (Step 6) + saturation check


def compute_window_aggregate(cache, selection) -> dict:
    i = selection["candidate_id"]
    s, e = selection["window_start"], selection["window_end"]
    kl_window = cache["kl_t"][i, s - 1:e]
    min_window = cache["min_t"][i, s - 1:e]

    kl_window_seq = kl_window.sum().item()
    pinsker_window_seq = math.sqrt(kl_window_seq / 2.0)
    bh_window_seq = math.sqrt(1.0 - math.exp(-kl_window_seq))
    min_window_seq = min(pinsker_window_seq, bh_window_seq)
    sum_tok_min = min_window.sum().item()

    return {
        "kl_window_seq": kl_window_seq,
        "pinsker_window_seq": pinsker_window_seq,
        "bh_window_seq": bh_window_seq,
        "min_window_seq": min_window_seq,
        "sum_of_token_level_min_bound_over_window": sum_tok_min,
    }


def preview_window_aggregate(cache, selection):
    agg = compute_window_aggregate(cache, selection)
    print("\n" + "=" * 70)
    print(f"WINDOW AGGREGATE PREVIEW  (candidate {selection['candidate_id']}, "
          f"window [{selection['window_start']}, {selection['window_end']}])")
    print("=" * 70)
    for k, v in agg.items():
        print(f"  {k:45s} = {v:.6f}")

    saturated = agg["min_window_seq"] >= SATURATION_THRESHOLD
    print(f"\n  min_window_seq = {agg['min_window_seq']:.6f}  ->  " + (
        f"SATURATED (>= {SATURATION_THRESHOLD}): graceful-vs-catastrophic contrast DOES NOT "
        f"hold on this example."
        if saturated else
        f"comfortably informative (< {SATURATION_THRESHOLD}): contrast holds."
    ))
    return agg, saturated


def diagnostic_all_candidates_best_case(cache):
    """For every candidate, find the 32-window with the SMALLEST possible KL
    sum (the most favorable case for avoiding saturation) via a sliding-window
    sum. If even this best case saturates, no window choice within that
    candidate can avoid it -- this isolates whether saturation is a
    background-rate property of the checkpoint, not a bad window pick."""
    kl_t = cache["kl_t"]
    C, T = kl_t.shape
    W = cache["meta"]["window_len"]
    print(f"\nDiagnostic: best-case (KL-minimizing) {W}-token window per candidate "
          f"(is saturation avoidable at all, for ANY window placement?)")
    print(f"  {'cand':>4} | {'min kl_window_seq':>18} | {'bh_window_seq':>14} | saturated?")
    any_unsaturated = False
    for i in range(C):
        vals = kl_t[i]
        csum = torch.cat([torch.zeros(1), vals.cumsum(0)])
        window_sums = csum[W:] - csum[:-W]
        min_sum, _ = torch.min(window_sums, dim=0)
        kl_window_seq = min_sum.item()
        bh = math.sqrt(1.0 - math.exp(-kl_window_seq))
        sat = bh >= SATURATION_THRESHOLD
        any_unsaturated = any_unsaturated or not sat
        print(f"  {i:>4} | {kl_window_seq:>18.3f} | {bh:>14.9f} | {'YES' if sat else 'no'}")
    if not any_unsaturated:
        m = cache["meta"]
        print(f"  -> EVERY candidate saturates even at its best-case (KL-minimizing) window.\n"
              f"     This is a background per-token KL-rate property of this checkpoint "
              f"(n_members={m['n_members']}, seed={m['seed']}, epoch={m['epoch']}), not a "
              f"fixable window pick.")
    return any_unsaturated


# Epoch scan (find the shallowest epoch where the window bound isn't saturated)


def _spike_anchored_window_kl(kl_row: torch.Tensor, spike_idx0: int, W: int, T: int) -> float:
    """Sum KL over a length-W window centered on 0-indexed spike_idx0, clipped
    to [0, T). Mirrors _best_window_containing's centering, but returns the
    KL sum directly (diagnostic use -- not the full selection logic)."""
    half = W // 2
    start = max(0, spike_idx0 - half)
    end = min(T, start + W)
    if end - start < W:
        start = max(0, end - W)
    return kl_row[start:end].sum().item()


@torch.no_grad()
def main_scan(device, epochs: list[int], n_candidates: int, window_lens: list[int]):
    """For each epoch, generate once and compute per-token KL once, then
    evaluate every requested window length against the SAME per-token KL
    tensor (cheap -- no extra model calls). Reports two numbers per
    (epoch, window_len):
      best_case_bh   -- the KL-minimizing window's bh (most favorable case,
                         but typically spike-free / not useful on its own)
      spike_anchored_bh -- bh of the window centered on each candidate's own
                         highest-KL position (the window we'd ACTUALLY use,
                         since it must contain the spike to be illustrative)
    """
    ckpt_root = resolve_ckpt_root()
    pre_path = ckpt_root / "gpt_neo_pretrained"
    tokenizer = AutoTokenizer.from_pretrained(str(pre_path))
    tokenizer.pad_token = tokenizer.eos_token
    print("Loading pretrained reference model (shared across all scanned epochs) ...")
    model_pre = load_model(pre_path, device)

    rows = []
    for epoch in epochs:
        ft_path = ckpt_root / f"gpt_neo_ft_N{N_MEMBERS}_seed{SEED}_epoch{epoch}"
        if not ft_path.exists():
            print(f"\n[skip] epoch={epoch}: checkpoint not found at {ft_path}")
            continue

        print(f"\n{'=' * 70}\nepoch={epoch}\n{'=' * 70}")
        print("  Loading fine-tuned checkpoint ...")
        model_ft = load_model(ft_path, device)

        t0 = time.time()
        seqs = generate_sequences(
            model_ft, tokenizer, m=n_candidates, T=T_GEN, device=device,
            gen_seed=GEN_SEED, gen_batch=n_candidates,
        )
        kl_t_all, _, _ = per_position_kl_and_logprobs(
            seqs, model_ft, model_pre, device, fwd_batch=n_candidates,
        )
        kl_t_all = kl_t_all.clamp(min=0.0)
        elapsed = time.time() - t0
        C, T = kl_t_all.shape
        print(f"  {n_candidates} candidates, T={T_GEN}  (elapsed {elapsed:.1f}s)")

        spike_idx0 = kl_t_all.argmax(dim=1)  # (C,) 0-indexed global-max position per candidate

        for W in window_lens:
            csum = torch.cat([torch.zeros(C, 1), kl_t_all.cumsum(dim=1)], dim=1)
            window_sums = csum[:, W:] - csum[:, :-W]
            min_window_per_cand, _ = window_sums.min(dim=1)
            bh_best_case = torch.sqrt(1.0 - torch.exp(-min_window_per_cand))

            spike_window_kl = torch.tensor([
                _spike_anchored_window_kl(kl_t_all[i], spike_idx0[i].item(), W, T)
                for i in range(C)
            ])
            bh_spike_anchored = torch.sqrt(1.0 - torch.exp(-spike_window_kl))

            n_unsat_best = (bh_best_case < SATURATION_THRESHOLD).sum().item()
            n_unsat_spike = (bh_spike_anchored < SATURATION_THRESHOLD).sum().item()
            min_bh_spike = bh_spike_anchored.min().item()

            print(f"    W={W:>3}  best-case bh_min={bh_best_case.min().item():.6f} "
                  f"({n_unsat_best}/{C} < {SATURATION_THRESHOLD})   "
                  f"spike-anchored bh_min={min_bh_spike:.6f} ({n_unsat_spike}/{C} < {SATURATION_THRESHOLD})  "
                  f"bh_median={bh_spike_anchored.median().item():.6f}")

            rows.append({
                "epoch": epoch, "window_len": W,
                "best_case_bh_min": bh_best_case.min().item(),
                "n_unsaturated_best_case": n_unsat_best,
                "spike_anchored_bh_min": min_bh_spike,
                "spike_anchored_bh_median": bh_spike_anchored.median().item(),
                "n_unsaturated_spike_anchored": n_unsat_spike,
                "n_candidates": C,
            })

        del model_ft
        if device.type == "cuda":
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    print(f"\n{'=' * 70}\nSCAN SUMMARY (spike-anchored is the number that matters for the real example)\n{'=' * 70}")
    print(df.to_string(index=False))

    viable = df[df["n_unsaturated_spike_anchored"] > 0]
    if viable.empty:
        print(f"\nNo (epoch, window_len) combination scanned produced ANY candidate whose "
              f"SPIKE-ANCHORED window is unsaturated (bh < {SATURATION_THRESHOLD}). "
              f"Consider even shorter windows and/or note that a genuine spike may be "
              f"fundamentally incompatible with 'comfortably below 1' at any practical window length.")
    else:
        best_row = viable.sort_values(["window_len", "epoch"]).iloc[0]
        print(f"\nRecommendation: (epoch={int(best_row['epoch'])}, window_len={int(best_row['window_len'])}) "
              f"is the smallest/shallowest scanned combination with >=1 candidate whose "
              f"spike-anchored window is not saturated "
              f"(bh_min={best_row['spike_anchored_bh_min']:.4f}, "
              f"{int(best_row['n_unsaturated_spike_anchored'])}/{int(best_row['n_candidates'])} candidates qualify).")

    out_path = TOKEN_LEVEL_DIR / "epoch_scan_summary.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\nWrote {out_path}")
    return df


# Stage 1: candidates


@torch.no_grad()
def main_candidates(device, epoch: int = DEFAULT_EPOCH, n_candidates: int = DEFAULT_C_CANDIDATES,
                     window_len: int = DEFAULT_WINDOW_LEN):
    ckpt_root = resolve_ckpt_root()
    ft_path = ckpt_root / f"gpt_neo_ft_N{N_MEMBERS}_seed{SEED}_epoch{epoch}"
    pre_path = ckpt_root / "gpt_neo_pretrained"
    print(f"Ckpt root      : {ckpt_root}")
    print(f"Fine-tuned ckpt: {ft_path}  exists={ft_path.exists()}")
    print(f"Pretrained ckpt: {pre_path}  exists={pre_path.exists()}")
    if not ft_path.exists() or not pre_path.exists():
        raise FileNotFoundError("Required checkpoint(s) missing -- see printout above.")

    tokenizer = AutoTokenizer.from_pretrained(str(pre_path))
    tokenizer.pad_token = tokenizer.eos_token
    tok_ft = AutoTokenizer.from_pretrained(str(ft_path))
    assert tok_ft.eos_token_id == tokenizer.eos_token_id == 50256, "eos_token_id mismatch"
    assert tok_ft.get_vocab() == tokenizer.get_vocab(), "vocab mismatch between ft/pretrained tokenizers"
    print(f"Tokenizer      : {tokenizer.__class__.__name__}  eos_id=pad_id=bos_id="
          f"{tokenizer.eos_token_id}  vocab_size={tokenizer.vocab_size}")
    print("  Confirmed: fine-tuned checkpoint's tokenizer matches the pretrained reference "
          "tokenizer exactly (same vocab, same EOS-as-BOS-as-PAD convention).")

    print(f"\nLoading real training corpus text: "
          f"{DATA_DIR / f'members_N{N_MEMBERS}_seed{SEED}.jsonl'} ...")
    members, _, _ = load_split(SEED, N_MEMBERS, data_dir=DATA_DIR)
    print(f"  First record keys: {sorted(members[0].keys())}")
    corpus_texts = [rec["text"] for rec in members]
    print(f"  {len(corpus_texts):,} member documents loaded (this IS the member-set split "
          f"actually used to fine-tune this checkpoint).")

    print("\nLoading fine-tuned + pretrained models ...")
    model_ft = load_model(ft_path, device)
    model_pre = load_model(pre_path, device)

    print(f"\nGenerating C={n_candidates} candidates, T={T_GEN} tokens each -- "
          f"do_sample=True, temperature=1.0, top_k=0 (disabled), top_p=1.0 (disabled), "
          f"min_new_tokens=max_new_tokens={T_GEN} (fixed length), gen_seed={GEN_SEED}.")
    print("  CONFIRMED: no truncation sampling (no top-k, no top-p, no greedy) anywhere in this call.")
    t0 = time.time()
    seqs = generate_sequences(
        model_ft, tokenizer, m=n_candidates, T=T_GEN, device=device,
        gen_seed=GEN_SEED, gen_batch=n_candidates,
    )
    print(f"  seqs shape={tuple(seqs.shape)}  elapsed={time.time() - t0:.1f}s")

    print("\nTeacher-forcing both models over generated candidates (exact per-position KL, "
          "one forward pass per model, no sampling) ...")
    kl_t_all, lp_ft_all, lp_pre_all = per_position_kl_and_logprobs(
        seqs, model_ft, model_pre, device, fwd_batch=n_candidates,
    )

    neg_mag = (-kl_t_all.clamp(max=0.0)).max().item()
    if neg_mag > 1e-4:
        print(f"  WARNING: KL_t negative beyond float noise (max |neg|={neg_mag:.6f}); clamping to 0.")
    kl_t_all = kl_t_all.clamp(min=0.0)

    bh_t_all = torch.sqrt(1.0 - torch.exp(-kl_t_all))
    pinsker_t_all = torch.sqrt(kl_t_all / 2.0)
    min_t_all = torch.minimum(pinsker_t_all, bh_t_all)

    print(f"\nCross-checking top-{TOP_K_CHECK} highest-KL_t positions per candidate against "
          f"{len(corpus_texts):,} member documents (n-gram spans={NGRAM_SPANS}, exact substring) ...")
    cross_check = {}
    any_match_count = 0
    for i in range(n_candidates):
        seq_ids_1d = seqs[i, 1:].tolist()
        cc = cross_check_candidate(seq_ids_1d, kl_t_all[i], tokenizer, corpus_texts, T_GEN)
        cross_check[i] = cc
        matched = {p: v for p, v in cc.items() if v["does_match"]}
        if matched:
            any_match_count += 1
        print(f"  candidate {i}: checked positions {sorted(cc.keys())}  "
              f"matched at {sorted(matched.keys()) if matched else 'none'}")

    print(f"\n  SANITY CHECK: {n_candidates}/{n_candidates} candidates were checked for corpus "
          f"matches; {any_match_count}/{n_candidates} had >=1 verbatim match at >=1 of their "
          f"top-{TOP_K_CHECK} positions.")

    cache_path = _cache_path(epoch)
    cache = {
        "seqs": seqs, "kl_t": kl_t_all, "pinsker_t": pinsker_t_all,
        "bh_t": bh_t_all, "min_t": min_t_all, "cross_check": cross_check,
        "meta": {
            "n_members": N_MEMBERS, "seed": SEED, "epoch": epoch,
            "C": n_candidates, "T": T_GEN, "gen_seed": GEN_SEED,
            "window_len": window_len, "crossover_kl": CROSSOVER_KL,
            "front_load_cutoff": FRONT_LOAD_CUTOFF,
        },
    }
    torch.save(cache, cache_path)
    print(f"\nCached generation + KL + cross-check data -> {cache_path}")

    summary_df = build_candidate_summary(cache)
    selection = select_candidate_and_window(cache)
    summary_df.loc[summary_df["candidate_id"] == selection["candidate_id"], "selected"] = True

    out_path = TOKEN_LEVEL_DIR / "candidates_summary.csv"
    summary_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\nWrote {out_path}")
    print(summary_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("SELECTION REASONING")
    print("=" * 70)
    print(selection["reasoning"])

    preview_window_aggregate(cache, selection)
    diagnostic_all_candidates_best_case(cache)

    return cache, summary_df, selection


# Stage 2: finalize


def main_finalize(epoch: int = DEFAULT_EPOCH):
    cache_path = _cache_path(epoch)
    if not cache_path.exists():
        raise FileNotFoundError(f"{cache_path} not found -- run --stage candidates --epoch {epoch} first.")
    cache = torch.load(cache_path, weights_only=False)

    tokenizer = AutoTokenizer.from_pretrained(str(resolve_ckpt_root() / "gpt_neo_pretrained"))
    tokenizer.pad_token = tokenizer.eos_token

    selection = select_candidate_and_window(cache)
    print(selection["reasoning"])

    agg, saturated = preview_window_aggregate(cache, selection)
    if saturated:
        print(f"\n*** min_window_seq >= {SATURATION_THRESHOLD} for the selected window. Confirmed "
              f"(via epoch x window-length scan, see epoch_scan_summary.csv) that this is not "
              f"fixable by choice of candidate/window/epoch -- the BH transform saturates almost "
              f"immediately once a real memorization spike is included (bh=0.893 already at the "
              f"theorem's own Pinsker/BH crossover, KL=1.5936). Per explicit user direction, "
              f"reporting this honestly rather than chasing a non-saturated number: the paper's "
              f"claim should rest on file 2's token-by-token profile (kl_t/pinsker_t keep "
              f"discriminating magnitude and location) vs file 3's single collapsed aggregate "
              f"(uninformative regardless of the shape underneath), not on the aggregate itself "
              f"being numerically low. Proceeding to write files 2/3 with the real number. ***")

    i = selection["candidate_id"]
    s, e = selection["window_start"], selection["window_end"]
    seqs, kl_t, pinsker_t, bh_t, min_t = (
        cache["seqs"], cache["kl_t"], cache["pinsker_t"], cache["bh_t"], cache["min_t"]
    )
    cc = cache["cross_check"].get(i, {})

    rows = []
    for local_pos, orig_pos in enumerate(range(s, e + 1), start=1):
        token_id = seqs[i, orig_pos].item()
        info = cc.get(orig_pos, {"does_match": False, "best_len": 0})
        rows.append({
            "local_position": local_pos,
            "original_position_in_candidate": orig_pos,
            "token_id": token_id,
            "token_str_decoded": tokenizer.decode([token_id]),
            "token_str_raw": tokenizer.convert_ids_to_tokens([token_id])[0],
            "kl_t": kl_t[i, orig_pos - 1].item(),
            "pinsker_t": pinsker_t[i, orig_pos - 1].item(),
            "bh_t": bh_t[i, orig_pos - 1].item(),
            "min_t": min_t[i, orig_pos - 1].item(),
            "is_corpus_match": info["does_match"],
            "matched_ngram_length": info["best_len"],
        })
    tokens_df = pd.DataFrame(rows)
    tokens_path = TOKEN_LEVEL_DIR / "selected_example_tokens.csv"
    tokens_df.to_csv(tokens_path, index=False, encoding="utf-8")
    print(f"\nWrote {tokens_path}  ({len(tokens_df)} rows)")
    print(tokens_df.to_string(index=False))

    meta = cache["meta"]
    agg_row = {
        "kl_window_seq": agg["kl_window_seq"],
        "pinsker_window_seq": agg["pinsker_window_seq"],
        "bh_window_seq": agg["bh_window_seq"],
        "min_window_seq": agg["min_window_seq"],
        "sum_of_token_level_min_bound_over_window": agg["sum_of_token_level_min_bound_over_window"],
        "n_members": meta["n_members"],
        "seed": meta["seed"],
        "epochs": meta["epoch"],
        "window_start_position": s,
        "window_end_position": e,
        "has_verified_corpus_match": selection["has_verified_match"],
    }
    agg_path = TOKEN_LEVEL_DIR / "selected_window_aggregate.csv"
    pd.DataFrame([agg_row]).to_csv(agg_path, index=False, encoding="utf-8")
    print(f"\nWrote {agg_path}")
    print(pd.DataFrame([agg_row]).to_string(index=False))

    return tokens_df, agg_row


def main():
    parser = argparse.ArgumentParser(description="Build one illustrative token-level example.")
    parser.add_argument("--stage", choices=["scan", "candidates", "finalize"], required=True)
    parser.add_argument("--epoch", type=int, default=DEFAULT_EPOCH,
                         help="Epoch checkpoint to use (stages: candidates, finalize).")
    parser.add_argument("--n_candidates", type=int, default=DEFAULT_C_CANDIDATES,
                         help="Number of candidates C to generate (stages: scan, candidates).")
    parser.add_argument("--window_len", type=int, default=DEFAULT_WINDOW_LEN,
                         help="Window length (stage: candidates).")
    parser.add_argument("--scan_epochs", type=int, nargs="+", default=[1, 2, 3, 5, 10],
                         help="Epochs to scan (stage: scan).")
    parser.add_argument("--scan_window_lens", type=int, nargs="+", default=[4, 8, 12, 16, 24, 32],
                         help="Window lengths to scan (stage: scan).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.stage == "scan":
        main_scan(device, args.scan_epochs, args.n_candidates, args.scan_window_lens)
    elif args.stage == "candidates":
        main_candidates(device, epoch=args.epoch, n_candidates=args.n_candidates, window_len=args.window_len)
    else:
        main_finalize(epoch=args.epoch)


if __name__ == "__main__":
    main()
