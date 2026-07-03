"""
    python signals/compute_signals.py                          # all combinations
    python signals/compute_signals.py --seed 0 --n 2000       # single (seed, N)
    python signals/compute_signals.py --seed 0 --n 2000 --epoch 5
"""

import json
import math
import statistics
import sys
import time
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_ROOT = Path(__file__).parent.parent
_SIG_DIR = Path(__file__).parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "data"))
sys.path.insert(0, str(_SIG_DIR))

from config import (
    CORPUS_SIZES,
    CKPT_DIR,
    CKPT_OUT_DIR,
    PRETRAINED_CKPT,
    EPOCH_SWEEP,
    MAX_SEQ_LEN,
    MIN_K_FRACTION,
    RESULTS_DIR,
    SEEDS,
)
from membership_assignment import load_split
from s_loss import s_loss
from s_mink import s_mink
from s_ref import s_ref
from s_zlib import s_zlib


@torch.no_grad()
def get_sequence_logprobs(
    text: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
) -> torch.Tensor:
    
    input_ids = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LEN,
    )["input_ids"].to(device)  # (1, seq_len)

    logits = model(input_ids=input_ids).logits  # (1, seq_len, vocab_size)

    shift_logits = logits[0, :-1, :].float()    # (seq_len−1, vocab_size)
    shift_labels = input_ids[0, 1:]             # (seq_len−1,)

    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs[
        torch.arange(len(shift_labels), device=device), shift_labels
    ]
    return token_log_probs  # (seq_len − 1,)


def _load_model(ckpt_path: Path, device: torch.device) -> AutoModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(str(ckpt_path)).to(device)
    model.eval()
    return model


def _fmt_eps(eps: float) -> str:
    return f"{eps:g}"


def compute_signals_for_checkpoint(
    seed: int,
    n_members: int,
    epoch: int,
    device: torch.device,
    model_pre: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    lora_rank: int | None = None,
    dp_eps: float | None = None,
    mica_rank: int | None = None,
) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if dp_eps is not None:
        ckpt_name = f"gpt_neo_ft_dp_eps{_fmt_eps(dp_eps)}_N{n_members}_seed{seed}_epoch{epoch}"
        ckpt_path = CKPT_OUT_DIR / ckpt_name
        out_path = RESULTS_DIR / f"signals_dp_eps{_fmt_eps(dp_eps)}_N{n_members}_seed{seed}_epoch{epoch}.jsonl"
    elif lora_rank is not None:
        ckpt_name = f"gpt_neo_ft_lora_r{lora_rank}_N{n_members}_seed{seed}_epoch{epoch}"
        ckpt_path = CKPT_OUT_DIR / ckpt_name
        out_path = RESULTS_DIR / f"signals_lora_r{lora_rank}_N{n_members}_seed{seed}_epoch{epoch}.jsonl"
    elif mica_rank is not None:
        ckpt_name = f"gpt_neo_ft_mica_r{mica_rank}_N{n_members}_seed{seed}_epoch{epoch}"
        ckpt_path = CKPT_OUT_DIR / ckpt_name
        out_path = RESULTS_DIR / f"signals_mica_r{mica_rank}_N{n_members}_seed{seed}_epoch{epoch}.jsonl"
    else:
        ckpt_name = f"gpt_neo_ft_N{n_members}_seed{seed}_epoch{epoch}"
        ckpt_path = CKPT_DIR / ckpt_name
        out_path = RESULTS_DIR / f"signals_N{n_members}_seed{seed}_epoch{epoch}.jsonl"

    if out_path.exists():
        print(f"  [skip] {out_path.name} already exists.")
        return

    if not ckpt_path.exists():
        print(f"  [skip] Checkpoint not found: {ckpt_path}")
        return

    print(f"\n{'='*60}")
    print(f"  Checkpoint : {ckpt_name}")
    print(f"  Output     : {out_path.name}")

    print("  Loading fine-tuned model ...")
    model_ft = _load_model(ckpt_path, device)

    members, nonmembers, _ = load_split(seed, n_members)
    sequences = (
        [(rec, "member")    for rec in members] +
        [(rec, "nonmember") for rec in nonmembers]
    )
    n_total = len(sequences)

    results: list[dict] = []
    t0 = time.time()

    for i, (rec, split_label) in enumerate(sequences):
        text   = rec["text"]
        seq_id = rec["id"]

        logprobs_ft  = get_sequence_logprobs(text, model_ft,  tokenizer, device)
        logprobs_pre = get_sequence_logprobs(text, model_pre, tokenizer, device)

        score_loss = s_loss(logprobs_ft)
        score_ref  = s_ref(logprobs_ft, logprobs_pre)
        score_zlib = s_zlib(text, logprobs_ft)
        score_mink = s_mink(logprobs_ft, MIN_K_FRACTION)

        results.append({
            "id":    seq_id,
            "split": split_label,
            "s_loss": score_loss,
            "s_ref":  score_ref,
            "s_zlib": score_zlib,
            "s_mink": score_mink,
        })

        if (i + 1) % 500 == 0 or (i + 1) == n_total:
            print(f"    {i+1:>5}/{n_total}  [{time.time()-t0:.1f}s]")

    with out_path.open("w", encoding="utf-8") as fh:
        for row in results:
            fh.write(json.dumps(row) + "\n")

    def _mean_ref(split: str) -> float:
        vals = [r["s_ref"] for r in results
                if r["split"] == split and not math.isnan(r["s_ref"])]
        return statistics.mean(vals) if vals else math.nan

    elapsed = time.time() - t0
    print(f"  Processed {n_total:,} sequences in {elapsed:.1f}s")
    print(
        f"  Mean s_ref — members: {_mean_ref('member'):.4f}"
        f"  |  nonmembers: {_mean_ref('nonmember'):.4f}"
        f"  (members > nonmembers expected, especially at high epochs)"
    )

    del model_ft
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute MIA attack signals for fine-tuned GPT-Neo checkpoints."
    )
    parser.add_argument("--seed",  type=int, default=None,
                        help="Single seed to run (default: all SEEDS from config)")
    parser.add_argument("--n",     type=int, default=None,
                        help="Single N_members to run (default: all CORPUS_SIZES)")
    parser.add_argument("--epoch", type=int, default=None,
                        help="Single epoch to run (default: all EPOCH_SWEEP)")
    parser.add_argument("--lora_rank", type=int, default=None,
                        help="LoRA rank of checkpoint to use (default: None = full fine-tuning checkpoints)")
    parser.add_argument("--dp_eps", type=float, default=None,
                        help="DP epsilon of checkpoint to use (default: None = non-DP checkpoints)")
    parser.add_argument("--mica_rank", type=int, default=None,
                        help="MiCA rank of checkpoint to use (default: None = non-MiCA checkpoints)")
    args = parser.parse_args()

    seeds  = [args.seed]  if args.seed  is not None else SEEDS
    ns     = [args.n]     if args.n     is not None else CORPUS_SIZES
    epochs = [args.epoch] if args.epoch is not None else EPOCH_SWEEP
    lora_rank = args.lora_rank
    dp_eps    = args.dp_eps
    mica_rank = args.mica_rank

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device    : {device}")
    print(f"Seeds     : {seeds}")
    print(f"N values  : {ns}")
    print(f"Epochs    : {epochs}")
    print(f"Min-K frac: {MIN_K_FRACTION}")
    print(f"LoRA rank : {lora_rank}")
    print(f"DP epsilon: {dp_eps}")
    print(f"MiCA rank : {mica_rank}")

    t_wall = time.time()

    for seed in seeds:
        for n in ns:
            print(f"\n{'#'*60}")
            print(f"  seed={seed}  N={n:,}  — loading shared pretrained reference ...")

            tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_CKPT)
            tokenizer.pad_token = tokenizer.eos_token

            model_pre = _load_model(PRETRAINED_CKPT, device)

            for epoch in epochs:
                compute_signals_for_checkpoint(
                    seed=seed,
                    n_members=n,
                    epoch=epoch,
                    device=device,
                    model_pre=model_pre,
                    tokenizer=tokenizer,
                    lora_rank=lora_rank,
                    dp_eps=dp_eps,
                    mica_rank=mica_rank,
                )

            del model_pre
            if device.type == "cuda":
                torch.cuda.empty_cache()

    total = time.time() - t_wall
    print(f"\n{'='*60}")
    print(f"Total wall time: {total:.1f}s  ({total/60:.1f} min)")


if __name__ == "__main__":
    main()
