"""
DP-SGD fine-tuning pipeline (Step 7, RQ4).

For each privacy budget epsilon, fine-tunes a fresh GPT-Neo on the N=6000,
seed=0 member set under DP-SGD (Opacus), saves the unwrapped checkpoint plus
a privacy receipt JSON, and records perplexity on the held-out eval set E.

Usage:
    python finetune/train_dp.py                    # all eps at N=6000 seed=0
    python finetune/train_dp.py --eps 1            # single budget
    python finetune/train_dp.py --seed 0 --n 6000 --epochs 3
"""

import sys
import json
import math
import time
import argparse
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "data"))
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    MAX_SEQ_LEN,
    FINETUNE, DP,
    PRETRAINED_CKPT, CKPT_DIR, RESULTS_DIR, LOG_DIR,
)
from membership_assignment import load_split
from train import EnronDataset, save_pretrained_reference

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)

from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from opacus.utils.batch_memory_manager import BatchMemoryManager


# Naming helpers

def _fmt_eps(eps: float) -> str:
    """Format epsilon for filenames: 1.0 -> '1', 0.5 -> '0.5'."""
    return f"{eps:g}"


def dp_ckpt_dir(eps: float, n_members: int, seed: int, epochs: int) -> Path:
    return CKPT_DIR / f"gpt_neo_ft_dp_eps{_fmt_eps(eps)}_N{n_members}_seed{seed}_epoch{epochs}"


def dp_receipt_path(eps: float, n_members: int, seed: int) -> Path:
    return RESULTS_DIR / f"dp_receipt_eps{_fmt_eps(eps)}_N{n_members}_seed{seed}.json"


# GPT-Neo tied-weights fix

def _untie_lm_head(model: nn.Module) -> nn.Module:
    """
    GPT-Neo ties lm_head.weight to transformer.wte.weight (standard weight
    tying for parameter efficiency). Opacus's per-sample-gradient
    GradSampleModule registers grad-sample hooks per-module, so a weight
    shared between two modules (Embedding and Linear) ends up with two
    competing/duplicate grad_sample entries for the same underlying
    nn.Parameter, which Opacus rejects (tied-weights / duplicate-parameter
    error). We break the tie by giving lm_head its own clone of the
    embedding weight and disabling tie_word_embeddings, so each module owns
    a distinct nn.Parameter.
    """
    if model.lm_head.weight is model.transformer.wte.weight:
        new_lm_head = nn.Linear(
            model.lm_head.in_features,
            model.lm_head.out_features,
            bias=(model.lm_head.bias is not None),
        )
        new_lm_head.weight = nn.Parameter(model.lm_head.weight.detach().clone())
        if model.lm_head.bias is not None:
            new_lm_head.bias = nn.Parameter(model.lm_head.bias.detach().clone())
        model.lm_head = new_lm_head
        model.config.tie_word_embeddings = False
        print("  Untied lm_head.weight from transformer.wte.weight (Opacus compatibility).")
    return model


# Perplexity

@torch.no_grad()
def compute_perplexity(model, tokenizer, eval_sequences: list[dict], device: torch.device) -> float:
    """
    Standard perplexity on held-out eval set E:
    exp(mean per-token negative log-likelihood) over all sequences in E.
    """
    model.eval()
    total_nll = 0.0
    total_tokens = 0

    for rec in eval_sequences:
        input_ids = tokenizer(
            rec["text"],
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SEQ_LEN,
        )["input_ids"].to(device)

        if input_ids.shape[1] < 2:
            continue

        loss = model(input_ids=input_ids, labels=input_ids).loss
        n_tokens = input_ids.shape[1] - 1
        total_nll += loss.item() * n_tokens
        total_tokens += n_tokens

    return math.exp(total_nll / total_tokens)


# DP training

def train_dp(
    seed: int,
    n_members: int,
    target_epsilon: float,
    epochs: int,
    device: torch.device,
) -> dict:
    torch.manual_seed(seed)

    # Data
    members, _, _ = load_split(seed, n_members)
    print(f"  Members: {len(members):,}")

    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_CKPT)
    tokenizer.pad_token = tokenizer.eos_token

    dataset  = EnronDataset(members, tokenizer)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    loader   = DataLoader(
        dataset,
        batch_size=FINETUNE["batch_size"],
        shuffle=True,
        collate_fn=collator,
        pin_memory=(device.type == "cuda"),
    )

    # Fresh pretrained model
    model = AutoModelForCausalLM.from_pretrained(PRETRAINED_CKPT).to(device)

    # Opacus-compatibility fixes
    model = _untie_lm_head(model)

    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        print(f"  ModuleValidator found {len(errors)} issue(s); fixing ...")
        for e in errors:
            print(f"    - {e}")
        model = ModuleValidator.fix(model).to(device)
    else:
        print("  ModuleValidator: model is Opacus-compatible.")

    # Optimizer (created BEFORE attaching the PrivacyEngine)
    optimizer = AdamW(
        model.parameters(),
        lr=FINETUNE["learning_rate"],
        weight_decay=FINETUNE["weight_decay"],
    )

    # Attach privacy — auto-computes noise_multiplier for target_epsilon
    target_delta = 1.0 / n_members
    privacy_engine = PrivacyEngine()
    model, optimizer, data_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        epochs=epochs,
        max_grad_norm=DP["max_grad_norm"],
    )

    sample_rate = getattr(data_loader, "sample_rate", FINETUNE["batch_size"] / len(dataset))
    print(
        f"  Noise multiplier : {optimizer.noise_multiplier:.4f}\n"
        f"  Sample rate      : {sample_rate:.6f}\n"
        f"  Target epsilon   : {target_epsilon}  |  delta: {target_delta:.2e}"
    )

    # CSV log
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"train_dp_eps{_fmt_eps(target_epsilon)}_N{n_members}_seed{seed}.csv"
    log_fh = log_path.open("w", newline="", encoding="utf-8")
    import csv
    writer = csv.writer(log_fh)
    writer.writerow(["epoch", "train_loss", "elapsed_s"])

    print(f"  {'epoch':>5}  {'loss':>9}  {'time':>8}")
    print(f"  {'-----':>5}  {'---------':>9}  {'--------':>8}")

    avg_loss = float("nan")
    try:
        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            n_steps = 0
            t0 = time.time()

            with BatchMemoryManager(
                data_loader=data_loader,
                max_physical_batch_size=4,
                optimizer=optimizer,
            ) as memory_safe_loader:
                for batch in memory_safe_loader:
                    if batch["input_ids"].shape[0] == 0:
                        continue  # Poisson sampling can yield empty batches

                    batch = {k: v.to(device) for k, v in batch.items()}
                    optimizer.zero_grad()
                    loss = model(**batch).loss
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    n_steps += 1

            avg_loss = running_loss / max(n_steps, 1)
            elapsed = time.time() - t0
            writer.writerow([epoch, f"{avg_loss:.6f}", f"{elapsed:.1f}"])
            log_fh.flush()
            print(f"  {epoch:>5}  {avg_loss:>9.4f}  {elapsed:>7.1f}s")
    finally:
        log_fh.close()

    achieved_epsilon = privacy_engine.get_epsilon(delta=target_delta)
    noise_multiplier = optimizer.noise_multiplier

    # Unwrap the GradSampleModule before saving
    model.remove_hooks()
    ft_model = model._module

    ckpt_dir = dp_ckpt_dir(target_epsilon, n_members, seed, epochs)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ft_model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)

    return {
        "model": ft_model,
        "tokenizer": tokenizer,
        "ckpt_dir": ckpt_dir,
        "achieved_epsilon": achieved_epsilon,
        "noise_multiplier": noise_multiplier,
        "sample_rate": sample_rate,
        "train_loss": avg_loss,
    }


# Entry point

def main() -> None:
    parser = argparse.ArgumentParser(description="DP-SGD fine-tune GPT-Neo on Enron members.")
    parser.add_argument("--seed",   type=int, default=0,
                        help="Seed (default: 0)")
    parser.add_argument("--n",      type=int, default=6000,
                        help="N_members (default: 6000)")
    parser.add_argument("--epochs", type=int, default=DP["epochs"],
                        help=f"Epochs (default: {DP['epochs']})")
    parser.add_argument("--eps",    type=float, default=None,
                        help="Single privacy budget (default: all DP['epsilon_values'])")
    args = parser.parse_args()

    eps_values = [args.eps] if args.eps is not None else DP["epsilon_values"]
    n_members  = args.n
    seed       = args.seed
    epochs     = args.epochs

    save_pretrained_reference()
    delta      = 1.0 / n_members

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eff_batch = FINETUNE["batch_size"] * FINETUNE.get("grad_accum_steps", 1)
    print(f"Device         : {device}")
    print(f"Seed           : {seed}")
    print(f"N_members      : {n_members:,}")
    print(f"Epochs         : {epochs}")
    print(f"Epsilon values : {eps_values}")
    print(f"Delta          : 1/{n_members} = {delta:.2e}")
    print(f"max_grad_norm  : {DP['max_grad_norm']}")
    print(f"Logical batch  : {FINETUNE['batch_size']}  (Opacus uses Poisson sampling, not "
          f"grad-accum; effective non-DP batch was {eff_batch})")
    print(
        "  NOTE: protocol fixes batch_size = 8; a logical batch of 32-64 would "
        "improve the privacy-utility tradeoff (lower noise for the same "
        "epsilon) if compute allows.\n"
    )

    # Non-DP utility baseline (N=6000, seed=0, epoch=3) — for Figure 4 comparison
    baseline_ckpt = CKPT_DIR / f"gpt_neo_ft_N{n_members}_seed{seed}_epoch{epochs}"
    baseline_perplexity = None
    if baseline_ckpt.exists():
        print(f"Computing non-DP baseline perplexity from {baseline_ckpt.name} ...")
        _, _, eval_set = load_split(seed, n_members)
        base_tokenizer = AutoTokenizer.from_pretrained(baseline_ckpt)
        base_tokenizer.pad_token = base_tokenizer.eos_token
        base_model = AutoModelForCausalLM.from_pretrained(baseline_ckpt).to(device)
        baseline_perplexity = compute_perplexity(base_model, base_tokenizer, eval_set, device)
        print(f"  Non-DP baseline perplexity: {baseline_perplexity:.4f}\n")

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        baseline_path = RESULTS_DIR / f"dp_receipt_baseline_N{n_members}_seed{seed}_epoch{epochs}.json"
        with baseline_path.open("w", encoding="utf-8") as fh:
            json.dump({
                "n_members": n_members,
                "seed": seed,
                "epochs": epochs,
                "checkpoint": str(baseline_ckpt.relative_to(_ROOT)),
                "perplexity": baseline_perplexity,
            }, fh, indent=2)

        del base_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        print(f"  [warn] Non-DP baseline checkpoint not found: {baseline_ckpt}\n")

    # Eval set for DP checkpoint perplexity (shared across all eps)
    _, _, eval_set = load_split(seed, n_members)

    summary_rows = []
    t_wall = time.time()

    for eps in eps_values:
        print(f"\n{'='*64}")
        print(f"  epsilon = {eps}  |  N={n_members:,}  seed={seed}  epochs={epochs}")
        print(f"{'='*64}")

        ckpt_dir = dp_ckpt_dir(eps, n_members, seed, epochs)
        receipt_path = dp_receipt_path(eps, n_members, seed)

        if ckpt_dir.exists() and receipt_path.exists():
            print(f"  [skip] checkpoint + receipt already exist: {ckpt_dir.name}")
            with receipt_path.open(encoding="utf-8") as fh:
                receipt = json.load(fh)
            summary_rows.append({
                "eps": eps,
                "achieved_epsilon": receipt["achieved_epsilon"],
                "noise_multiplier": receipt["noise_multiplier"],
                "train_loss": None,
                "perplexity": receipt["perplexity"],
                "wall_time": 0.0,
            })
            continue

        t0 = time.time()

        if ckpt_dir.exists():
            print(f"  Checkpoint exists, skipping training: {ckpt_dir.name}")
            tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
            tokenizer.pad_token = tokenizer.eos_token
            ft_model = AutoModelForCausalLM.from_pretrained(ckpt_dir).to(device)
            train_loss = None
            achieved_epsilon = None
            noise_multiplier = None
            sample_rate = FINETUNE["batch_size"] / n_members
        else:
            result = train_dp(seed, n_members, eps, epochs, device)
            ft_model = result["model"]
            tokenizer = result["tokenizer"]
            train_loss = result["train_loss"]
            achieved_epsilon = result["achieved_epsilon"]
            noise_multiplier = result["noise_multiplier"]
            sample_rate = result["sample_rate"]

        # Perplexity on held-out eval set E
        ft_model.to(device)
        perplexity = compute_perplexity(ft_model, tokenizer, eval_set, device)

        wall_time = time.time() - t0

        # Privacy receipt
        if achieved_epsilon is None:
            target_delta = 1.0 / n_members
            print("  [warn] Re-loaded checkpoint without re-running PrivacyEngine; "
                  "achieved_epsilon/noise_multiplier left as null in receipt.")

        receipt = {
            "target_epsilon": eps,
            "achieved_epsilon": achieved_epsilon,
            "delta": delta,
            "noise_multiplier": noise_multiplier,
            "max_grad_norm": DP["max_grad_norm"],
            "epochs": epochs,
            "n_members": n_members,
            "sample_rate": sample_rate,
            "batch_size": FINETUNE["batch_size"],
            "perplexity": perplexity,
        }
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with receipt_path.open("w", encoding="utf-8") as fh:
            json.dump(receipt, fh, indent=2)

        summary_rows.append({
            "eps": eps,
            "achieved_epsilon": achieved_epsilon,
            "noise_multiplier": noise_multiplier,
            "train_loss": train_loss,
            "perplexity": perplexity,
            "wall_time": wall_time,
        })

        if achieved_epsilon is not None:
            deviation = abs(achieved_epsilon - eps) / eps
            if deviation > 0.05:
                print(
                    f"  WARNING: achieved_epsilon ({achieved_epsilon:.4f}) deviates from "
                    f"target ({eps}) by {deviation:.1%} (> 5%)."
                )

        del ft_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    total = time.time() - t_wall

    print(f"\n{'='*64}")
    print(f"Summary — N={n_members:,}  seed={seed}  epochs={epochs}")
    if baseline_perplexity is not None:
        print(f"  Non-DP baseline perplexity: {baseline_perplexity:.4f}")
    hdr = (
        f"  {'eps':>5} | {'achieved_eps':>12} | {'noise_mult':>10} | "
        f"{'train_loss':>10} | {'perplexity':>10} | {'wall_time':>9}"
    )
    sep = "  " + "-"*5 + "+" + "-"*14 + "+" + "-"*12 + "+" + "-"*12 + "+" + "-"*12 + "+" + "-"*11
    print(hdr)
    print(sep)
    for row in summary_rows:
        ach = f"{row['achieved_epsilon']:.4f}" if row["achieved_epsilon"] is not None else "n/a"
        nm  = f"{row['noise_multiplier']:.4f}" if row["noise_multiplier"] is not None else "n/a"
        tl  = f"{row['train_loss']:.4f}" if row["train_loss"] is not None else "n/a"
        print(
            f"  {row['eps']:>5} | {ach:>12} | {nm:>10} | "
            f"{tl:>10} | {row['perplexity']:>10.4f} | {row['wall_time']:>8.1f}s"
        )

    print(f"\nTotal wall time: {total:.1f}s  ({total/60:.1f} min)")


if __name__ == "__main__":
    main()
