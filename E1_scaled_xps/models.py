"""
Model registry for E1.

The workshop pipeline hardcodes one model: config.MODEL_NAME plus a
'gpt_neo_ft_...' filename prefix in eleven files, and LoRA/MiCA target modules
resolved against GPT-Neo's q_proj/v_proj. Pythia is GPTNeoX and exposes a
*fused* query_key_value projection instead, so that resolution silently fails
on Pythia. This module centralises everything that varies with model family.

Deduped Pythia variants are used deliberately: they partially answer the
"Enron is already in the Pile" contamination objection, and the paper should
say so explicitly.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_E1_DIR = Path(__file__).parent.resolve()
if str(_E1_DIR) not in sys.path:
    sys.path.insert(0, str(_E1_DIR))

from config_e1 import DEFAULT_DTYPE


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    n_params: int                    # reference total param count, for the scaling fit
    per_device_batch: int            # fits 16GB P100 at 256 tokens, fp16 autocast, fp32 master weights
    peft_target_modules: list[str]   # unused by E1 (full FT only); E5 imports this
    needs_8bit_adam: bool = False
    needs_grad_checkpointing: bool = False


# per_device_batch assumes seq_len=256 and fp32 master weights with fp16
# autocast. grad-accum brings every model to the same effective batch of 32,
# so the optimisation trajectory is comparable across sizes (required by E3).
MODELS: dict[str, ModelSpec] = {
    "pythia-70m": ModelSpec(
        key="pythia-70m",
        hf_id="EleutherAI/pythia-70m-deduped",
        n_params=70_426_624,
        per_device_batch=32,
        peft_target_modules=["query_key_value"],
    ),
    "pythia-160m": ModelSpec(
        key="pythia-160m",
        hf_id="EleutherAI/pythia-160m-deduped",
        n_params=162_322_944,
        per_device_batch=16,
        peft_target_modules=["query_key_value"],
    ),
    "pythia-410m": ModelSpec(
        key="pythia-410m",
        hf_id="EleutherAI/pythia-410m-deduped",
        n_params=405_334_016,
        per_device_batch=8,
        peft_target_modules=["query_key_value"],
    ),
    # E3 scaling points. Do not fit a P100 for full fine-tuning: AdamW fp32
    # states alone are ~22GB at 1.4B. These run on RunPod A100 40GB.
    "pythia-1.4b": ModelSpec(
        key="pythia-1.4b",
        hf_id="EleutherAI/pythia-1.4b-deduped",
        n_params=1_414_647_808,
        per_device_batch=4,
        peft_target_modules=["query_key_value"],
        needs_8bit_adam=True,
    ),
    "pythia-2.8b": ModelSpec(
        key="pythia-2.8b",
        hf_id="EleutherAI/pythia-2.8b-deduped",
        n_params=2_775_208_960,
        per_device_batch=2,
        peft_target_modules=["query_key_value"],
        needs_8bit_adam=True,
        needs_grad_checkpointing=True,
    ),
    # E6 continuity check against the workshop results.
    "gpt-neo-125m": ModelSpec(
        key="gpt-neo-125m",
        hf_id="EleutherAI/gpt-neo-125m",
        n_params=125_198_592,
        per_device_batch=16,
        peft_target_modules=["q_proj", "v_proj"],
    ),
}


def get_spec(model_key: str) -> ModelSpec:
    if model_key not in MODELS:
        raise KeyError(
            f"Unknown model {model_key!r}. Known: {sorted(MODELS)}"
        )
    return MODELS[model_key]


def grad_accum_steps(model_key: str, effective_batch: int) -> int:
    """How many micro-batches to accumulate to reach effective_batch."""
    per_device = get_spec(model_key).per_device_batch
    return max(1, effective_batch // per_device)


# Precision

def resolve_dtype(device: torch.device) -> torch.dtype:
    """Autocast dtype. The P100 (sm_60) has no bf16 support, so E1 defaults to
    fp16 + dynamic loss scaling. On Ampere+ (T4 is sm_75 and also lacks bf16;
    A100 is sm_80 and has it) prefer bf16, which needs no GradScaler.

    Note this is the *autocast* dtype only. Weights are always held in fp32 as
    master weights -- loading weights directly in fp16 and stepping AdamW on
    them diverges, which is the bug in finetune/train.py (it loads bfloat16
    *and* autocasts, then wraps the step in a GradScaler with enabled=False)."""
    if device.type != "cuda":
        return torch.float32
    if DEFAULT_DTYPE == "bf16" or torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def needs_grad_scaler(dtype: torch.dtype) -> bool:
    return dtype == torch.float16


# Loading

def load_tokenizer(model_key: str) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(get_spec(model_key).hf_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base_model(
    model_key: str,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> AutoModelForCausalLM:
    """Load the pretrained model. dtype=None means fp32 master weights (for
    training); pass an explicit dtype for eval, where half precision is safe
    because there is no optimizer step."""
    spec = get_spec(model_key)
    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_id,
        torch_dtype=dtype if dtype is not None else torch.float32,
    ).to(device)
    return model


def load_checkpoint(
    path: Path,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> AutoModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(
        str(path),
        torch_dtype=dtype if dtype is not None else torch.float32,
    ).to(device)
    model.eval()
    return model


def count_params(model) -> int:
    """True parameter count, recorded per run. The n_params field above is a
    reference value; the scaling fit in E3 should use this."""
    return sum(p.numel() for p in model.parameters())


# Reference model P_pre

def reference_model_key(model_key: str) -> str:
    """The reference distribution for the Ref / RMIA / neighborhood attacks is
    the *same* architecture and size, untuned. Using a different size would
    conflate calibration with capacity.

    This is also the P_pre in KL(P_ft || P_pre), so it must be the exact model
    the fine-tune started from."""
    return model_key


def eval_devices() -> tuple[torch.device, torch.device]:
    """The 2xT4 trick from extension.md: pin P_ft to cuda:0 and P_pre to
    cuda:1 so both forward passes overlap instead of serialising on one card.
    Falls back to a single device (or CPU) when only one is visible."""
    if not torch.cuda.is_available():
        cpu = torch.device("cpu")
        return cpu, cpu
    n = torch.cuda.device_count()
    if n >= 2:
        return torch.device("cuda:0"), torch.device("cuda:1")
    dev = torch.device("cuda:0")
    return dev, dev
