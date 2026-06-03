import math

import torch


def s_mink(logprobs: torch.Tensor, k_fraction: float = 0.20) -> float:
    n = len(logprobs)
    # logprobs has length seq_len - 1; fewer than 5 tokens → n < 4
    if n < 4:
        return math.nan
    k = max(1, math.ceil(k_fraction * n))
    lowest_k = torch.topk(logprobs, k, largest=False).values
    return lowest_k.mean().item()
