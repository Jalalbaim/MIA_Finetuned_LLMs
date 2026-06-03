import math
import zlib

import torch


def s_zlib(text: str, logprobs: torch.Tensor) -> float:
    compressed_len = len(zlib.compress(text.encode("utf-8")))
    if compressed_len == 0:
        return math.nan
    return logprobs.mean().item() / compressed_len
