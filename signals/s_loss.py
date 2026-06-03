import torch


def s_loss(logprobs: torch.Tensor) -> float:
    return logprobs.mean().item()
