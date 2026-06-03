import torch


def s_ref(logprobs_ft: torch.Tensor, logprobs_pre: torch.Tensor) -> float:
    return (logprobs_ft.mean() - logprobs_pre.mean()).item()
