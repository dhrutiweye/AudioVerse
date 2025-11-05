from .QualitiHelper import compute_quality_signals
import torch

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


__all__ = [
    "compute_quality_signals",
    'get_device'
]