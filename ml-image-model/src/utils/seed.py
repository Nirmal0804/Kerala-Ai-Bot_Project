"""Reproducibility helper functions."""

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seed across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
