"""Utilities for reproducible ML experiments."""

import random
import numpy as np
import torch


def set_seed(seed: int):
    """Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_config_hash(config: dict) -> str:
    """Get hash of config dictionary for reproducibility tracking.

    Args:
        config: Configuration dictionary

    Returns:
        Hash string
    """
    import hashlib
    config_str = str(sorted(config.items()))
    return hashlib.md5(config_str.encode()).hexdigest()[:8]
