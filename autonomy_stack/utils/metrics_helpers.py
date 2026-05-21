"""Shared utility functions for metrics computation."""

import numpy as np


def _mean_abs_jerk(accelerations: np.ndarray) -> float:
    """Compute mean absolute jerk from accelerations.

    Jerk is the rate of change of acceleration.

    Args:
        accelerations: Array of acceleration values

    Returns:
        Mean absolute jerk
    """
    if len(accelerations) < 2:
        return 0.0

    jerks = np.abs(np.diff(accelerations))
    return float(np.mean(jerks))
