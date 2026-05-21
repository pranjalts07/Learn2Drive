"""Automatic failure detection and hard-case mining."""

from typing import List, Tuple
import numpy as np
from autonomy_stack.schemas import Episode, SafetyMetrics
from autonomy_stack.utils.metrics_helpers import _mean_abs_jerk


class FailureMiner:
    """Mines failure cases from episodes for data curation."""

    def __init__(self, collision_threshold: float = 0.1):
        self.collision_threshold = collision_threshold
        self.failures = []

    def detect_failures(self, episode: Episode, metrics: SafetyMetrics) -> Tuple[bool, str]:
        """Detect if episode contains failure.

        Args:
            episode: Episode trajectory
            metrics: Safety metrics for episode

        Returns:
            Tuple of (is_failure, failure_reason)
        """
        if metrics.collision_count > 0:
            return True, "collision"

        if metrics.violation_count > 0:
            return True, "traffic_violation"

        if metrics.mean_jerk > 2.0:
            return True, "high_jerk"

        return False, "no_failure"

    def rank_hard_cases(self, episodes: List[Episode], metrics_list: List[SafetyMetrics]) -> List[Tuple[Episode, SafetyMetrics, float]]:
        """Rank episodes by difficulty for hard-case selection.

        Args:
            episodes: List of episodes
            metrics_list: Corresponding metrics

        Returns:
            Sorted list of (episode, metrics, difficulty_score) tuples
        """
        ranked = []
        for ep, met in zip(episodes, metrics_list):
            # Difficulty score: higher = harder
            score = (
                met.collision_count * 100 +
                met.violation_count * 50 +
                met.mean_jerk * 10 +
                (1.0 - met.success_rate) * 20
            )
            ranked.append((ep, met, score))

        ranked.sort(key=lambda x: x[2], reverse=True)
        return ranked

    def mine_failures(self, episodes: List[Episode], metrics_list: List[SafetyMetrics]) -> List[Episode]:
        """Extract failure episodes for data augmentation.

        Args:
            episodes: List of episodes
            metrics_list: Corresponding metrics

        Returns:
            List of failure episodes
        """
        failures = []
        for ep, met in zip(episodes, metrics_list):
            is_failure, _ = self.detect_failures(ep, met)
            if is_failure:
                failures.append(ep)
        self.failures.extend(failures)
        return failures
