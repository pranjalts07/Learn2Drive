"""Model comparison and evaluation utilities."""

from typing import Dict, List
from autonomy_stack.schemas import SafetyMetrics


class ModelComparison:
    """Compare multiple models on safety metrics."""

    def __init__(self):
        self.results = {}

    def add_result(self, model_name: str, metrics: SafetyMetrics):
        """Add evaluation result for a model.

        Args:
            model_name: Name of the model
            metrics: Safety metrics for this model
        """
        self.results[model_name] = metrics

    def rank_models(self) -> List[tuple]:
        """Rank models by overall performance.

        Returns:
            List of (model_name, score) sorted by score
        """
        scores = {}
        for name, metrics in self.results.items():
            score = (
                metrics.success_rate * 100 -
                metrics.collision_count * 50 -
                metrics.violation_count * 30 -
                metrics.mean_jerk * 5
            )
            scores[name] = score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked

    def to_dict(self) -> Dict:
        """Convert results to dictionary."""
        return {
            name: metrics.to_dict()
            for name, metrics in self.results.items()
        }
