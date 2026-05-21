"""Dataset Aggregation (DAgger) for imitation learning."""

from typing import List
from autonomy_stack.schemas import Episode


class DAggerCollector:
    """Collects data using DAgger procedure."""

    def __init__(self, expert_policy, student_policy, config: dict = None):
        self.expert = expert_policy
        self.student = student_policy
        self.config = config or {}
        self.collected_episodes = []

    def collect_episode(self, episode: Episode, relabel: bool = True) -> Episode:
        """Collect episode, optionally relabeling with expert actions.

        Args:
            episode: Episode trajectory
            relabel: Whether to relabel with expert actions

        Returns:
            Episode with expert labels (if relabel=True)
        """
        if relabel:
            for i, obs in enumerate(episode.observations[:-1]):
                expert_action = self.expert.predict(obs)
                episode.actions[i] = expert_action

        self.collected_episodes.append(episode)
        return episode

    def get_collected_data(self) -> List[Episode]:
        """Get all collected episodes."""
        return self.collected_episodes
