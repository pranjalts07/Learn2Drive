"""Replay buffer for offline RL."""

from collections import deque
import numpy as np
from typing import Tuple, List
from autonomy_stack.schemas import Episode


class ReplayBuffer:
    """Experience replay buffer for offline RL training."""

    def __init__(self, max_size: int = 100000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def add(self, episode: Episode):
        """Add episode to buffer.

        Args:
            episode: Episode trajectory
        """
        self.buffer.append(episode)

    def sample(self, batch_size: int) -> Tuple[List, List, List, List]:
        """Sample batch from buffer.

        Args:
            batch_size: Number of episodes to sample

        Returns:
            Tuple of (observations, actions, rewards, dones)
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)

        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        episodes = [self.buffer[i] for i in indices]

        observations = [obs for ep in episodes for obs in ep.observations]
        actions = [act for ep in episodes for act in ep.actions]
        rewards = [r for ep in episodes for r in ep.rewards]
        dones = [d for ep in episodes for d in ep.dones]

        return observations, actions, rewards, dones

    def __len__(self) -> int:
        return len(self.buffer)
