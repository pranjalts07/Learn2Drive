"""Episode storage and management."""

import os
import pickle
from typing import List, Optional
from autonomy_stack.schemas import Episode


class EpisodeStore:
    """Persistent episode storage."""

    def __init__(self, storage_dir: str = "data/episodes"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        self.episodes = []

    def add_episode(self, episode: Episode, episode_id: str):
        """Store episode to disk.

        Args:
            episode: Episode trajectory
            episode_id: Unique identifier for episode
        """
        path = os.path.join(self.storage_dir, f"{episode_id}.pkl")
        with open(path, 'wb') as f:
            pickle.dump(episode, f)
        self.episodes.append(episode_id)

    def load_episode(self, episode_id: str) -> Optional[Episode]:
        """Load episode from disk.

        Args:
            episode_id: Episode identifier

        Returns:
            Episode or None if not found
        """
        path = os.path.join(self.storage_dir, f"{episode_id}.pkl")
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f)
        return None

    def list_episodes(self) -> List[str]:
        """List all stored episode IDs."""
        return self.episodes

    def __len__(self) -> int:
        return len(self.episodes)
