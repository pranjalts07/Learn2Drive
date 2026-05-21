"""Test episode storage."""

import pytest
import tempfile
import os
from autonomy_stack.data_engine.episode_store import EpisodeStore
from autonomy_stack.schemas import Episode, ControlAction, ObservationFrame
import numpy as np


def test_episode_store_creation():
    """Test episode store creation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = EpisodeStore(storage_dir=tmpdir)
        assert len(store) == 0


def test_add_and_load_episode():
    """Test adding and loading episodes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = EpisodeStore(storage_dir=tmpdir)

        obs = [ObservationFrame(np.zeros((64, 64, 3)))]
        actions = [ControlAction(0.5, 0.0, 0.0)]
        episode = Episode(obs, actions, [1.0], [False])

        store.add_episode(episode, "test_ep")
        assert len(store) == 1

        loaded = store.load_episode("test_ep")
        assert loaded is not None
        assert loaded.length == 1


def test_list_episodes():
    """Test listing episodes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = EpisodeStore(storage_dir=tmpdir)

        for i in range(3):
            obs = [ObservationFrame(np.zeros((64, 64, 3)))]
            actions = [ControlAction(0.5, 0.0, 0.0)]
            episode = Episode(obs, actions, [1.0], [False])
            store.add_episode(episode, f"ep_{i}")

        episodes = store.list_episodes()
        assert len(episodes) == 3
