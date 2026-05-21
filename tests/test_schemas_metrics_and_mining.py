"""Test schemas, metrics, and failure mining."""

import pytest
import numpy as np
from autonomy_stack.schemas import ControlAction, ObservationFrame, Episode, SafetyMetrics
from autonomy_stack.evaluation.metrics import MetricsEvaluator
from autonomy_stack.data_engine.failure_mining import FailureMiner


def test_control_action():
    """Test control action schema."""
    action = ControlAction(throttle=0.5, brake=0.0, steering=0.0)
    assert action.throttle == 0.5
    assert action.brake == 0.0
    assert action.steering == 0.0


def test_control_action_clamping():
    """Test control action bounds."""
    action = ControlAction(throttle=1.5, brake=-0.5, steering=2.0)
    assert 0 <= action.throttle <= 1
    assert 0 <= action.brake <= 1
    assert -1 <= action.steering <= 1


def test_observation_frame():
    """Test observation frame."""
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    obs = ObservationFrame(rgb_image=rgb)
    assert obs.rgb_image.shape == (64, 64, 3)
    assert obs.speed == 0.0


def test_episode():
    """Test episode schema."""
    observations = [ObservationFrame(np.zeros((64, 64, 3)))]
    actions = [ControlAction(0.5, 0.0, 0.0)]
    rewards = [1.0]
    dones = [False]

    episode = Episode(observations, actions, rewards, dones)
    assert episode.length == 1


def test_safety_metrics():
    """Test safety metrics."""
    metrics = SafetyMetrics(
        collision_count=0,
        violation_count=0,
        success_rate=1.0,
        mean_latency_ms=10.0,
    )
    assert metrics.collision_count == 0
    assert metrics.success_rate == 1.0


def test_metrics_evaluator():
    """Test metrics evaluation."""
    evaluator = MetricsEvaluator()

    actions = [
        ControlAction(0.5, 0.0, 0.0),
        ControlAction(0.6, 0.0, 0.1),
    ]
    speeds = [5.0, 5.5]

    metrics = evaluator.compute_metrics(
        actions=actions,
        speeds=speeds,
        collisions=0,
        violations=0,
        success=True,
    )

    assert metrics.collision_count == 0
    assert metrics.violation_count == 0
    assert metrics.success_rate == 1.0


def test_failure_miner():
    """Test failure mining."""
    miner = FailureMiner()

    actions = [ControlAction(0.5, 0.0, 0.0)] * 5
    obs = [ObservationFrame(np.zeros((64, 64, 3)))] * 5
    episode = Episode(obs, actions, [1.0] * 4, [False] * 5)

    metrics = SafetyMetrics(
        collision_count=1,
        success_rate=0.0,
    )

    is_failure, reason = miner.detect_failures(episode, metrics)
    assert is_failure
    assert reason == "collision"


def test_hard_case_ranking():
    """Test hard-case ranking."""
    miner = FailureMiner()

    episodes = [
        Episode([ObservationFrame(np.zeros((64, 64, 3)))], [ControlAction(0.5, 0.0, 0.0)], [1.0], [False]),
        Episode([ObservationFrame(np.zeros((64, 64, 3)))], [ControlAction(0.5, 0.0, 0.0)], [1.0], [False]),
    ]

    metrics = [
        SafetyMetrics(collision_count=0, success_rate=1.0),
        SafetyMetrics(collision_count=2, success_rate=0.5),
    ]

    ranked = miner.rank_hard_cases(episodes, metrics)
    assert len(ranked) == 2
    # Second episode should have higher difficulty score
    assert ranked[0][2] > ranked[1][2]
