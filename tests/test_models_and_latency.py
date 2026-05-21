"""Test models and latency profiling."""

import pytest
import numpy as np
import torch
from autonomy_stack.models.policy import DrivingPolicy, CompactCNNPolicy, MediumCNNPolicy
from autonomy_stack.models.world_model import WorldModel
from autonomy_stack.optimization.latency import LatencyProfiler
from autonomy_stack.schemas import ObservationFrame


def test_compact_policy_creation():
    """Test compact policy model creation."""
    policy = CompactCNNPolicy(input_channels=3, img_height=64, img_width=64)
    assert policy is not None
    param_count = policy.count_parameters()
    assert param_count > 100000


def test_medium_policy_creation():
    """Test medium policy model creation."""
    policy = MediumCNNPolicy(input_channels=3, img_height=64, img_width=64)
    assert policy is not None
    param_count = policy.count_parameters()
    assert param_count > 1000000  # Medium should have >1M parameters


def test_medium_policy_larger_than_compact():
    """Test that medium policy has more parameters than compact."""
    compact = CompactCNNPolicy()
    medium = MediumCNNPolicy()
    assert medium.count_parameters() > compact.count_parameters()


def test_policy_creation():
    """Test policy model creation (alias test)."""
    policy = DrivingPolicy(input_channels=3, img_height=64, img_width=64)
    assert policy is not None
    param_count = policy.count_parameters()
    assert param_count > 100000


def test_policy_inference():
    """Test policy inference."""
    policy = DrivingPolicy(input_channels=3, img_height=64, img_width=64)
    rgb = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

    action = policy.predict(rgb)
    assert action.throttle >= 0
    assert action.brake >= 0
    assert -1 <= action.steering <= 1


def test_world_model_creation():
    """Test world model creation."""
    model = WorldModel(input_channels=3, latent_dim=32)
    assert model is not None


def test_latency_profiler():
    """Test latency profiling."""
    policy = DrivingPolicy(input_channels=3, img_height=64, img_width=64)
    profiler = LatencyProfiler()

    results = profiler.profile_inference(policy, (1, 3, 64, 64), num_runs=10)

    assert 'p50_ms' in results
    assert 'p95_ms' in results
    assert 'mean_ms' in results
    assert 'fps' in results
    assert results['fps'] > 0


def test_model_size():
    """Test model size computation."""
    policy = DrivingPolicy(input_channels=3, img_height=64, img_width=64)
    profiler = LatencyProfiler()
    size_mb = profiler.get_model_size_mb(policy)
    assert size_mb > 0
    assert size_mb < 10  # Should be less than 10 MB


def test_forward_pass():
    """Test model forward pass."""
    policy = DrivingPolicy(input_channels=3, img_height=64, img_width=64)
    x = torch.randn(1, 3, 64, 64)
    throttle, brake, steering = policy.forward(x)

    assert throttle.shape == (1, 1)
    assert brake.shape == (1, 1)
    assert steering.shape == (1, 1)
    assert 0 <= throttle.item() <= 1
    assert 0 <= brake.item() <= 1
    assert -1 <= steering.item() <= 1
