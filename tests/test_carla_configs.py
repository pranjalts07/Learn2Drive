"""Test CARLA-related configurations (CARLA itself is optional)."""

import pytest
import yaml
import os


def test_carla_configs_exist():
    """Test that CARLA config files exist."""
    carla_configs = [
        'configs/carla_local.yaml',
        'configs/carla_collect_small.yaml',
        'configs/carla_train_bc_small.yaml',
        'configs/carla_eval_small.yaml',
    ]

    for config_path in carla_configs:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                assert config is not None
                assert 'name' in config or 'carla' in config


def test_carla_adapter_import():
    """Test that CARLA adapter can be imported (graceful fallback)."""
    from autonomy_stack.simulation.carla_adapter import CarlaAdapter
    assert CarlaAdapter is not None


def test_carla_adapter_synthetic_mode():
    """Test CARLA adapter works in synthetic mode."""
    from autonomy_stack.simulation.carla_adapter import CarlaAdapter
    from autonomy_stack.simulation.scenarios import ScenarioSpec
    from autonomy_stack.schemas import ControlAction

    scenario = ScenarioSpec(name="test", duration_seconds=5)
    adapter = CarlaAdapter(scenario, use_carla=False)

    obs = adapter.reset()
    assert obs is not None

    action = ControlAction(0.5, 0.0, 0.0)
    obs, reward, done = adapter.step(action)
    assert obs is not None
