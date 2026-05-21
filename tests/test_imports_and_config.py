"""Test imports and configuration."""

import pytest
import yaml


def test_import_autonomy_stack():
    """Test basic package import."""
    import autonomy_stack
    assert autonomy_stack.__version__


def test_import_schemas():
    """Test schemas import."""
    from autonomy_stack.schemas import ControlAction, SafetyMetrics
    assert ControlAction
    assert SafetyMetrics


def test_import_agents():
    """Test agents module."""
    from autonomy_stack.agents import ExpertPolicy, StudentPolicy
    assert ExpertPolicy
    assert StudentPolicy


def test_import_models():
    """Test models module."""
    from autonomy_stack.models import DrivingPolicy, WorldModel
    assert DrivingPolicy
    assert WorldModel


def test_import_data_engine():
    """Test data engine modules."""
    from autonomy_stack.data_engine import EpisodeStore, FailureMiner, ModelCard, DatasetCard
    assert EpisodeStore
    assert FailureMiner
    assert ModelCard
    assert DatasetCard


def test_import_evaluation():
    """Test evaluation module."""
    from autonomy_stack.evaluation import MetricsEvaluator
    assert MetricsEvaluator


def test_import_training():
    """Test training modules."""
    from autonomy_stack.training import BCLoss, TD3BCLoss, ReplayBuffer, DAggerCollector
    assert BCLoss
    assert TD3BCLoss
    assert ReplayBuffer
    assert DAggerCollector


def test_config_names():
    """Test that config names are correct."""
    configs = [
        'configs/behavior_cloning.yaml',
        'configs/dagger_loop.yaml',
        'configs/data_engine.yaml',
        'configs/deployment.yaml',
        'configs/evaluation.yaml',
        'configs/offline_rl.yaml',
        'configs/world_model.yaml',
    ]

    config_names = {
        'behavior_cloning.yaml': 'autonomy_stack_bc_rgb_state',
        'dagger_loop.yaml': 'autonomy_stack_dagger_loop',
        'data_engine.yaml': 'autonomy_stack_data_engine',
        'deployment.yaml': 'autonomy_stack_deployment_benchmark',
        'evaluation.yaml': 'autonomy_stack_closed_loop_eval',
        'offline_rl.yaml': 'autonomy_stack_offline_rl',
        'world_model.yaml': 'autonomy_stack_future_prediction',
    }

    for config_file, expected_name in config_names.items():
        try:
            with open(f'configs/{config_file}', 'r') as f:
                config = yaml.safe_load(f)
                assert config.get('name') == expected_name, f"Config {config_file} has wrong name"
        except FileNotFoundError:
            pass
