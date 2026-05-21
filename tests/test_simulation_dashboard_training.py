"""Test simulation, dashboard, and training integration."""

import pytest
import numpy as np
from autonomy_stack.simulation.scenarios import ScenarioSpec, get_default_scenarios
from autonomy_stack.simulation.carla_adapter import CarlaAdapter
from autonomy_stack.models.policy import DrivingPolicy
from autonomy_stack.training.bc import BCLoss
from autonomy_stack.training.offline_rl import TD3BCLoss
from autonomy_stack.training.replay_buffer import ReplayBuffer
from autonomy_stack.training.dagger import DAggerCollector
from autonomy_stack.agents.expert import ExpertPolicy
from autonomy_stack.agents.student import StudentPolicy
from autonomy_stack.schemas import ControlAction, ObservationFrame, Episode


def test_scenario_spec():
    """Test scenario specification."""
    scenario = ScenarioSpec(
        name="test",
        duration_seconds=30,
        num_vehicles=5,
        weather="rain",
    )
    assert scenario.name == "test"
    assert scenario.duration_seconds == 30
    assert scenario.num_vehicles == 5


def test_get_default_scenarios():
    """Test default scenarios."""
    scenarios = get_default_scenarios()
    assert len(scenarios) == 3
    assert scenarios[0].name == "simple_straight"


def test_carla_adapter_synthetic_fallback():
    """Test CARLA adapter fallback to synthetic."""
    scenario = ScenarioSpec(name="test", duration_seconds=10)
    adapter = CarlaAdapter(scenario, use_carla=False)

    obs = adapter.reset()
    assert obs is not None

    for _ in range(5):
        action = ControlAction(0.5, 0.0, 0.0)
        obs, reward, done = adapter.step(action)
        assert obs is not None
        assert reward >= 0
        assert isinstance(done, bool)


def test_bc_loss():
    """Test behavior cloning loss."""
    import torch
    bc_loss = BCLoss()

    throttle_pred = torch.tensor([[0.5]], requires_grad=True)
    brake_pred = torch.tensor([[0.0]], requires_grad=True)
    steering_pred = torch.tensor([[0.1]], requires_grad=True)

    throttle_gt = torch.tensor([[0.6]])
    brake_gt = torch.tensor([[0.1]])
    steering_gt = torch.tensor([[0.0]])

    loss = bc_loss(
        (throttle_pred, brake_pred, steering_pred),
        (throttle_gt, brake_gt, steering_gt),
    )
    assert loss.item() > 0


def test_offline_rl_loss():
    """Test offline RL loss."""
    rl_loss = TD3BCLoss(alpha=0.5)
    td_loss = 0.1
    bc_loss = 0.05
    loss = rl_loss(td_loss, bc_loss)
    assert loss > 0


def test_replay_buffer():
    """Test replay buffer."""
    buffer = ReplayBuffer(max_size=1000)

    obs = [ObservationFrame(np.zeros((64, 64, 3)))]
    actions = [ControlAction(0.5, 0.0, 0.0)]
    episode = Episode(obs, actions, [1.0], [False])

    buffer.add(episode)
    assert len(buffer) == 1

    observations, actions_sampled, rewards, dones = buffer.sample(1)
    assert len(observations) == 1


def test_dagger_collector():
    """Test DAgger collector."""
    expert = ExpertPolicy({'max_throttle': 0.8})
    student = StudentPolicy()
    collector = DAggerCollector(expert, student)

    obs = [ObservationFrame(np.zeros((64, 64, 3)))]
    actions = [ControlAction(0.5, 0.0, 0.0)]
    episode = Episode(obs, actions, [1.0], [False])

    collected = collector.collect_episode(episode, relabel=True)
    assert collected is not None
    assert len(collector.get_collected_data()) == 1


def test_expert_policy():
    """Test expert policy."""
    expert = ExpertPolicy({'max_throttle': 0.8})
    obs = ObservationFrame(np.zeros((64, 64, 3)))
    action = expert.predict(obs)
    assert action.throttle >= 0


def test_student_policy():
    """Test student policy."""
    student = StudentPolicy()
    obs = ObservationFrame(np.zeros((64, 64, 3)))
    action = student.predict(obs)
    assert action is not None


def test_integration_scenario_and_policy():
    """Test integration: scenario + policy + evaluation."""
    from autonomy_stack.evaluation.metrics import MetricsEvaluator

    scenario = ScenarioSpec(name="test", duration_seconds=5)
    policy = DrivingPolicy()
    evaluator = MetricsEvaluator()

    actions = []
    speeds = []
    for _ in range(5):
        rgb = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        action = policy.predict(rgb)
        actions.append(action)
        speeds.append(5.0)

    metrics = evaluator.compute_metrics(actions, speeds, success=True)
    assert metrics.success_rate == 1.0
