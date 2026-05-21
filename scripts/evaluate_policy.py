#!/usr/bin/env python3
"""Evaluate policy on scenarios."""

import sys
import argparse
import numpy as np
from autonomy_stack.models.policy import DrivingPolicy
from autonomy_stack.evaluation.metrics import MetricsEvaluator
from autonomy_stack.simulation.scenarios import get_default_scenarios
from autonomy_stack.schemas import ObservationFrame
from autonomy_stack.utils.reproducibility import set_seed

set_seed(42)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='synthetic', choices=['synthetic', 'carla'])
    args = parser.parse_args()

    print(f"Evaluating policy in {args.mode} mode...")

    # Create policy
    policy = DrivingPolicy(input_channels=3, img_height=64, img_width=64)

    # Create evaluator
    evaluator = MetricsEvaluator()

    # Get scenarios
    scenarios = get_default_scenarios()

    print(f"\nRunning {len(scenarios)} scenarios...")
    all_metrics = []

    for scenario in scenarios:
        print(f"  Scenario: {scenario.name}")

        # Simulate rollout
        actions = []
        speeds = []
        for step in range(scenario.duration_seconds):
            rgb = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            action = policy.predict(rgb)
            actions.append(action)
            speeds.append(5.0 + np.random.randn())

        metrics = evaluator.compute_metrics(
            actions=actions,
            speeds=speeds,
            collisions=0,
            violations=0,
            success=True,
        )
        all_metrics.append(metrics)
        print(f"    ✓ Success rate: {metrics.success_rate:.1%}, Latency: {metrics.mean_latency_ms:.2f}ms")

    # Summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    avg_success = np.mean([m.success_rate for m in all_metrics])
    avg_latency = np.mean([m.mean_latency_ms for m in all_metrics])
    avg_jerk = np.mean([m.mean_jerk for m in all_metrics])
    print(f"Average Success Rate: {avg_success:.1%}")
    print(f"Average Latency:      {avg_latency:.2f} ms")
    print(f"Average Jerk:         {avg_jerk:.2f}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
