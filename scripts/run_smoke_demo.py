#!/usr/bin/env python3
"""Quick smoke test for autonomy_stack (synthetic mode, no CARLA required)."""

import sys
import numpy as np
from autonomy_stack.models.policy import DrivingPolicy
from autonomy_stack.evaluation.metrics import MetricsEvaluator
from autonomy_stack.schemas import ControlAction, ObservationFrame
from autonomy_stack.utils.reproducibility import set_seed

set_seed(42)


def main():
    """Run smoke test."""
    print("=" * 60)
    print("Learn2Drive Smoke Demo (Synthetic Mode)")
    print("=" * 60)

    # Create policy
    print("\n1. Creating policy model...")
    policy = DrivingPolicy(input_channels=3, img_height=64, img_width=64)
    param_count = policy.count_parameters()
    print(f"   ✓ Policy created with {param_count:,} parameters")

    # Create evaluator
    print("\n2. Creating metrics evaluator...")
    evaluator = MetricsEvaluator()
    print("   ✓ Evaluator ready")

    # Run synthetic rollout
    print("\n3. Running synthetic rollout (10 steps)...")
    actions = []
    speeds = []

    for step in range(10):
        rgb = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        obs = ObservationFrame(rgb_image=rgb.astype(np.float32) / 255.0)

        action = policy.predict(obs)
        actions.append(action)
        speeds.append(5.0 + np.random.randn())

        print(f"   Step {step + 1}/10: throttle={action.throttle:.2f}, brake={action.brake:.2f}, steering={action.steering:.2f}")

    # Compute metrics
    print("\n4. Computing safety metrics...")
    metrics = evaluator.compute_metrics(
        actions=actions,
        speeds=speeds,
        collisions=0,
        violations=0,
        success=True,
        latencies_ms=[10.0] * len(actions),
    )
    print(f"   ✓ Success rate: {metrics.success_rate:.1%}")
    print(f"   ✓ Mean latency: {metrics.mean_latency_ms:.2f} ms")
    print(f"   ✓ Mean FPS: {metrics.mean_fps:.1f}")
    print(f"   ✓ Mean jerk: {metrics.mean_jerk:.2f}")

    print("\n" + "=" * 60)
    print("✓ Smoke demo completed successfully!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
