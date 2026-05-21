#!/usr/bin/env python3
"""Compare compact and medium policy models on latency and size metrics."""

import sys
import json
import time
import numpy as np
import torch
from autonomy_stack.models.policy import CompactCNNPolicy, MediumCNNPolicy
from autonomy_stack.optimization.latency import LatencyProfiler


def main():
    """Compare both policy models."""
    print("=" * 70)
    print("Learn2Drive Policy Comparison")
    print("=" * 70)
    print()

    models = {
        "Compact": CompactCNNPolicy(),
        "Medium": MediumCNNPolicy(),
    }

    results = {}

    for model_name, model in models.items():
        print(f"Benchmarking {model_name} Policy...")
        print("-" * 70)

        # Parameter count
        param_count = model.count_parameters()
        print(f"  Parameters:        {param_count:,}")

        # Model size
        profiler = LatencyProfiler()
        size_mb = profiler.get_model_size_mb(model)
        print(f"  Model Size:        {size_mb:.2f} MB")

        # Latency profiling
        latency_results = profiler.profile_inference(model, (1, 3, 64, 64), num_runs=100)
        print(f"  Latency (p50):     {latency_results['p50_ms']:.2f} ms")
        print(f"  Latency (p95):     {latency_results['p95_ms']:.2f} ms")
        print(f"  Latency (mean):    {latency_results['mean_ms']:.2f} ms")
        print(f"  Throughput (FPS):  {latency_results['fps']:.1f}")

        # Test prediction
        print(f"  Output Check:      ", end="")
        rgb = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        action = model.predict(rgb)
        assert 0 <= action.throttle <= 1
        assert 0 <= action.brake <= 1
        assert -1 <= action.steering <= 1
        print("OK")

        results[model_name] = {
            "parameters": param_count,
            "size_mb": round(size_mb, 2),
            "p50_latency_ms": round(latency_results["p50_ms"], 2),
            "p95_latency_ms": round(latency_results["p95_ms"], 2),
            "mean_latency_ms": round(latency_results["mean_ms"], 2),
            "fps": round(latency_results["fps"], 1),
        }

        print()

    # Generate comparison report
    report = """
# Policy Model Comparison

These are local CPU benchmarks. They show model size, inference speed, and parameter count.
They do not represent real driving performance or CARLA benchmark results.

## Summary

| Model | Parameters | Size (MB) | p50 (ms) | p95 (ms) | FPS | Purpose |
|-------|-----------|-----------|----------|----------|-----|---------|
"""

    report += f"| Compact | {results['Compact']['parameters']:,} | {results['Compact']['size_mb']} | {results['Compact']['p50_latency_ms']} | {results['Compact']['p95_latency_ms']} | {results['Compact']['fps']} | Fast baseline |\n"
    report += f"| Medium  | {results['Medium']['parameters']:,} | {results['Medium']['size_mb']} | {results['Medium']['p50_latency_ms']} | {results['Medium']['p95_latency_ms']} | {results['Medium']['fps']} | Portfolio model |\n"

    report += """
## Details

### Compact CNN Policy

- Used for smoke tests and local validation
- Designed for fast inference and small memory footprint
- Baseline for speed comparison

### Medium CNN Policy

- Larger architecture with more capacity
- Used to demonstrate model-size and latency tradeoffs
- Representative of medium-complexity policies

## What This Does Not Claim

- Real driving performance
- Official CARLA leaderboard results
- Production deployment
- Model quality rankings (both are equal on synthetic data)

## Next Steps

These models could be extended by:
- Training on real CARLA-collected episodes
- Comparing loss values after training
- Evaluating on challenge scenarios
- Profiling on actual deployment hardware
"""

    # Write reports
    with open("reports/model_comparison.md", "w") as f:
        f.write(report)

    with open("reports/model_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    print("=" * 70)
    print("Reports Generated")
    print("=" * 70)
    print(f"  reports/model_comparison.md")
    print(f"  reports/model_comparison.json")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
