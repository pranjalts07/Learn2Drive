#!/usr/bin/env python3
"""Benchmark driving policy latency and model size."""

import sys
import argparse
from autonomy_stack.models.policy import CompactCNNPolicy, MediumCNNPolicy
from autonomy_stack.optimization.latency import LatencyProfiler


def main():
    """Run benchmark."""
    parser = argparse.ArgumentParser(description="Benchmark policy models")
    parser.add_argument("--model", choices=["compact", "medium"], default="compact",
                        help="Which model to benchmark")
    args = parser.parse_args()

    print("=" * 60)
    print("Learn2Drive Policy Benchmark")
    print("=" * 60)

    # Create policy
    print(f"\nCreating {args.model} policy model...")
    if args.model == "compact":
        policy = CompactCNNPolicy(input_channels=3, img_height=64, img_width=64)
        model_type = "Compact"
    else:
        policy = MediumCNNPolicy(input_channels=3, img_height=64, img_width=64)
        model_type = "Medium"

    param_count = policy.count_parameters()
    print(f"✓ {model_type} Policy: {param_count:,} parameters")

    # Benchmark
    print("\nProfiling inference (100 runs)...")
    profiler = LatencyProfiler()
    results = profiler.profile_inference(policy, (1, 3, 64, 64), num_runs=100)

    model_size = profiler.get_model_size_mb(policy)

    # Print results
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Model Size:        {model_size:.2f} MB")
    print(f"Parameters:        {param_count:,}")
    print(f"Latency (p50):     {results['p50_ms']:.2f} ms")
    print(f"Latency (p95):     {results['p95_ms']:.2f} ms")
    print(f"Latency (mean):    {results['mean_ms']:.2f} ms")
    print(f"Throughput (FPS):  {results['fps']:.1f}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
