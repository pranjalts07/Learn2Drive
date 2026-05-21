
# Policy Model Comparison

These are local CPU benchmarks. They show model size, inference speed, and parameter count.
They do not represent real driving performance or CARLA benchmark results.

## Summary

| Model | Parameters | Size (MB) | p50 (ms) | p95 (ms) | FPS | Purpose |
|-------|-----------|-----------|----------|----------|-----|---------|
| Compact | 604,323 | 2.31 | 0.64 | 1.16 | 1353.3 | Fast baseline |
| Medium  | 2,840,643 | 10.84 | 1.58 | 2.62 | 587.8 | Portfolio model |

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
