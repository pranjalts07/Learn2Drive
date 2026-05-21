# Learn2Drive

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![pytest](https://img.shields.io/badge/pytest-9.0-green?logo=pytest&logoColor=white)](https://pytest.org/)
[![License MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CARLA Optional](https://img.shields.io/badge/CARLA-Optional-blue)](http://carla.org/)

A small autonomy machine learning project for driving data, failure mining, policy learning, safety metrics, and latency testing.

## What This Is

Learn2Drive is a Python and PyTorch project that shows how an autonomous driving learning pipeline can be organized. It creates driving scenarios, logs episodes, finds failure cases, trains small model components, evaluates safety metrics, and measures inference speed.

The default demo runs without CARLA so the whole pipeline can be tested on a normal laptop. CARLA support is optional for simulator-backed experiments.

## What It Does

- Creates synthetic driving scenarios
- Logs driving episodes with actions, events, and metadata
- Finds failure cases such as collisions, low route completion, high jerk, and interventions
- Selects hard cases for retraining
- Runs behavior cloning, offline RL, and world model training steps
- Computes safety metrics
- Benchmarks model latency, FPS, parameter count, and model size
- Generates model and dataset cards
- Provides an optional CARLA integration path

## Why I Built It

I built this project to understand the full engineering loop behind learning-based driving systems. Instead of only training one model, I wanted to build the surrounding system: data collection, failure analysis, model updates, safety reporting, and deployment checks.

## Current Results

These are local synthetic pipeline results. They show that the software pipeline works. They are not real CARLA benchmark results.

**Test Coverage and Compilation**
- 47 tests passed
- All code compiles without errors

**Synthetic Pipeline Run**
- 3 scenarios executed
- 3 episodes generated (60 total steps)
- 0 failure cases detected in synthetic data
- 3 episodes ranked by difficulty
- 100% success rate on all scenarios

**Dual Policy Models**

The project includes two policy architectures to demonstrate model-size and latency tradeoffs:

| Model | Parameters | Size | p50 Latency | p95 Latency | FPS | Use Case |
|-------|-----------|------|-------------|-------------|-----|----------|
| Compact | 604K | 2.31 MB | 0.64 ms | 1.16 ms | 1353 | Fast baseline, smoke tests |
| Medium | 2.84M | 10.84 MB | 1.58 ms | 2.62 ms | 588 | Portfolio demonstration |

**Safety Metrics**
- 0 collisions
- 0 traffic violations
- Mean jerk: 0.32

**Reports Generated**
- Model comparison report (reports/model_comparison.md)
- Model card created
- Dataset card created

## Architecture

The pipeline flows through these stages:

1. Scenario generation
2. Episode logging
3. Failure mining
4. Hard-case selection
5. Policy training steps
6. World model training step
7. Safety evaluation
8. Latency benchmarking
9. Reports and cards

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

pytest
python scripts/run_end_to_end_demo.py
python scripts/benchmark_policy.py --model compact
python scripts/benchmark_policy.py --model medium
python scripts/compare_policies.py
python scripts/evaluate_policy.py --mode synthetic
```

On Windows:
```bash
.venv\Scripts\activate
```

## What This Project Proves

- The codebase is importable and tested
- The pipeline runs end to end
- Failure mining and hard-case selection work on generated episodes
- Training steps run and return real losses
- Latency and model size are measured
- Reports and cards are generated

## What This Project Does Not Claim

- It does not claim real-world driving performance
- It does not claim official CARLA leaderboard results
- It does not claim production deployment
- Synthetic results are used for local validation

## Roadmap

- Collect a small real CARLA dataset
- Train the policy for multiple epochs on collected episodes
- Compare baseline and trained policy safety metrics
- Add stronger future occupancy or risk prediction
- Add a small dashboard for browsing failures and reports

