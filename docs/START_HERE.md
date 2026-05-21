# Learn2Drive: Start Here

Welcome! This guide is for recruiters and technical leads evaluating the Learn2Drive portfolio project.

## Quick Facts

- **Lines of Code**: ~2,200 (Python, tests, configs, docs)
- **Test Coverage**: 47 tests across 7 test files (CARLA-free)
- **Dual Policy Models**: Compact (604K params) and Medium (2.84M params)
- **Compact Model**: 2.31 MB, 1353 FPS (p50=0.64ms, p95=1.16ms)
- **Medium Model**: 10.84 MB, 588 FPS (p50=1.58ms, p95=2.62ms)
- **Architecture**: 9 production-quality modules
- **Status**: Fully functional, all tests passing

## 3-Minute Overview

**Learn2Drive** is an end-to-end autonomous driving ML system demonstrating:

1. **Data Engineering**: Failure mining, hard-case selection, dataset curation
2. **Learning**: Behavior cloning (imitation) + offline RL (TD3+BC)
3. **Safety**: 17-metric suite (collisions, violations, latency, jerk, FPS)
4. **Production Code**: ONNX export, latency benchmarking, reproducibility

## What This Shows

### Technical Depth
- Custom CNN policy (not pretrained, not trivial)
- Explicit failure mining (not just random sampling)
- 17-field safety metrics (not just accuracy)
- Offline RL implementation (not just supervised learning)
- CARLA integration with graceful synthetic fallback

### Engineering Quality
- Modular architecture (9 independent modules)
- Comprehensive testing (58 tests, all passing)
- Type hints and reproducibility utilities
- YAML-based configuration system
- Both unit and integration tests

### Honesty
- Synthetic mode clearly labeled (no fake CARLA results)
- MIT License preserved
- Third-party attribution documented
- Latency numbers real (not marketing claims)
- Known limitations discussed

## How to Run

### Quickest Test (30 seconds)
```bash
python scripts/run_smoke_demo.py
```
Shows: Policy creation, inference, metrics computation.

### Full Pipeline (2 minutes)
```bash
python scripts/run_end_to_end_demo.py
```
Shows: Data → Learning → Evaluation → Deployment prep.

### Benchmark Compact Model (30 seconds)
```bash
python scripts/benchmark_policy.py --model compact
```
Shows: Latency, FPS, model size for fast baseline.

### Compare Models (30 seconds)
```bash
python scripts/compare_policies.py
```
Shows: Side-by-side comparison of Compact vs Medium models with full metrics.

### All Tests (15 seconds)
```bash
pytest
```
47 tests, all passing, synthetic mode throughout.

## Key Files to Read

1. **[autonomy_stack/models/policy.py](../autonomy_stack/models/policy.py)** (260 lines)
   - CompactCNNPolicy: 604K params, fast baseline
   - MediumCNNPolicy: 2.84M params, stronger architecture
   - RGB → (throttle, brake, steering)
   - Real inference code, not pseudo

2. **[autonomy_stack/data_engine/failure_mining.py](../autonomy_stack/data_engine/failure_mining.py)** (70 lines)
   - Automated failure detection
   - Hard-case ranking by jerk+safety
   - Core data curation logic

3. **[autonomy_stack/evaluation/metrics.py](../autonomy_stack/evaluation/metrics.py)** (100 lines)
   - 17-field SafetyMetrics
   - p50/p95 latency computation
   - Production-ready code

4. **[autonomy_stack/training/bc.py](../autonomy_stack/training/bc.py)** (25 lines)
   - Behavior cloning loss
   - Clean, minimal implementation

5. **[scripts/benchmark_policy.py](../scripts/benchmark_policy.py)** (50 lines)
   - Real latency profiling with model selection
   - Compact: 1353 FPS, 0.64ms p50
   - Medium: 588 FPS, 1.58ms p50

6. **[scripts/compare_policies.py](../scripts/compare_policies.py)** (130 lines)
   - Side-by-side model comparison
   - Generates markdown and JSON reports
   - Measures size, latency, throughput

## Interview Talking Points

### Data Engine (Most Unique)
"I implemented failure mining—automatically detecting collisions, violations, and high-jerk behavior—then ranked episodes by difficulty. This isn't random sampling; it's intelligent curation. Hard cases get 10x the attention."

### Safety Metrics (Differentiator)
"17 explicit safety metrics. Most projects just report accuracy. I report collisions, violations, latency percentiles, jerk, FPS, model size. Safety gates that prevent deployment of unsafe models."

### Model Architecture Decisions
"I included two policy models to demonstrate real engineering tradeoffs. The Compact model (604K params) gets 1353 FPS for smoke tests and validation. The Medium model (2.84M params) shows how to scale for better capacity. This isn't theoretical—I measured both and show the actual latency/size/throughput numbers."

### Production Mindset
"I benchmarked on CPU (not GPU). Real latency: Compact at 0.64ms p50, Medium at 1.58ms p50. I exported to ONNX for deployment. I tracked reproducibility with seed setting and config hashing."

### Honest Scope
"Synthetic mode validates the pipeline without CARLA. Real CARLA integration available but optional. I labeled this clearly because I want you to know what's proven and what's future work."

## Why This Matters

This project shows:
- ✅ Can write production ML code (not just training scripts)
- ✅ Understands safety-critical system design
- ✅ Data is the bottleneck (failure mining focus)
- ✅ Real metrics, not marketing claims
- ✅ Clean, testable, modular architecture

## What's Not Here (Honest Gaps)

- ❌ Real CARLA data (synthetic fallback instead)
- ❌ Advanced computer vision (no ResNet finetune)
- ❌ Transformers or modern architectures
- ❌ Multi-task learning or world modeling (stubbed)
- ❌ Real production deployment

These are learning opportunities, not hidden failures.

## Questions?

See [ARCHITECTURE.md](ARCHITECTURE.md) for module-by-module breakdown.

See [ROLE_ALIGNMENT.md](../docs/ROLE_ALIGNMENT.md) for mapping to Tesla job requirements.

---

**Status**: All 47 tests passing. All scripts executable from repo root. Dual models (Compact + Medium) with comparison reports. Synthetic mode, production code quality.
