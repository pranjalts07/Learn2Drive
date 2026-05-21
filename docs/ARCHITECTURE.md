# Architecture Overview

## Module Structure

Learn2Drive consists of 9 core modules + utilities:

```
autonomy_stack/
├── agents/           # Policy abstractions (expert, student)
├── models/           # Neural networks (policy CNN, world model)
├── training/         # Learning algorithms (BC, offline RL, DAgger)
├── data_engine/      # Data curation (storage, failure mining, cards)
├── evaluation/       # Safety metrics (17-field suite)
├── simulation/       # Environment (scenarios, CARLA adapter)
├── deployment/       # Production (ONNX export)
├── optimization/     # Benchmarking (latency, throughput)
└── utils/            # Helpers (metrics, reproducibility)
```

## Data Flow

```
[Expert Policy] → [Episode Collection] → [Failure Mining]
                                              ↓
                                    [Hard-Case Selection]
                                              ↓
                          [Episode Store] (persistent storage)
                                              ↓
                    [Replay Buffer] (training dataset)
                          ↓
              ┌───────────┼───────────┐
              ↓           ↓           ↓
        [BC Training] [Offline RL] [DAgger]
              ↓           ↓           ↓
              └───────────┼───────────┘
                          ↓
                  [Student Policy]
                          ↓
                  ┌──────────────┐
                  ↓              ↓
            [Evaluation]    [Deployment]
         (17 metrics)      (ONNX export)
                  ↓              ↓
            [Safety Report]  [Benchmark]
```

## Key Design Decisions

### 1. Failure Mining (data_engine/failure_mining.py)
**Why**: Most autonomous driving systems do random data selection. We explicitly mine failures.

**How**: 
- Collision detection → label as failure
- Traffic violation detection → label as failure
- High jerk detection → label as failure
- Rank episodes by difficulty score: jerk + safety violations

**Impact**: Hard cases get concentrated attention during training.

### 2. 17-Field Safety Metrics (evaluation/metrics.py)
**Why**: Standard ML focuses on accuracy. Autonomous driving must focus on safety.

**Fields**:
- Collision count
- Traffic violations
- Success rate
- Latency (p50, p95, mean)
- FPS / throughput
- Jerk (smoothness)
- Max acceleration
- Mean speed
- Route completion
- Comfort score
- Offroad incidents
- Near-miss count
- Inference time
- Model size

**Gate**: Deploy only if all safety gates pass.

### 3. Synthetic + CARLA (simulation/carla_adapter.py)
**Why**: CARLA is optional, not mandatory. Honest scope.

**How**:
- Default: Synthetic mode (no CARLA required)
- Optional: Real CARLA with graceful fallback
- Same interface, different implementations

**Benefit**: Works on any machine, validates pipeline, CARLA as enhancement.

### 4. Modular Training (training/)
**Three algorithms, same interface**:
- BC (BCLoss): Pure imitation learning
- Offline RL (TD3BCLoss): Offline RL with BC regularization
- DAgger (DAggerCollector): Interactive imitation learning

**Why**: Different problems need different approaches.

### 5. Reproducibility (utils/reproducibility.py)
**Seed setting**: Deterministic results
**Config hashing**: Track which config generated which results
**Explicit randomness**: No hidden random state

## Module Responsibilities

### agents/
- `ExpertPolicy`: Reference policy for data collection
- `StudentPolicy`: Learned policy wrapper

### models/
- `DrivingPolicy`: CNN (191K params) → control action
- `WorldModel`: Encoder-decoder for future prediction

### training/
- `BCLoss`: Behavior cloning loss (MSE)
- `TD3BCLoss`: Offline RL (TD3 + BC)
- `ReplayBuffer`: Experience storage
- `DAggerCollector`: Interactive learning
- `WorldModelLoss`: Frame prediction loss

### data_engine/
- `EpisodeStore`: Disk-based episode persistence
- `FailureMiner`: Failure detection + ranking
- `ModelCard`: ML model metadata
- `DatasetCard`: Dataset metadata
- `ModelComparison`: Compare models on metrics

### evaluation/
- `MetricsEvaluator`: Compute 17-field safety suite

### simulation/
- `ScenarioSpec`: Scenario definition
- `CarlaAdapter`: CARLA + synthetic fallback

### deployment/
- `ONNXExporter`: Export to ONNX format

### optimization/
- `LatencyProfiler`: Benchmark inference

### utils/
- `_mean_abs_jerk`: Jerk calculation (shared)
- `set_seed`: Reproducibility
- `get_config_hash`: Config tracking

## Import Dependencies

**Minimal external dependencies**:
- PyTorch (models, training)
- NumPy (numerics)
- YAML (configs)
- ONNX (deployment, optional)
- CARLA (simulation, optional)

**No heavy frameworks**: No TensorFlow, no Ray, no HuggingFace.

## Testing Strategy

**7 test files, 58 tests, all CARLA-free**:

1. `test_imports_and_config.py` (9 tests): Module imports, config correctness
2. `test_models_and_latency.py` (6 tests): CNN inference, latency profiling
3. `test_schemas_metrics_and_mining.py` (8 tests): Data structures, metrics, failure mining
4. `test_episode_store.py` (3 tests): Data persistence
5. `test_simulation_dashboard_training.py` (10 tests): Full integration (scenarios, training, evaluation)
6. `test_training_and_deployment.py` (6 tests): Losses, ONNX export, model cards
7. `test_carla_configs.py` (3 tests): CARLA configs, graceful fallback

**Coverage**: All critical paths tested without requiring CARLA.

## Execution Modes

### Synthetic (default)
- No external dependencies
- Validates end-to-end pipeline
- All 58 tests pass
- Real metrics on synthetic data

### CARLA (optional)
- Requires CARLA simulator running
- Real sensor data
- Same code interface
- Graceful fallback if CARLA unavailable

## Performance Characteristics

### Model Size
- **Parameters**: 191,012
- **Disk**: 0.73 MB
- **Memory**: < 1 MB at inference

### Latency (CPU, single batch)
- **p50**: 1.22 ms
- **p95**: 1.50 ms
- **Mean**: 1.25 ms
- **FPS**: 798

### Data Volume
- Episode storage: Disk-based, scalable
- Replay buffer: In-memory, configurable
- Failure mining: O(N) scan once per batch

## Extension Points

**Easy to extend**:
- New loss functions (training/losses.py)
- New metrics (evaluation/metrics.py)
- New scenarios (simulation/scenarios.py)
- New policy architectures (models/policy.py)
- New training algorithms (training/__init__.py)

**Hard to change without impact**:
- SafetyMetrics schema (17 fields used throughout)
- ControlAction schema (throttle, brake, steering)
- Episode schema (observations, actions, rewards, dones)

---

See individual module files for implementation details.
