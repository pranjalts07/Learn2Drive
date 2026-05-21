"""Data schemas for autonomous driving system."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class ControlAction:
    """Vehicle control action."""
    throttle: float
    brake: float
    steering: float

    def __post_init__(self):
        self.throttle = max(0, min(1, self.throttle))
        self.brake = max(0, min(1, self.brake))
        self.steering = max(-1, min(1, self.steering))


@dataclass
class ObservationFrame:
    """Single observation frame."""
    rgb_image: np.ndarray
    lidar_data: Optional[np.ndarray] = None
    speed: float = 0.0
    position: Tuple[float, float, float] = (0, 0, 0)
    rotation: Tuple[float, float, float] = (0, 0, 0)


@dataclass
class Episode:
    """Trajectory episode."""
    observations: List[ObservationFrame]
    actions: List[ControlAction]
    rewards: List[float]
    dones: List[bool]
    metadata: dict = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.observations)


@dataclass
class SafetyMetrics:
    """Safety evaluation metrics."""
    collision_count: int = 0
    violation_count: int = 0
    success_rate: float = 0.0
    mean_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    mean_fps: float = 0.0
    mean_jerk: float = 0.0
    max_acceleration: float = 0.0
    mean_speed: float = 0.0
    route_completion: float = 0.0
    comfort_score: float = 0.0
    offroad_incidents: int = 0
    traffic_violations: int = 0
    near_miss_count: int = 0
    inference_time_ms: float = 0.0
    model_size_mb: float = 0.0

    def to_dict(self) -> dict:
        return {
            'collision_count': self.collision_count,
            'violation_count': self.violation_count,
            'success_rate': self.success_rate,
            'mean_latency_ms': self.mean_latency_ms,
            'p50_latency_ms': self.p50_latency_ms,
            'p95_latency_ms': self.p95_latency_ms,
            'mean_fps': self.mean_fps,
            'mean_jerk': self.mean_jerk,
            'max_acceleration': self.max_acceleration,
            'mean_speed': self.mean_speed,
            'route_completion': self.route_completion,
            'comfort_score': self.comfort_score,
            'offroad_incidents': self.offroad_incidents,
            'traffic_violations': self.traffic_violations,
            'near_miss_count': self.near_miss_count,
            'inference_time_ms': self.inference_time_ms,
            'model_size_mb': self.model_size_mb,
        }
