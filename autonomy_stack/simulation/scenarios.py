"""Driving scenario definitions."""

from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class ScenarioSpec:
    """Specification for a driving scenario."""

    name: str
    duration_seconds: int
    num_vehicles: int = 0
    weather: str = "clear"
    start_position: Tuple[float, float, float] = (0, 0, 0)
    target_position: Tuple[float, float, float] = (100, 0, 0)
    traffic_density: float = 0.0
    difficulty: str = "easy"

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'duration_seconds': self.duration_seconds,
            'num_vehicles': self.num_vehicles,
            'weather': self.weather,
            'start_position': self.start_position,
            'target_position': self.target_position,
            'traffic_density': self.traffic_density,
            'difficulty': self.difficulty,
        }


def get_default_scenarios() -> List[ScenarioSpec]:
    """Get default test scenarios."""
    return [
        ScenarioSpec(
            name="simple_straight",
            duration_seconds=30,
            num_vehicles=0,
            weather="clear",
            difficulty="easy",
        ),
        ScenarioSpec(
            name="urban_driving",
            duration_seconds=60,
            num_vehicles=5,
            weather="clear",
            traffic_density=0.5,
            difficulty="medium",
        ),
        ScenarioSpec(
            name="rainy_traffic",
            duration_seconds=60,
            num_vehicles=10,
            weather="rain",
            traffic_density=0.8,
            difficulty="hard",
        ),
    ]
