"""Expert policy coach for data collection and evaluation."""

from typing import Tuple
import numpy as np
from autonomy_stack.schemas import ControlAction, ObservationFrame


class ExpertPolicy:
    """Expert policy that provides demonstrations."""

    def __init__(self, config: dict):
        self.config = config
        self.max_throttle = config.get('max_throttle', 0.8)
        self.max_brake = config.get('max_brake', 0.8)

    def predict(self, observation: ObservationFrame) -> ControlAction:
        """Predict action from observation.

        Args:
            observation: Current observation frame

        Returns:
            Control action (throttle, brake, steering)
        """
        throttle = min(0.5, self.max_throttle)
        brake = 0.0
        steering = 0.0

        action = ControlAction(
            throttle=throttle,
            brake=brake,
            steering=steering
        )
        return action

    def reset(self):
        """Reset policy state."""
        pass
