"""Student policy for autonomous driving."""

from typing import Optional
import numpy as np
from autonomy_stack.schemas import ControlAction, ObservationFrame


class StudentPolicy:
    """Learned student policy."""

    def __init__(self, model=None, config: dict = None):
        self.model = model
        self.config = config or {}

    def predict(self, observation: ObservationFrame) -> ControlAction:
        """Predict action from observation.

        Args:
            observation: Current observation frame

        Returns:
            Control action
        """
        if self.model is None:
            return ControlAction(throttle=0.3, brake=0.0, steering=0.0)

        action = self.model.forward(observation)
        return action

    def reset(self):
        """Reset policy state."""
        pass

    def save(self, path: str):
        """Save policy to disk."""
        if self.model is not None:
            self.model.save(path)

    def load(self, path: str):
        """Load policy from disk."""
        if self.model is not None:
            self.model.load(path)
