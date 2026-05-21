"""CARLA simulator adapter with graceful fallback to synthetic mode."""

from typing import Optional, Tuple
import numpy as np
from autonomy_stack.schemas import ObservationFrame, ControlAction
from autonomy_stack.simulation.scenarios import ScenarioSpec

try:
    import carla
    CARLA_AVAILABLE = True
except ImportError:
    CARLA_AVAILABLE = False


class CarlaAdapter:
    """Adapter for CARLA simulator with fallback to synthetic mode."""

    def __init__(self, scenario: ScenarioSpec, use_carla: bool = True):
        self.scenario = scenario
        self.use_carla = use_carla and CARLA_AVAILABLE
        self.client = None
        self.world = None
        self.vehicle = None
        self._init_simulation()

    def _init_simulation(self):
        """Initialize CARLA or fallback to synthetic."""
        if self.use_carla:
            try:
                self.client = carla.Client('localhost', 2000)
                self.client.set_timeout(10.0)
                self.world = self.client.get_world()
            except Exception as e:
                print(f"CARLA initialization failed: {e}")
                print("Falling back to synthetic mode")
                self.use_carla = False

    def step(self, action: ControlAction) -> Tuple[ObservationFrame, float, bool]:
        """Execute action and return observation.

        Args:
            action: Control action

        Returns:
            Tuple of (observation, reward, done)
        """
        if self.use_carla:
            return self._step_carla(action)
        else:
            return self._step_synthetic(action)

    def _step_carla(self, action: ControlAction) -> Tuple[ObservationFrame, float, bool]:
        """Step in CARLA simulator."""
        # Would interact with real CARLA here
        control = carla.VehicleControl()
        control.throttle = action.throttle
        control.brake = action.brake
        control.steer = action.steering
        self.vehicle.apply_control(control)
        self.world.tick()

        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        obs = ObservationFrame(rgb_image=rgb)
        reward = 1.0
        done = False

        return obs, reward, done

    def _step_synthetic(self, action: ControlAction) -> Tuple[ObservationFrame, float, bool]:
        """Step in synthetic mode (no CARLA required)."""
        rgb = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        obs = ObservationFrame(rgb_image=rgb)
        reward = 1.0
        done = False
        return obs, reward, done

    def reset(self) -> ObservationFrame:
        """Reset simulation."""
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        return ObservationFrame(rgb_image=rgb)

    def close(self):
        """Clean up simulator."""
        if self.client is not None:
            self.client = None
