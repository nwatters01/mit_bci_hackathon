"""Duckie controller."""

import time
from duckietown.sdk.robots.duckiebot import DB21J
import calibration
import lsl_api
import numpy as np

SIMULATED_ROBOT_NAME: str = "map_0/vehicle_0"
REAL_ROBOT_NAME: str = "rover"

robot: DB21J = DB21J(SIMULATED_ROBOT_NAME, simulated=True)  # change accordingly

# Parameters
_SMOOTHING_WINDOW_SIZE = 5
_SPEED_INTERCEPT = 0.
_SPEED_GAIN = 0.3
_ROTATION_GAIN = 0.25
_HIGHWAY_GAIN = 0.


def _action_to_wheels(action,
                      speed_intercept=_SPEED_INTERCEPT,
                      speed_gain=_SPEED_GAIN,
                      rotation_gain=_ROTATION_GAIN,
                      highway_gain=_HIGHWAY_GAIN):
    r, theta = action
    rot_gain = rotation_gain * np.exp(-highway_gain * r)
    wheel_left = speed_intercept + speed_gain * (r + rot_gain * theta)
    wheel_right = speed_intercept + speed_gain * (r - rot_gain * theta)
    # print(wheel_left, wheel_right)
    
    return [wheel_left, wheel_right]

    
class DuckieControllerEEG():
    
    def __init__(self, snapshot_name=None):
        self._feature_stream = lsl_api.get_lsl_api()
        self._agent = calibration.Agent(
            in_features=self._feature_stream.n_features,
            out_features=2,
            name='',
            snapshot_name=snapshot_name,
        )
    
    def __call__(self):
        """Get 2-dimensional action for duckie bot."""
        features = self._feature_stream()
        action = self._agent(features)
        wheels = _action_to_wheels(action)
        return wheels


class DuckieControllerSpinner():
    
    def __init__(self, r=0.5, theta=-0.5):
        self._r = r
        self._theta = theta
    
    def __call__(self):
        """Get 2-dimensional action for duckie bot."""
        action = [self._r, self._theta]
        wheels = _action_to_wheels(action)
        return wheels
        

def _smooth(history, new_sample, window_size=_SMOOTHING_WINDOW_SIZE):
    """Smooths sample based on history with half-triangular kernel."""
    # Add new sample and truncate history to window_size length
    history.append(new_sample)
    while len(history) > window_size:
        history.pop(0)

    # Pad history with zeros if necessary
    history = (window_size - len(history)) * [0] + history

    # Create normalized half-triangular kernel
    kernel = np.linspace(0, 1, window_size + 1)[1:]
    kernel /= np.sum(kernel)

    # Apply kernel
    output = np.dot(kernel, np.array(history))

    return output


def main():
    """Move robot."""

    # controller = DuckieControllerEEG(snapshot_name='')
    controller = DuckieControllerSpinner()

    wheel_left_history = []
    wheel_right_history = []

    robot.motors.start()
    stime: float = time.time()
    while time.time() - stime < 2:
        wheel_left, wheel_right = controller()
        smoothed_wheel_left = _smooth(wheel_left_history, wheel_left)
        smoothed_wheel_right = _smooth(wheel_right_history, wheel_right)
        speeds = (smoothed_wheel_left, smoothed_wheel_right)
        robot.motors.publish(speeds)

        time.sleep(0.25)
    robot.motors.stop()
    print("Stopped.")


if __name__ == '__main__':
    main()
