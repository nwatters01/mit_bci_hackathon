"""Duckie controller."""

import time
from duckietown.sdk.robots.duckiebot import DB21J
import calibration
import lsl_api
import numpy as np
import sys

SIMULATED_ROBOT_NAME: str = "map_0/vehicle_0"
# REAL_ROBOT_NAME: str = "curiosity"
REAL_ROBOT_NAME: str = "perseverance"
# REAL_ROBOT_NAME: str = "rover"

# robot: DB21J = DB21J(SIMULATED_ROBOT_NAME, simulated=True)
robot: DB21J = DB21J(REAL_ROBOT_NAME, simulated=False)

_DURATION = 60


# # Nick cal_nw_2_v2 real
# _BOUCEBACK = 0.2
# _BOUCEBACK_DELAY = 20
# _SMOOTHING_WINDOW_R = 10
# _SMOOTHING_WINDOW_THETA = 10
# # _ROTATION_INTERCEPT = -0.17  # Good for simulation
# _ROTATION_INTERCEPT = 0.05  # Good for curiosity
# _ROTATION_GAIN = 0.28
# _HIGHWAY_GAIN = 0.
# _STRAIGHT_SPEED_GAIN = 0.6
# _FIXED_SPEED = 0.8


# Nick cal_nw_2_v2 simulation
_BOUCEBACK = 0.2
_BOUCEBACK_DELAY = 25
_SMOOTHING_WINDOW_R = 10
_SMOOTHING_WINDOW_THETA = 7
_ROTATION_INTERCEPT = -0.15  # Good for simulation
# _ROTATION_INTERCEPT = 0.05  # Good for curiosity
# _ROTATION_GAIN = 0.28
_ROTATION_GAIN = 0.24
_HIGHWAY_GAIN = 0.
_STRAIGHT_SPEED_GAIN = 0.75
_FIXED_SPEED = 0.73


def _get_kernel(window_size, bounceback=0, bounceback_delay=None):
    # Create normalized half-triangular kernel
    kernel = np.linspace(0, 1, window_size + 1)[1:]

    if bounceback_delay is not None:
        kernel_bounceback = -1 * bounceback * np.linspace(0, 1, window_size + 1)[1:]
        kernel = np.concatenate([kernel_bounceback[::-1], np.zeros(bounceback_delay), kernel])

    kernel /= np.sum(kernel)
    return kernel

_kernel_r = _get_kernel(_SMOOTHING_WINDOW_R)
_kernel_theta = _get_kernel(_SMOOTHING_WINDOW_THETA, _BOUCEBACK, _BOUCEBACK_DELAY)

_all_corrected_r_theta = []
_all_wheels = []

_corrected_r_history = []
_corrected_theta_history = []


def _smooth(history, new_sample, kernel):
    """Smooths sample based on history with half-triangular kernel."""
    # Add new sample and truncate history to window_size length
    history.append(new_sample)
    while len(history) > len(kernel):
        history.pop(0)

    # Pad history with zeros if necessary
    history = (len(kernel) - len(history)) * [0] + history

    # Apply kernel
    output = np.dot(kernel, np.array(history))

    return output


def _action_to_wheels(action,
                      rotation_intercept=_ROTATION_INTERCEPT,
                      rotation_gain=_ROTATION_GAIN,
                      highway_gain=_HIGHWAY_GAIN,
                      straight_speed_gain=_STRAIGHT_SPEED_GAIN):
    theta, r = action
    r = _FIXED_SPEED
    theta -= rotation_intercept
    if theta > 0:
        theta *= 1.
    elif theta < 0:
        theta *= 1.1

    # r = _smooth(_corrected_r_history, r, _kernel_r)
    theta = _smooth(_corrected_theta_history, theta, _kernel_theta)

    print(theta)

    # r *= np.exp(-straight_speed_gain * np.abs(theta))
    r *= (1 - straight_speed_gain * np.abs(theta))
    _all_corrected_r_theta.append((r, theta))
    rot_gain = rotation_gain * np.exp(-highway_gain * r)
    wheel_left = r + rot_gain * theta
    wheel_right = r - rot_gain * theta
    _all_wheels.append((wheel_left, wheel_right))
    
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
        features, _ = self._feature_stream()
        action = self._agent(features)
        # print(f'action = {action}')
        wheels = _action_to_wheels(action)
        # print(f'wheel = {wheels}')
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


def main(total_time=_DURATION):
    """Move robot."""

    controller = DuckieControllerEEG(snapshot_name='cal_nw_1_v2')
    # controller = DuckieControllerSpinner() 

    wheel_left_history = []
    wheel_right_history = []

    input('Start?')

    robot.motors.start()
    stime: float = time.time()
    sample_freqs = [0.]
    times = [0.]
    while time.time() - stime < total_time:
        wheel_left, wheel_right = controller()
        speeds = (wheel_left, wheel_right)
        robot.motors.publish(speeds)

        w = 20
        mean_corr = np.mean(_all_corrected_r_theta[-w:], axis=0)
        mean_wheel = np.mean(_all_wheels[-w:], axis=0)
        print(f'mean_corr = {mean_corr}')
        # print(f'mean_wheel = {mean_wheel}')

        time.sleep(0.001)
        times.append(time.time())
        times = times[-50:]
        times_array = np.array(times)
        framerate = np.mean(times_array[1:] - times_array[:-1])
        # print(f'framerate = {framerate}')
    robot.motors.publish((0, 0))
    robot.motors.stop()
    print("Stopped.")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\ninterrupted\n')
        robot.motors.stop()
        sys.exit()
