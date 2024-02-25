"""Duckie controller."""

import calibration
import lsl_api


def _action_to_wheels(action,
                      speed_intercept=0,
                      speed_gain=1.,
                      rotation_gain=1.,
                      auto_steer_gain=0.):
    r, theta = action
    rot_coef = 1 + auto_steer_gain * (r - speed_intercept)
    rot = rotation_gain * rot_coef * theta
    wheel_left = speed_intercept + speed_gain * (r - rot)
    wheel_right = speed_intercept + speed_gain * (r + rot)
    
    return [wheel_left, wheel_right]

    
class DuckieController():
    
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
        