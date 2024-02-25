"""Duckie controller."""

import calibration
import lsl_api

    
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
        return action
        