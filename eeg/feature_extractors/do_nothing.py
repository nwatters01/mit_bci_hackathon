"""Abstract environment class."""

from . import abstract_feature_extractor


class DoNothing(abstract_feature_extractor.AbstractFeatureExtractor):
    
    def __init__(self, n_features=10):
        self._n_features = n_features
    
    def __call__(self, data):
        # Data has shape [n_timesteps, n_channels]
        return data
    
    @property
    def n_features(self):
        return self._n_features
    