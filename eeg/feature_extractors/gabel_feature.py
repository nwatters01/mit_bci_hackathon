"""Abstract environment class."""

import numpy as np
from . import abstract_feature_extractor


class Gabel(abstract_feature_extractor.AbstractFeatureExtractor):
    
    def __init__(self, n_features=10, threshold=0.5):
        self._n_features = n_features
        self._threshold = threshold
    
    def __call__(self, data):
        # Data has shape [n_timesteps, n_channels]
        filtered_data = np.copy(data)
        filtered_data[data < self._threshold] = np.nan
        
        return filtered_data
    
    @property
    def n_features(self):
        return self._n_features
    