"""Abstract environment class."""

from . import abstract_feature_extractor

import csv
import numpy as np


class Stddev(abstract_feature_extractor.AbstractFeatureExtractor):
    
    def __init__(self,
                 n_channels=7,
                 write_file='./data/stddevs_v0.csv',
                 history=100):
        self._n_channels = n_channels
        self._recent_stddevs = []
        self._writer = csv.writer(open(write_file, 'w'))
        self._history = history
    
    def __call__(self, data):
        """Process batch of data of shape [n_timesteps, n_channels]."""
        
        # Trim unused channels
        data = data[:, :self._n_channels]
        
        # Zero-mean the data
        data -= np.nanmean(data, axis=0, keepdims=True)
        
        # Append stddevs to self._recent_stddevs
        current_stddevs = np.nanstd(data, axis=0)
        self._recent_stddevs.append(current_stddevs)
        if len(self._recent_stddevs) > self._history:
            self._recent_stddevs.pop(0)
            
        # write stddevs to csv
        to_write = [str(x) for x in current_stddevs]
        self._writer.writerow(to_write)
        
        # Compute mean stddev per channel
        mean_stddevs = np.mean(self._recent_stddevs, axis=0)
        
        # Features are normalized stddevs
        features = current_stddevs / mean_stddevs
        
        return features
    
    @property
    def n_features(self):
        return self._n_channels
    