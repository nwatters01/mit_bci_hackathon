"""Abstract environment class."""

from . import abstract_feature_extractor

import csv


class DataSaver(abstract_feature_extractor.AbstractFeatureExtractor):
    
    def __init__(self, n_channels=10, filename='v0'):
        self._n_channels = n_channels
        data_path = f'./data/{filename}.csv'
        self._writer = csv.writer(open(data_path, 'w'))
        self._key = ''
    
    def __call__(self, data):
        # Data has shape [n_timesteps, n_channels]
        n_timesteps, n_channels = data.shape
        for t in range(n_timesteps):
            to_write = [str(x) for x in data[t]]
            to_write = [self._key] + to_write
            self._writer.writerow(to_write)
        
        return data
    
    def key_press(self, key):
        self._key = key
        
    def key_release(self):
        self._key = ''
    
    @property
    def n_features(self):
        return self._n_channels
    