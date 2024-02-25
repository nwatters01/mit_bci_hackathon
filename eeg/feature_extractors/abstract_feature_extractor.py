"""Abstract environment class."""

import abc


class AbstractFeatureExtractor(abc.ABC):
    
    @abc.abstractmethod
    def __call__(self, data):
        raise NotImplementedError
    
    @abc.abstractproperty
    def n_features(self):
        raise NotImplementedError
    