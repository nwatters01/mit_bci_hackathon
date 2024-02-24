"""Dummy agent that outputs zero actions."""

from . import abstract_agent
import numpy as np


class Dummy(abstract_agent.AbstractAgent):
    
    def reset(self, *args, **kwargs):
        del args
        del kwargs
        pass
    
    def collect_data(self, *args, **kwargs):
        del args
        del kwargs
        pass
        
    def action(self, *args, **kwargs):
        del args
        del kwargs
        return np.array([0., 0])
    
    def snapshot(self):
        pass
    