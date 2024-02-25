"""Abstract agent class."""

import abc
import json
from pathlib import Path
import shutil
import torch

_SNAPSHOT_DIR = Path('./snapshots/')
    

class AbstractAgent(abc.ABC):
    
    @abc.abstractmethod
    def reset(self, trial_index):
        raise NotImplementedError
    
    @abc.abstractmethod
    def collect_data(self, action, target_action, trial_index):
        raise NotImplementedError
    
    @abc.abstractmethod
    def action(self, action):
        raise NotImplementedError
    
    @abc.abstractmethod
    def snapshot(self):
        raise NotImplementedError
        
    @property
    def snapshot_dir(self):
        return self._snapshot_dir
        

class AbstractAgentTorch(torch.nn.Module, metaclass=abc.ABCMeta):
    
    def __init__(self, name, **kwargs):
        super(AbstractAgentTorch, self).__init__()
        self._name = name
        self._snapshot_dir = _SNAPSHOT_DIR / name
        self._kwargs = kwargs
        
        # Clear and create snapshot directory
        if self.snapshot_dir.exists():
            shutil.rmtree(self.snapshot_dir)
        self.snapshot_dir.mkdir(parents=True)
        print(f'\nsnapshot_dir = {self.snapshot_dir}\n')
        kwargs_path = self.snapshot_dir / 'kwargs.json'
        json.dump(self._kwargs, open(kwargs_path, 'w'))
        class_path = self.snapshot_dir / 'class.json'
        json.dump(type(self).__name__, open(class_path, 'w'))
    
    @abc.abstractmethod
    def reset(self, trial_index):
        raise NotImplementedError
    
    @abc.abstractmethod
    def collect_data(self, action, target_action, trial_index):
        raise NotImplementedError
    
    @abc.abstractmethod
    def action(self, action):
        raise NotImplementedError
    
    def snapshot(self):
        snapshot_filepath = self.snapshot_dir / 'state_dict'
        torch.save(self.state_dict(), snapshot_filepath)
        print(f'Saved snapshot to {snapshot_filepath}')
        
    @property
    def snapshot_dir(self):
        return self._snapshot_dir
        
