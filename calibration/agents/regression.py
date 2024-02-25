"""Regression agent."""

from . import abstract_agent
import numpy as np
from sklearn import linear_model as sklearn_linear_model


class Regression(abstract_agent.AbstractAgent):
    
    def __init__(self, name):
        self._name = name
        self.data = {}
        self._prev_trial_index = None
        self._reg = None
    
    def reset(self, trial_index):
        """Fit linear regression."""
        
        if self._prev_trial_index is not None:
            # Convert data from previous trial into numpy array
            self.data[self._prev_trial_index] = {
                'input': np.array(self.data[self._prev_trial_index]['input']),
                'output': np.array(self.data[self._prev_trial_index]['output']),
            }
            
            # Fit linear regression
            data_keys = sorted(self.data.keys())
            data_input = np.concatenate(
                [self.data[k]['input'] for k in data_keys], axis=0)
            data_output = np.concatenate(
                [self.data[k]['output'] for k in data_keys], axis=0)
            reg = sklearn_linear_model.LinearRegression()
            self._reg = reg.fit(data_input, data_output)
        
        # Clean trial_index data for upcoming trial to overwrite
        self.data[trial_index] = {
            'input': [],
            'output': [],
        }
        self._prev_trial_index = trial_index
            
    def _extract_agent_input(self, agent_input):
        if 'mouse' in agent_input:
            return agent_input['mouse']
        else:
            raise ValueError(f'Found no input field in {agent_input.keys()}')
    
    def collect_data(self, agent_input, target_action, trial_index):
        agent_input = self._extract_agent_input(agent_input)
        self.data[trial_index]['input'].append(agent_input)
        self.data[trial_index]['output'].append(target_action)
        
    def action(self, agent_input):
        if self._reg is None:
            return np.array([0., 0])
            
        agent_input = self._extract_agent_input(agent_input)
        agent_action = self._reg.predict(agent_input[None])[0]
        return agent_action
    
    def snapshot(self):
        pass
    