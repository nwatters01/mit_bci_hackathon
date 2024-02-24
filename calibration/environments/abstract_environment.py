"""Abstract environment class."""

import abc
import enum

    
class Action(enum.IntEnum):
    LEFT = 0
    RIGHT = 1
    DOWN = 2
    UP = 3
    

class AbstractEnvironment(abc.ABC):
    
    def __init__(self,
                 renderer,
                 agent,
                 num_trials,
                 prev_trial_action=Action.LEFT,
                 next_trial_action=Action.RIGHT):
        self._renderer = renderer
        self._agent = agent
        self._num_trials = num_trials
        self._prev_trial_action = prev_trial_action
        self._next_trial_action = next_trial_action
        
        self._trial_index = -1
        self._step_count = 0
        
    @abc.abstractmethod
    def initial_state(self):
        raise NotImplementedError
    
    @abc.abstractmethod
    def target_action(self):
        raise NotImplementedError
    
    @abc.abstractmethod
    def _step(self, action, agent_action):
        raise NotImplementedError
    
    @abc.abstractproperty
    def should_collect_data(self):
        raise NotImplementedError
    
    @abc.abstractproperty
    def title(self):
        raise NotImplementedError
    
    def reset(self, next_trial=True):
        self._step_count = 0
        if next_trial:
            self._trial_index += 1
        self._trial_index = self._trial_index % self._num_trials
        self.state = self.initial_state()
        self._agent.reset(self._trial_index)
        return self.observation()
    
    def step(self, agent_input):
        # print(agent_input, self.target_action())
        
        if self.should_collect_data:
            agent_action = self._agent.collect_data(
                agent_input,
                self.target_action(),
                trial_index=self._trial_index,
            )
        agent_action = self._agent.action(agent_input)
        
        self._step(agent_input, agent_action)
        self._step_count += 1
        return self.observation()
    
    def observation(self):
        return self._renderer(self.state)
    
    @property
    def agent(self):
        return self._agent
    
