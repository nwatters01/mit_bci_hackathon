"""Calibration 0 environment."""

from . import abstract_environment
import collections
import enum
import numpy as np
import sprite


class Phase(enum.IntEnum):
    READY = 0
    SET = 1
    GO = 2
    ITI = 3
    
    
class Action(enum.IntEnum):
    LEFT = 0
    RIGHT = 1


def _get_arrow(curvature,
               length,
               linewidth=0.005,
               step_size=0.005):
    num_steps = int(np.abs(length) // step_size)
    step_size = length / num_steps
    
    total_theta = curvature * length
    delta_thetas = num_steps * [total_theta / num_steps]
    thetas = np.cumsum(delta_thetas)
    step_vectors_x = np.sin(thetas)
    step_vectors_y = np.cos(thetas)
    
    step_vectors = np.stack([step_vectors_x, step_vectors_y], axis=1)
    step_vectors_left = np.stack([step_vectors_y, -1 * step_vectors_x], axis=1)
    step_vectors_right = np.stack([-1 * step_vectors_y, step_vectors_x], axis=1)
    
    centerpoints = step_size * np.cumsum(step_vectors, axis=0)
    left_points = centerpoints + linewidth * step_vectors_left
    right_points = centerpoints + linewidth * step_vectors_right
    
    arrow_shape = np.concatenate([left_points, right_points[::-1]], axis=0)
    arrow_endpoint = centerpoints[-1]
    
    return arrow_shape, arrow_endpoint


class CalibrationBase(abstract_environment.AbstractEnvironment):
    
    def __init__(self,
                 renderer,
                 agent,
                 ready_steps=20,
                 set_steps=20,
                 go_steps=20,
                 render_agent_action=True,
                 target_curvatures=(-5., 0., 5.),
                 target_lengths=(0.3, -0.3),
                 origin=(0.5, 0.5)):
        """Constructor."""
        
        # Create targets.
        self._target_curvatures_lengths = []
        self._target_shapes = []
        self._target_endpoints = []
        for l in target_lengths:
            for c in target_curvatures:
                self._target_curvatures_lengths.append((c, l))
                target_shape, target_endpoint = _get_arrow(c, l)
                self._target_shapes.append(target_shape)
                self._target_endpoints.append(target_endpoint)
        
        # Initialize AbstractEnvironment
        num_trials = len(self._target_shapes)
        super(CalibrationBase, self).__init__(
            renderer=renderer,
            agent=agent,
            num_trials=num_trials,
        )
        
        self._ready_steps = ready_steps
        self._set_steps = set_steps
        self._go_steps = go_steps
        self._render_agent_action = render_agent_action
        self._origin = np.array(origin)
        
    def initial_state(self):
        # Make duckie
        duckie_shape = 0.02 * np.array([
            [1, 1], [1, -1], [-1, -1], [-1, 1], [0, 2],
        ])
        duckie = sprite.Sprite(
            x=self._origin[0], y=self._origin[1], shape=duckie_shape,
            c0=255, c1=255, c2=255,
        )
        
        # Make target arrow
        target_shape = self._target_shapes[self._trial_index]
        target_endpoint = self._target_endpoints[self._trial_index]
        target = sprite.Sprite(
            x=self._origin[0], y=self._origin[1], shape=target_shape,
            c0=128, c1=128, c2=128, opacity=0,
        )
        target_end = sprite.Sprite(
            x=self._origin[0] + target_endpoint[0],
            y=self._origin[1] + target_endpoint[1],
            shape='circle', scale=0.03,
            c0=128, c1=128, c2=128, opacity=0,
        )
        
        # Make agent
        if self._render_agent_action:
            agent_action_opacity = 255
        else:
            agent_action_opacity = 0
        agent_action = sprite.Sprite(
            x=self._origin[0], y=self._origin[1],
            shape='circle', scale=0.02, c0=0, c1=255, c2=0,
            opacity=agent_action_opacity,
        )
        
        state = collections.OrderedDict([
            ('target', [target, target_end]),
            ('duckie', [duckie]),
            ('agent_action', [agent_action])
        ])
        
        return state
    
    def target_action(self):
        return self._target_endpoints[self._trial_index]
    
    def _step(self, agent_input, agent_action):
        # Handle phase transitions
        target = self.state['target']
        if self.phase == Phase.ITI:
            for s in target:
                s.opacity = 0
            if agent_input['keyboard'] == self._prev_trial_action:
                self.reset(next_trial=False)
            elif agent_input['keyboard'] == self._next_trial_action:
                self.reset(next_trial=True)
        elif self.phase == Phase.SET:
            for s in target:
                s.opacity = 255
        elif self.phase == Phase.GO:
            for s in target:
                s.c0 = 255
                s.c1 = 0
                s.c2 = 0
                
        # Move agent
        agent = self.state['agent_action'][0]
        agent.position = self._origin + agent_action
    
    @property
    def should_collect_data(self):
        return self.phase == Phase.GO
    
    @property
    def phase(self):
        ready_steps = self._ready_steps
        set_steps = self._set_steps
        go_steps = self._go_steps
        if self._step_count < ready_steps:
            return Phase.READY
        elif self._step_count < ready_steps + set_steps:
            return Phase.SET
        elif self._step_count < ready_steps + set_steps + go_steps:
            return Phase.GO
        else:
            return Phase.ITI
        
    @property
    def title(self):
        title = (
            f'Trial {self._trial_index} / {self._num_trials}; '
            f'phase {self.phase.name}'
        )
        return title
    

class CalBlind(CalibrationBase):
    
    def __init__(self, renderer, agent):
        super(CalBlind, self).__init__(
            renderer=renderer,
            agent=agent,
            render_agent_action=False,
        )
        
        
class CalFeedback(CalibrationBase):
    
    def __init__(self, renderer, agent):
        super(CalFeedback, self).__init__(
            renderer=renderer,
            agent=agent,
            render_agent_action=True,
        )