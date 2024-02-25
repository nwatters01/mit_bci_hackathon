"""Wiggles environment."""

from . import abstract_environment
# import abstract_environment
import collections
import enum
import math
import numpy as np

import sys
sys.path.append('..')
import sprite


class Phase(enum.IntEnum):
    READY = 0
    SET = 1
    GO = 2
    ITI = 3
    
    
class Action(enum.IntEnum):
    LEFT = 0
    RIGHT = 1


def sample_wiggle(length=50,
                  curvature_range=(0.1, 0.2),
                  max_turn=1. * np.pi,
                  max_length=15,
                  origin=(0.5, 0.05)):
    """Sample wiggle."""
    
    # Sample thetas
    thetas = [np.random.uniform(-0.01, 0.01)]
    while len(thetas) < length:
        # Sample curvature
        curvature = np.random.uniform(curvature_range[0], curvature_range[1])
        if thetas[-1] > 0:
            curvature *= -1
        
        # Compute thetas
        max_turn_length = math.floor(np.abs(max_turn / curvature))
        max_turn_length = min(max_turn_length, max_length)
        turn_length = np.random.randint(1, max_turn_length)
        for _ in range(turn_length):
            thetas.append(thetas[-1] + curvature)
            
    # Convert thetas to deltas
    deltas = np.stack([np.sin(thetas), np.cos(thetas)], axis=1)
    
    # Convert deltas to path
    path = np.cumsum(deltas, axis=0)
    
    # Normalize path
    min_path_y = np.min(path[:, 1])
    max_path_y = np.max(path[:, 1])
    path /= (max_path_y - min_path_y)
    scale = 1 - (2 * origin[1])
    path *= scale
    path += np.array(origin) - path[0]
    
    # Reject if path exits bounding box
    if np.any(path < 0) or np.any(path > 1):
        return sample_wiggle(
            length=length,
            curvature_range=curvature_range,
            max_turn=max_turn,
            max_length=max_length,
            origin=origin,
        )
    
    return path


def sample_track(length=200,
                  curvature_range=(0.05, 0.3),
                  max_turn=1. * np.pi,
                  max_length=12,
                  border=0.05,
                  r_scale=0.25,
                  theta_scale=1.8 * np.pi):
    """Sample wiggle."""
    
    # Sample wiggle path
    wiggle_path = sample_wiggle(
        length=length,
        curvature_range=curvature_range,
        max_turn=max_turn,
        max_length=max_length,
        origin=(0., 0.),
    )
    
    # Rotate path to be horizontal
    wiggle_path_mean = wiggle_path[-1] - wiggle_path[0]
    theta = np.arctan2(*wiggle_path_mean)
    rotation_matrix = np.array([
        [np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)],
    ])
    path = np.matmul(wiggle_path, rotation_matrix)
    
    # Normalize path
    path_y_min = path[0][1]
    path_y_max = path[-1][1]
    path[:, 1] -= path_y_min
    path /= (path_y_max - path_y_min)
    
    # Roll path into a circle by transforming to polar coordinates
    path_r = r_scale + path[:, 0]
    path_theta = theta_scale * path[:, 1]
    path = (
        path_r[:, None] *
        np.stack([np.sin(path_theta), np.cos(path_theta)], axis=1)
    )
    
    # Apply arbitrary rotation
    theta = np.random.uniform(0, 2 * np.pi)
    rotation_matrix = np.array([
        [np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)],
    ])
    path = np.matmul(path, rotation_matrix)  
    
    # Normalize path
    path[:, 0] -= np.min(path[:, 0])
    path[:, 1] -= np.min(path[:, 1])
    path /= np.max(path)
    path = border + (1 - 2 * border) * path
    
    return path
    

class NavBase(abstract_environment.AbstractEnvironment):
    
    def __init__(self,
                 renderer,
                 agent,
                 path_sampler,
                 num_trials=10,
                 reward_freq_threshold=5,
                 target_action_scale=0.1,
                 backwards_prob=0.5,
                 ready_steps=30,
                 set_steps=30,
                 timeout_steps=600,
                 render_agent_action=True,
                 origin=(0.5, 0.05)):
        """Constructor."""
        
        # Initialize AbstractEnvironment
        super(NavBase, self).__init__(
            renderer=renderer,
            agent=agent,
            num_trials=num_trials,
        )
        
        self._reward_freq_threshold = reward_freq_threshold
        self._target_action_scale = target_action_scale
        self._backwards_prob = backwards_prob
        self._ready_steps = ready_steps
        self._set_steps = set_steps
        self._timeout_steps = timeout_steps
        self._render_agent_action = render_agent_action
        self._origin = np.array(origin)
        self._wiggle_paths = [path_sampler() for _ in range(self._num_trials)]
        
    def initial_state(self):
        # Make duckie
        duckie_shape = 0.02 * np.array([
            [1, 1], [1, -1], [-1, -1], [-1, 1], [0, 2],
        ])
        
        # Make targets
        if np.random.rand() < self._backwards_prob:
            targets = [
                sprite.Sprite(
                    x=p[0], y=1. - p[1], shape='square', scale=0.01,
                    c0=128, c1=128, c2=128, opacity=0,
                )
                for p in self._wiggle_paths[self._trial_index]
            ]
        else:
            targets = [
                sprite.Sprite(
                    x=p[0], y=p[1], shape='square', scale=0.01,
                    c0=128, c1=128, c2=128, opacity=0,
                )
                for p in self._wiggle_paths[self._trial_index]
            ]
            
        # Make agent
        duckie = sprite.Sprite(
            x=targets[0].x, y=targets[0].y, shape=duckie_shape,
            c0=255, c1=255, c2=255,
        )
        
        state = collections.OrderedDict([
            ('targets', targets),
            ('duckie', [duckie]),
        ])
        
        # Initialize reward steps
        self._trial_complete = False
        self._go_step_index = 0
        self._reward_steps = []
        
        return state
    
    def target_action(self):
        if len(self.state['targets']) < 2:
            return None
        pos_0 = self.state['targets'][0].position
        pos_1 = self.state['targets'][1].position
        target_action = self._target_action_scale * (pos_1 - pos_0)
        return target_action
    
    def _step(self, agent_input, agent_action):
        # Handle phase transitions
        target = self.state['targets']
        phase = self.phase
        if phase == Phase.ITI:
            for s in target:
                s.opacity = 0
            if agent_input['keyboard'] == self._prev_trial_action:
                self.reset(next_trial=False)
            elif agent_input['keyboard'] == self._next_trial_action:
                self.reset(next_trial=True)
        elif phase == Phase.SET:
            for s in target:
                s.opacity = 255
        elif phase == Phase.GO:
            for s in target:
                s.c0 = 255
                s.c1 = 0
                s.c2 = 0
                
        if phase == Phase.SET:
            # Orient duckie
            duckie = self.state['duckie'][0]
            motion = agent_action
            angle = np.arctan2(-1 * motion[0], motion[1])
            duckie.angle = angle
        elif phase == Phase.GO:
            self._go_step_index += 1
            
            # Move duckie
            duckie = self.state['duckie'][0]
            motion = agent_action
            angle = np.arctan2(-1 * motion[0], motion[1])
            duckie.position = duckie.position + motion
            duckie.angle = angle
            
            # Remove acquired targets
            if len(self.state['targets']) == 0:
                self._trial_complete = True
                return
            next_target = self.state['targets'][0]
            while (
                    next_target is not None and
                    duckie.overlaps_sprite(next_target)
                ):
                self.state['targets'] = self.state['targets'][1:]
                if len(self.state['targets']) > 0:
                    next_target = self.state['targets'][0]
                else:
                    next_target = None
                self._reward_steps.append(self._go_step_index)
            
            # TODO: Consider rendering agent action for more visual feedback
    
    @property
    def should_collect_data(self):
        # Should not collect data if not in GO phase
        if self.phase != Phase.GO:
            return False
        
        # Should not collect data if there is no target action
        if self.target_action() is None:
            return False
        
        # TODO: Should not collect data if performance is very poor
        if len(self._reward_steps) == 0:
            return False
        since_last_step = self._go_step_index - self._reward_steps[-1]
        if since_last_step > self._reward_freq_threshold:
            return False
        
        return True
    
    @property
    def phase(self):
        ready_steps = self._ready_steps
        set_steps = self._set_steps
        timeout_steps = self._timeout_steps
        if self._trial_complete:
            return Phase.ITI
        if self._step_count < ready_steps:
            return Phase.READY
        elif self._step_count < ready_steps + set_steps:
            return Phase.SET
        elif self._step_count < ready_steps + set_steps + timeout_steps:
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
    
    
class Wiggles(NavBase):
    
    def __init__(self,
                 renderer,
                 agent,
                 num_trials=10,
                 reward_freq_threshold=5,
                 backwards_prob=0.5,
                 ready_steps=20,
                 set_steps=20,
                 timeout_steps=200,
                 render_agent_action=True,
                 origin=(0.5, 0.05)):
        super(Wiggles, self).__init__(
            renderer=renderer,
            agent=agent,
            path_sampler=sample_wiggle,
            num_trials=num_trials,
            reward_freq_threshold=reward_freq_threshold,
            backwards_prob=backwards_prob,
            ready_steps=ready_steps,
            set_steps=set_steps,
            timeout_steps=timeout_steps,
            render_agent_action=render_agent_action,
            origin=origin,
        )
        
        
class Tracks(NavBase):
    
    def __init__(self,
                 renderer,
                 agent,
                 num_trials=10,
                 reward_freq_threshold=5,
                 backwards_prob=0.5,
                 ready_steps=20,
                 set_steps=20,
                 timeout_steps=200,
                 render_agent_action=True,
                 origin=(0.5, 0.05)):
        super(Tracks, self).__init__(
            renderer=renderer,
            agent=agent,
            path_sampler=sample_track,
            num_trials=num_trials,
            reward_freq_threshold=reward_freq_threshold,
            backwards_prob=backwards_prob,
            ready_steps=ready_steps,
            set_steps=set_steps,
            timeout_steps=timeout_steps,
            render_agent_action=render_agent_action,
            origin=origin,
        )