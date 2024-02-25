"""Calibration.

Usage examples:
$ python3 calibration.py name=test_v0 stream=noise render_agent=True
or
$ python3 calibration.py name=test_v0 stream=lsl_api render_agent=True test=True
"""

import numpy as np
import gui as gui_lib
import lsl_api

import torch
from pathlib import Path
import sys

_SNAPSHOT_DIR = Path('./snapshots')


class MLP(torch.nn.Module):
    """MLP model."""
    
    def __init__(self, in_features, layer_features, activation=None):
        """Constructor."""
        super(MLP, self).__init__()

        self._in_features = in_features
        self._layer_features = layer_features
        if activation is None:
            activation = torch.nn.Sigmoid()
        self.activation = activation

        features_list = [in_features] + list(layer_features)
        module_list = []
        for i in range(len(features_list) - 1):
            if i > 0:
                module_list.append(activation)
            layer = torch.nn.Linear(
                in_features=features_list[i],
                out_features=features_list[i + 1]
            )
            module_list.append(layer)
        
        self.net = torch.nn.Sequential(*module_list)

    def forward(self, x):
        return self.net(x)

    @property
    def in_features(self):
        return self._in_features

    @property
    def layer_features(self):
        return self._layer_features
        
    @property
    def out_features(self):
        return self._layer_features[-1]
    
    
class Agent(torch.nn.Module):
    
    def __init__(self, in_features, out_features, name, snapshot_name=None):
        super(Agent, self).__init__()
        self._name = name
        self._snapshot_path = _SNAPSHOT_DIR / name
        
        # if self._snapshot_path.exists():
        #     raise ValueError(
        #         f'Cannot override path {self._snapshot_path}'
        #     )
        
        self._net = MLP(
            in_features=in_features,
            layer_features=(256, 256, out_features),
        )
        if snapshot_name is not None:
            # Load agent from snapshot
            state_dict_path = _SNAPSHOT_DIR / snapshot_name
            self._net.load_state_dict(torch.load(state_dict_path))
            print(f'Loaded from snapshot {state_dict_path}')
            
    def __call__(self, features, as_numpy=True):
        if len(features.shape) == 1:
            no_batch = True
            features = features[None]
        else:
            no_batch = False
        action = self._net(torch.from_numpy(features.astype(np.float32)))
        if no_batch:
            action = action[0]
        if as_numpy:
            action = action.detach().numpy()
        return action
    
    def snapshot(self):
        torch.save(self._net.state_dict(), self._snapshot_path)
        print(f'Saved snapshot to {self._snapshot_path}')


def _sample_batch(*arrays, batch_size):
    num_samples = arrays[0].shape[0]
    indices = np.random.choice(num_samples, size=batch_size)
    batch_arrays = [x[indices] for x in arrays]
    return batch_arrays
    
    
class Calibrator():
    
    def __init__(self,
                 name,
                 feature_stream,
                 gui,
                 snapshot_name=None,
                 render_agent=False,
                 batch_size=64,
                 training_steps=100,
                 optimizer=torch.optim.SGD,
                 lr=0.001,
                 grad_clip=1):
        self._feature_stream = feature_stream
        self._gui = gui
        self._name = name
        self._render_agent = render_agent
        
        self._agent = Agent(
            in_features=feature_stream.n_features,
            out_features=gui.n_features,
            name=name,
            snapshot_name=snapshot_name,
        )
        
        # Optimization
        self._batch_size = batch_size
        self._training_steps = training_steps
        self._optimizer = optimizer(self._agent.parameters(), lr=lr)
        self._grad_clip = grad_clip
        
    
    def __call__(self, test=False):
        """Run data collection and training for a trial."""
        
        # Run data collection
        all_inputs = []
        all_targets = []
        def _callback(target, fin):
            if fin:
                self._gui.root.destroy()
                
            features = self._feature_stream()
            agent_pos = self._agent(features) if self._render_agent else None
            all_inputs.append(features)
            all_targets.append(target)
            return agent_pos
        self._gui.set_callback(_callback)
        self._gui.root.after(3, self._gui.step)
        self._gui.root.mainloop()
        all_inputs = np.array(all_inputs)
        all_targets = np.array(all_targets)
        
        if test:
            return
        
        # Run training
        for _ in range(self._training_steps):
            self._optimizer.zero_grad()
            
            # Sample batch
            batch_inputs, batch_targets = _sample_batch(
                all_inputs, all_targets, batch_size=self._batch_size)
            batch_outputs = self._agent(batch_inputs, as_numpy=False)
            batch_targets = torch.from_numpy(batch_targets.astype(np.float32))
            
            # Evaluate loss
            loss = torch.mean(torch.sum(torch.square(
                batch_targets - batch_outputs), axis=1))
            
            # Backprop
            loss.backward()
            if self._grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self._agent.parameters(), self._grad_clip)
            self._optimizer.step()
        
        # Save model
        self._agent.snapshot()
        
        
def _get_boolean_arg(arg, name):
    if arg == 'True' or arg == True:
        arg = True
    elif arg == 'False' or arg == False:
        arg = False
    else:
        raise ValueError(f'Invalid {name} {arg}')
    return arg

        
def main(name,
         snapshot_name=None,
         render_agent=False,
         stream='lsl',
         test=False):
    render_agent = _get_boolean_arg(render_agent, name='render_agent')
    test = _get_boolean_arg(test, name='test')
    if stream == 'lsl':
        feature_stream = lsl_api.get_lsl_api()
    else:
        feature_stream = lsl_api.get_noise_api()
    gui = gui_lib.CalibrationGUI()
    
    calibrator = Calibrator(
        name=name,
        feature_stream=feature_stream,
        gui=gui,
        snapshot_name=snapshot_name,
        render_agent=render_agent,
    )
    calibrator(test=test)


if __name__ == '__main__':
    main(**dict(arg.split('=') for arg in sys.argv[1:]))
    