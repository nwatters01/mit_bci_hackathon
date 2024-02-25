"""MLP agent."""

from . import abstract_agent
import numpy as np
import torch


class MLPNet(torch.nn.Module):
    """MLPNet model."""
    
    def __init__(self, in_features, layer_features, activation=None):
        """Constructor."""
        super(MLPNet, self).__init__()

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
    
    
def _sample_batch(*arrays, batch_size):
    num_samples = arrays[0].shape[0]
    indices = np.random.choice(num_samples, size=batch_size)
    batch_arrays = [x[indices] for x in arrays]
    return batch_arrays
    
    
class MLP(abstract_agent.AbstractAgentTorch):
    
    def __init__(self,
                 name,
                 gain=1.,
                 target_action_scale=1.,
                 in_features=2,
                 layer_features=(256, 2),
                 batch_size=64,
                 training_steps=100,
                 optimizer='SGD',
                 lr=0.001,
                 grad_clip=1):
        if isinstance(gain, str):
            gain = float(gain)
        super(MLP, self).__init__(
            name=name,
            gain=gain,
            target_action_scale=target_action_scale,
            in_features=in_features,
            layer_features=layer_features,
            batch_size=batch_size,
            training_steps=training_steps,
            optimizer=optimizer,
            lr=lr,
            grad_clip=grad_clip,
        )
        self._gain = gain
        self._target_action_scale = target_action_scale
        self._model = MLPNet(
            in_features=in_features, layer_features=layer_features)
        self._batch_size = batch_size
        self._training_steps = training_steps
        optimizer_class = getattr(torch.optim, optimizer)
        self._optimizer = optimizer_class(self._model.parameters(), lr=lr)
        self._grad_clip = grad_clip
        
        self.data = {}
        self._prev_trial_index = None
        
    def _eval(self, x, as_numpy=False):
        model_output = self._model(torch.from_numpy(x.astype(np.float32)))
        output = self._gain * model_output
        if as_numpy:
            return output.detach().numpy()
        else:
            return output
        
    def _train(self, inputs, targets):
        """Run training loop."""
        
        # Scale targets
        targets *= self._target_action_scale
        
        # Run training loop
        training_losses = []
        for _ in range(self._training_steps):
            self._optimizer.zero_grad()
            
            # Sample batch
            batch_inputs, batch_targets = _sample_batch(
                inputs, targets, batch_size=self._batch_size)
            batch_outputs = self._eval(batch_inputs)
            batch_targets = torch.from_numpy(batch_targets.astype(np.float32))
            
            # Evaluate loss
            loss = torch.mean(torch.sum(torch.square(
                batch_targets - batch_outputs), axis=1))
            
            # Backprop
            loss.backward()
            if self._grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self._model.parameters(), self._grad_clip)
            self._optimizer.step()
            training_losses.append(float(loss.detach()))
            
        return training_losses
        
    def reset(self, trial_index):
        """Fit linear regression."""
        
        if self._prev_trial_index is not None:
            # Convert data from previous trial into numpy array
            self.data[self._prev_trial_index] = {
                'input': np.array(self.data[self._prev_trial_index]['input']),
                'output': np.array(self.data[self._prev_trial_index]['output']),
            }
            
            # Train
            data_keys = sorted(self.data.keys())
            data_input = np.concatenate(
                [self.data[k]['input'] for k in data_keys], axis=0)
            data_output = np.concatenate(
                [self.data[k]['output'] for k in data_keys], axis=0)
            self._train(data_input, data_output)
        
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
        agent_input = self._extract_agent_input(agent_input)
        agent_action = self._eval(agent_input[None], as_numpy=True)[0]
        return agent_action