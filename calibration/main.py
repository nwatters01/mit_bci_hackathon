"""Environment calibration 0."""

import agents
import environments
import gui
import json
from pathlib import Path
import pil_renderer
import sys
import torch

_RENDER_SIZE = 700
_GUI_SCALE = 1.
_SNAPSHOT_DIR = Path('./snapshots/')


def main(env, name, agent=None, seed=None, **agent_kwargs):
    # Create renderer
    renderer = pil_renderer.PILRenderer(image_size=(_RENDER_SIZE, _RENDER_SIZE))
    
    # Create agent
    if seed is None:
        # Create randomly initialized agent
        agent_instance = getattr(agents, agent)(name=name, **agent_kwargs)
    else:
        # Load agent from snapshot
        snapshot_dir = _SNAPSHOT_DIR / seed
        print(f'\nLoading seed from {snapshot_dir}\n')
        kwargs_path = snapshot_dir / 'kwargs.json'
        kwargs = json.load(open(kwargs_path, 'r'))
        kwargs.update(agent_kwargs)
        class_path = snapshot_dir / 'class.json'
        agent_class = json.load(open(class_path, 'r'))
        agent_instance = getattr(agents, agent_class)(name=name, **kwargs)
        state_dict_path = snapshot_dir / 'state_dict'
        agent_instance.load_state_dict(torch.load(state_dict_path))
        
    # Create environment
    env_instance = getattr(environments, env)(
        renderer=renderer, agent=agent_instance)
    
    # Run GUI
    gui.GUI(env_instance, render_size=int(_GUI_SCALE * _RENDER_SIZE))


if __name__ == '__main__':
    main(**dict(arg.split('=') for arg in sys.argv[1:]))