# Usage

Create conda environment with `$ conda create -n duckie python=3.10`

Then run the following:
```
$ pip install moog-games
$ pip install torch
```

Then to run calibration, run
```
python3 main.py env=CalFeedback agent=MLP name=version_0
```

Use the arrow keys in the ITI to advance to the next trial. Use the mouse to give inputs. Quick the application (by pressinve Esc) when you want to save and exit.

If after doing that you want to fine-tune on more complex tasks, run another task, loading the one you just began fitting as the seed, e.g.
```
python3 main.py env=Wiggles seed=version_0 name=verion_1
```

Another good environment is `Track`.