# Code for MIT BCI Hackathon on Feb 25, 2024.

This code was written by Nick Watters, John Gabel, and Mahdi Ramadan. It
implements their (winnin) solution for the
[MIT BCI Hackathon](https://bci-i.github.io/hackathon) on Feb 25, 2024. The
hackathon involved controlling a duckie robot using an EEG headset. The goal was
to navigate an obstacle course accurately and efficiently.

## Approach

Our approach was to use the following pipeline:

[EEG input] -> [EEG features] -> [Calibrated (rotation, speed) actions] ->
[robot wheel speeds]

The convert EEG input to EEG features, we first filtered the data with a finite
impulse response filter, then took the rolling standdard deviation of each EEG
channel (which roughly corresponds to the envelope of the EEG). See `lsl_api.py`
for details.

To conver EEG features to calibrated (rotation, speed) actions, we ran a
calibration protocol during which a subject tracked a ball representing
(rotation, speed) as it traversed a display, and an MLP neural network learned
the mapping from EEG features to (rotation, speed). See `calibration.py` for
details.

To map (rotation, speed) actions to robot wheel speeds, we implemented a control
policy that set wheel speed differential proportional to the rotation. This
control policy included a few extra tricks, like smoothing and turn rebounding.
See `run_duckie.py` for details.

## Installation and setup

### EEG setup for X.on EEG

Set up the EEG hardware:
* Soak the EEG sponges in salt water and squeeze them out so they are not
  dripping wet. 
* Use forceps to insert the sponges into the sponge-holders, then attach the
  sponge-holders to the EEG. 
* Put on the EEG: Adjust the strap so that the headset stays still on the head.
* Try to move hair out of the way so that the sensors are as close as possible 
  to the skin. Lastly, put a bit of salt water on the ground clip and attach it
  to the earlobe.
* Download the X.on EEG app: Make sure to use an Android device. It is not 
  available on the Google Play Store so you will need a private link from the
  EEG company.
* Power on the EEG. There is a power button on the back of the device. You
  should see the serial number pop up on the app. Click on the device to
  connect. You can use the app to check the electrode impedances and control
  various settings like filters. 
* Stream the EEG data: Click the ‘monitor’ view on the app to start streaming
  the data. Make sure your computer is connected to the same wifi network as the
  Android. Note that we were not able to use large (e.g. university-wide) wifi
  networks. 
* Connect the computer: Run the following python code from the BCI Initiative
  team to ensure the EEG is streaming data.

```
import pylsl

# Resolve EEG streams
streams = pylsl.resolve_streams()

# Iterate through streams
print(f"Found {len(streams)} streams")
print("---------------")

for stream in streams:
    print("Stream Name:", stream.name())
    print("Stream Type:", stream.type())
    print("Stream ID:", stream.source_id())   # this should match your X.on Serial Number
    print("Stream Unique Identifier:", stream.uid())
    print("---------------")
```


### Duckie and duckiematrix setup

Follow the instructions on the
[hackathon website](https://bci-i.github.io/hackathon-materials) to set up
duckiematrix. Note that it requires native python (no virtual environment) and
only works on some computers (notably does not work on apple computer with M1 or
M2 chips).

### Dependencies

This codebase depends on the `pylsl` and `torch`, which can be installed with:
```
pip install pylsl
pip install torch
```

## Usage

To stream and visualize EEG data and features, run `$ python3 lsl_api.py`.

To run calibration of the (rotation, speed) action space, run
`$ calibration.py`.

To run and control a duckie robot, run `run_duckie.py`. See parameters and
documentation in that file for how to run in duckiematrix and with a physical
duckie robot.