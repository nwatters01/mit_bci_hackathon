import time
from typing import Tuple, List
from duckietown.sdk.robots.duckiebot import DB21J
from duckietown.sdk.types import LEDsPattern, RGBAColor
import matplotlib.pyplot as plt
import cv2

SIMULATED_ROBOT_NAME: str = "map_0/vehicle_0"
REAL_ROBOT_NAME: str = "rover"

robot: DB21J = DB21J(SIMULATED_ROBOT_NAME, simulated=True)  # change accordingly


def add_sample(samples, new_sample, window_size):
    """
    Adds a new sample to the list of samples and calculates the triangular moving mean.
    
    Parameters:
    - samples: List of the most recent samples.
    - new_sample: The new sample to add.
    - window_size: The size of the moving window for the mean calculation.
    
    Returns:
    - The triangular moving mean of the latest samples, considering a window size.
    """
    samples.append(new_sample)  # Add new sample
    
    # Calculate the first moving mean
    if len(samples) > window_size:
        first_moving_means = [sum(samples[i:i+window_size])/window_size for i in range(len(samples)-window_size+1)]
    else:
        first_moving_means = samples[:]  # Copy the list if it's shorter than window_size
    
    # Calculate the second moving mean on the results of the first moving mean
    if len(first_moving_means) > window_size:
        second_moving_means = [sum(first_moving_means[i:i+window_size])/window_size for i in range(len(first_moving_means)-window_size+1)]
        return second_moving_means[-1]  # Return the last element as the current triangular moving mean
    elif len(first_moving_means) > 1:
        return sum(first_moving_means) / len(first_moving_means)  # Average if there are enough samples
    else:
        return new_sample
    
GAIN = 0.5
SENSITIVITY = 0.5

R = 1
THETA = -1

Left_Wheel = GAIN*(R + SENSITIVITY*THETA)
Right_Wheel = GAIN*(R - SENSITIVITY*THETA)

WINDOW_SIZE= 5
samples_left = []
samples_right = []

# move wheels
robot.motors.start()
stime: float = time.time()
while time.time() - stime < 2:
    Smoothed_Left_Wheel = add_sample(samples_left, Left_Wheel, WINDOW_SIZE)
    Smoothed_Right_Wheel = add_sample(samples_right, Right_Wheel, WINDOW_SIZE)

    speeds = (Smoothed_Left_Wheel, Smoothed_Right_Wheel)
    robot.motors.publish(speeds)

    time.sleep(0.25)
    print("speedy")
robot.motors.stop()
print("Stopped.")


# # LED lights show
# frequency: float = 1.4
# off: RGBAColor = (0, 0, 0, 0.0)
# amber: RGBAColor = (1, 0.7, 0, 1.0)
# lights_on: LEDsPattern = LEDsPattern(front_left=amber, front_right=amber, rear_right=amber, rear_left=amber)
# lights_off: LEDsPattern = LEDsPattern(front_left=off, front_right=off, rear_right=off, rear_left=off)
# pattern: List[LEDsPattern] = [lights_on, lights_off]
# robot.lights.start()
# stime: float = time.time()
# i: int = 0
# while time.time() - stime < 8:
#     lights: LEDsPattern = pattern[i % 2]
#     robot.lights.publish(lights)
#     time.sleep(1. / frequency)
#     i += 1
# robot.lights.stop()
# print("Stopped.")


# # ToF
# def _range_finder_cb(data):
#     if data is None:
#         print("Out of range.")
#     return print(f"Range: {data} meters.")

# robot.range_finder.start()
# stime: float = time.time()
# while time.time() - stime < 2:
#     data = robot.range_finder.capture(block=True)
#     _range_finder_cb(data)
# print("Stopped.")
# robot.range_finder.stop()


# # Camera --- not working now
# def _camera_cb(data):
#     print(f"Received image of shape: {data.shape}")
    
#     cv2.imshow("Live Stream", data[:, :, ::-1])
#     cv2.waitKey(1)
#     # plt.plot(1,1), plt.imshow(data, interpolation='nearest')  
#     # plt.pause(0.001)
#     # plt.show()

# robot.camera.start()
# stime: float = time.time()
# # plt.ion()
# while time.time() - stime < 2:
#     data = robot.camera.capture(block=True)
#     _camera_cb(data)

# print("Stopped.")
# robot.camera.stop()
