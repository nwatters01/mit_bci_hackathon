"""Calibration 0 environment."""

from matplotlib import pyplot as plt
import numpy as np
import navigation


def main():
    # x = navigation.sample_wiggle()
    x = navigation.sample_track()
    
    _, ax = plt.subplots()
    ax.scatter(x[:, 0], x[:, 1])
    ax.set_aspect('equal')
    plt.show()


if __name__ == '__main__':
    main()