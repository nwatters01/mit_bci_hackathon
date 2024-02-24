"""Calibration 0 environment."""

from matplotlib import pyplot as plt
import numpy as np
import wiggles


def main():
    x = wiggles.sample_wiggle()
    
    _, ax = plt.subplots()
    ax.scatter(x[:, 0], x[:, 1])
    ax.set_aspect('equal')
    plt.show()


if __name__ == '__main__':
    main()