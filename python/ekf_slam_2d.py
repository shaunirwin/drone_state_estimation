# This script simulates a robot performing SLAM in 2D

import numpy as np
import random
from matplotlib import pyplot as plt

from python.lib.robot import move


def display(r, landmarks):
    """
    Display robot and landmarks

    :param r: robot pose
    :param landmarks: list of landmark locations
    """

    plt.figure()

    # plot robot

    length = 0.8
    r_x = r[0]
    r_y = r[1]
    r_alpha = np.deg2rad(r[2])

    plt.plot(r_x, r_y, ".r")
    plt.plot([r_x, r_x + length * np.cos(r_alpha)], [r_y, r_y + length * np.sin(r_alpha)], "-r")

    # plot landmarks
    plt.plot(landmarks[:, 0], landmarks[:, 1], "xb")

    # TODO: plot landmark IDs o help with debugging sensor readings

    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.show()


def main():
    # random seed
    random.seed(20)
    np.random.seed(20)

    # simulation time step (Kalman filter propagation frequency) [s]
    time_step = 0.020

    # measurement update period [s]
    measurement_period = 0.1

    # total simulation duration [s]
    sim_duration = 5.

    # initial robot pose (x [m]; y [m]; alpha [deg])
    r = [0., 0., 90.]

    # process noise (for control input [x; alpha])
    u_x_stddev = 0.05 * time_step       # [m]
    u_alpha_stddev = 4. * time_step     # [deg]

    # landmarks ([meters])
    min_x = -5.
    max_x = 5.
    min_y = 0.
    max_y = 10.
    num_landmarks = 6
    landmarks = np.random.random_sample((num_landmarks, 2))
    landmarks[:, 0] = min_x + landmarks[:, 0] * (max_x - min_x)
    landmarks[:, 1] = min_y + landmarks[:, 1] * (max_y - min_y)

    # simulate robot
    for i in range(int(sim_duration / time_step)):
        # generate control input (for now just try drive straight, ignoring perturbations)
        u = [1. * time_step, 0]

        # generate perturbation
        n = [u_x_stddev * np.random.randn(), u_alpha_stddev * np.random.randn()]

        # move robot
        r = move(r, u, n)

        # plot robot and map
        if i % 10 == 0:
            display(r, landmarks)


if __name__ == "__main__":
    main()
