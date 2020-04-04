# This script simulates a robot performing SLAM in 2D

import numpy as np
import random
from matplotlib import pyplot as plt

from python.lib.robot import move


def display(r, landmarks, sim_time):
    """
    Display robot and landmarks

    :param r: robot pose
    :param landmarks: list of landmark locations
    :param sim_time: simulation time
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

    # TODO: plot landmark IDs to help with debugging sensor readings

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Time: {}".format(sim_time))
    plt.grid()
    plt.show()


def main():
    # random seed
    random.seed(20)
    np.random.seed(20)

    # simulation time step (Kalman filter propagation frequency) [s]
    time_step = 0.020

    # measurement update period [s]
    measurement_period = 0.5

    # total simulation duration [s]
    sim_duration = 5.

    # initial robot pose (x [m]; y [m]; alpha [deg])
    r = [0., 0., 90.]

    # angular_velocity [deg/sec]. Used to simulate random walk on the angle control
    d_alpha = 0

    # process noise (for control input [x; alpha])
    u_x_stddev = 0.0 * time_step       # [m]
    u_alpha_stddev = 4.0 * time_step     # [deg]

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
        u = [2. * time_step, 0]

        # generate perturbation
        d_alpha = d_alpha + u_alpha_stddev * np.random.randn()
        n = [u_x_stddev * np.random.randn(), d_alpha]
        # n = [0, 0]

        # move robot
        r = move(r, u, n)

        # plot robot and map
        if i % 5 == 0:
            print(i, "current pose:", r, ", control input:", u, ", noise:", n)
            display(r, landmarks, i)


if __name__ == "__main__":
    main()
