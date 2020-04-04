# This script simulates a robot performing SLAM in 2D

import numpy as np
import random
from matplotlib import pyplot as plt
import matplotlib.transforms as mtransforms

from python.lib.robot import move
from python.lib.sensors import observe_range_bearing
from python.lib.transforms import angle_to_rotation_matrix


def display(r, landmarks, raw_measurements, sim_time):
    """
    Display robot and landmarks

    :param r: robot pose
    :param landmarks: list of landmark locations
    :param raw_measurements: sensor measurements of landmarks
    :param sim_time: simulation time
    """

    fig = plt.figure()

    # plot robot

    length = 0.8
    r_x = r[0]
    r_y = r[1]
    r_alpha = r[2]

    plt.plot(r_x, r_y, ".r")
    plt.plot([r_x, r_x + length * np.cos(r_alpha)], [r_y, r_y + length * np.sin(r_alpha)], "-r")

    # plot landmarks
    trans_offset = mtransforms.offset_copy(plt.gca().transData, fig=fig, x=0.05, y=0.10, units='inches')

    for j in range(landmarks.shape[0]):
        landmk_x, landmk_y = landmarks[j, 0], landmarks[j, 1]
        meas_dist, meas_angle = raw_measurements[j]
        plt.plot(landmk_x, landmk_y, "xb")
        plt.text(landmk_x, landmk_y, '({dist:0.2f},{angle:0.1f})'.format(dist=meas_dist, angle=np.rad2deg(meas_angle)),
                 transform=trans_offset)

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
    r = [0., 0., np.deg2rad(90.)]

    # angular_velocity [rad/sec]. Used to simulate random walk on the angle control
    d_alpha = 0

    # process noise (for control input [x; alpha])
    u_x_stddev = 0.0 * time_step                    # [m]
    u_alpha_stddev = np.deg2rad(4.0 * time_step)    # [rad]

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

        # sensor readings of environment
        R_true = angle_to_rotation_matrix(r[2])
        p_robot_world_true = np.array([r[0], r[1]])
        raw_measurements = [observe_range_bearing(R_true, p_robot_world_true, landmarks[j, :])
                            for j in range(landmarks.shape[0])]

        # plot robot and map
        if i % 5 == 0:
            print(i, "current pose:", r, ", control input:", u, ", noise:", n)
            display(r, landmarks, raw_measurements, i)


if __name__ == "__main__":
    main()
