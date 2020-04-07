# This script simulates a robot performing SLAM in 2D

import numpy as np
import random
from matplotlib import pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.patches import Ellipse
import time

from python.lib.robot import move
from python.lib.sensors import observe_range_bearing, inv_observe_range_bearing
from python.lib.transforms import angle_to_rotation_matrix
from python.lib.ekf import EKFSLAM
from python.lib.utils import confidence_ellipse


def display(r, estimator, landmarks_true, landmarks_est, raw_measurements, sim_time):
    """
    Display robot and landmarks

    :param r: robot pose
    :param estimator: estimator object, which contains state estimates and state covariance matrix
    :param landmarks_true: list of true landmark locations
    :param landmarks_est: list of estimated landmark locations
    :param raw_measurements: sensor measurements of landmarks
    :param sim_time: simulation time
    """

    fig = plt.figure()

    # plot true robot states

    length = 0.8
    r_x, r_y, r_alpha = r
    plt.plot(r_x, r_y, ".r")
    plt.plot([r_x, r_x + length * np.cos(r_alpha)], [r_y, r_y + length * np.sin(r_alpha)], "-r")

    # plot estimated robot states

    r_x_est, r_y_est, r_alpha_est = estimator.X[:3]
    plt.plot(r_x_est, r_y_est, ".g")
    plt.plot([r_x_est, r_x_est + length * np.cos(r_alpha_est)], [r_y_est, r_y_est + length * np.sin(r_alpha_est)], "-g")

    # plot 2 sigma robot state uncertainty bounds
    confidence_ellipse(mean=estimator.X[:2], cov=estimator.P[:2, :2], ax=plt.gca(), n_std=2, facecolor="none", edgecolor="g", alpha=0.4)
    # confidence_ellipse(mean=estimator.X[:2], cov=np.array([[0.3, 0.], [0., 0.2]]), ax=plt.gca(), n_std=2, facecolor="none", edgecolor="k", alpha=0.4)

    # plot true landmarks
    trans_offset = mtransforms.offset_copy(plt.gca().transData, fig=fig, x=0.05, y=0.10, units='inches')

    for j in range(landmarks_true.shape[0]):
        # plot true landmarks
        landmk_x, landmk_y = landmarks_true[j, 0], landmarks_true[j, 1]
        meas_dist, meas_angle = raw_measurements[j]
        plt.plot(landmk_x, landmk_y, "xb")
        plt.text(landmk_x, landmk_y, '{landmk_id}: ({dist:0.2f},{angle:0.1f})'.
                 format(landmk_id=j, dist=meas_dist, angle=np.rad2deg(meas_angle)),
                 transform=trans_offset)

        # plot estimated landmarks
        landmk_x_est, landmk_y_est = landmarks_est[j]
        plt.plot(landmk_x_est, landmk_y_est, ".g")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Time: {}".format(sim_time))
    plt.grid()
    plt.axis('equal')
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
    r_true = [0., 0., np.deg2rad(90.)]

    # angular_velocity [rad/sec]. Used to simulate random walk on the angle control
    d_alpha = 0

    # process noise (for control input [x; alpha])
    u_x_stddev = 2.0 * time_step                    # [m]
    u_alpha_stddev = np.deg2rad(4.0 * time_step)    # [rad]

    # landmarks ([meters])
    min_x = -5.
    max_x = 5.
    min_y = 0.
    max_y = 10.
    num_landmarks = 6
    landmarks_true = np.random.random_sample((num_landmarks, 2))
    landmarks_true[:, 0] = min_x + landmarks_true[:, 0] * (max_x - min_x)
    landmarks_true[:, 1] = min_y + landmarks_true[:, 1] * (max_y - min_y)

    # EKF-SLAM estimator

    est = EKFSLAM()
    est.X[:3] = r_true      # we know the true robot pose initially
    est.P = np.zeros((3, 3)) * 1.
    est.N = np.array([[u_x_stddev**2, 0], [0, u_alpha_stddev**2]])

    # simulate robot
    for i in range(int(sim_duration / time_step)):
        t_start = time.time()

        # generate control input
        control_input_type = "straight_with_noise"

        if control_input_type == "straight_with_noise":
            u = [2. * time_step, 0]

            # generate perturbation
            n = [u_x_stddev * np.random.randn(), d_alpha]
        elif control_input_type == "straight_with_random_walk_angle_noise":
            u = [2. * time_step, 0]

            # generate perturbation
            d_alpha = d_alpha + u_alpha_stddev * np.random.randn()
            n = [u_x_stddev * np.random.randn(), d_alpha]
        elif control_input_type == "straight_without_noise":
            u = [2. * time_step, 0]

            # generate perturbation
            n = [0, 0]
        elif control_input_type == "circle_with_noise":
            u = [2. * time_step, np.deg2rad(15.) * time_step]
            n = [u_x_stddev * np.random.randn(), d_alpha]
        elif control_input_type == "circle_without_noise":
            u = [2. * time_step, np.deg2rad(15.) * time_step]
            n = [0, 0]
        else:
            raise ValueError("Use valid control input motion type")

        # move robot
        r_true = move(r_true, u, n)

        # propagate EKF
        est.state_propagation(u)

        print("States:", est.X, "\nP:", est.P)

        # sensor readings of environment
        R_true = angle_to_rotation_matrix(r_true[2])
        p_robot_world_true = np.array([r_true[0], r_true[1]])
        raw_measurements = [observe_range_bearing(R_true, p_robot_world_true, landmarks_true[j, :])
                            for j in range(landmarks_true.shape[0])]

        # estimated landmark positions (by using estimated robot pose and inverse sensor measurements)
        landmarks_est = [inv_observe_range_bearing(R_true, p_robot_world_true, m) for m in raw_measurements]

        print(time.time() - t_start)

        # plot robot and map
        if i % 5 == 0:
            print(i, "current pose:", r_true, ", control input:", u, ", noise:", n)
            display(r_true, est, landmarks_true, landmarks_est, raw_measurements, i)


if __name__ == "__main__":
    main()
