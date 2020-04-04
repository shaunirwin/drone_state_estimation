import numpy as np


class EKFSLAM:
    def __init__(self):
        # state vector (start off not seeing any landmarks)
        self.X = np.array([0, 0, 0])

        # state covariance matrix
        self.P = np.eye(3, 3)

    def jacobian_X_r(self, x_u, x_n, alpha_r, alpha_u, alpha_n):
        """
        Compute the Jacobian of the state transition matrix (just the robot pose states) w.r.t. robot pose

        :param x_u:
        :param x_n:
        :param alpha_r:
        :param alpha_u:
        :param alpha_n:
        :return:
        """

        return np.array([[1, 0, -(x_n + x_u)*np.sin(alpha_n + alpha_r + alpha_u)],
                        [0, 1,  (x_n + x_u)*np.cos(alpha_n + alpha_r + alpha_u)],
                        [0, 0, 1]])

    def jacobian_N(self, x_u, x_n, alpha_r, alpha_u, alpha_n):
        """
        Compute the Jacobian of the state transition matrix (just the robot pose states) w.r.t. input noise perturbation

        :param x_u:
        :param x_n:
        :param alpha_r:
        :param alpha_u:
        :param alpha_n:
        :return:
        """

        return np.array([[np.cos(alpha_n + alpha_r + alpha_u), -(x_n + x_u)*np.sin(alpha_n + alpha_r + alpha_u)],
                        [np.sin(alpha_n + alpha_r + alpha_u),  (x_n + x_u)*np.cos(alpha_n + alpha_r + alpha_u)],
                        [0, 1]])


def ekf_covariance_propagation(F_x, P, F_n, N):
    """
    Propagation update of the EKF

    Note that this is the naive way of doing the calculation. In practice, since the landmarks do not move their
    covariance is always zero and are unaffected by process noise. Therefore, we could

    :param F_x: linearised state transition function w.r.t. the state vectors
    :param P: state covariance matrix
    :param F_n: linearised state transition function w.r.t the process noise
    :param N: process noise covariance matrix
    :return: state covariance matrix after propagation
    """

    P_new = np.dot(np.dot(F_x, P), F_x.T) + np.dot(np.dot(F_n, N), F_n.T)

    return P_new
