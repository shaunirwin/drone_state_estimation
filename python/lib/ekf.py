import os
if os.environ.get("USE_JAX", False):
    import jax.numpy as np
    from jax.config import config
    # check that JAX is set to 64-bit precision
    assert config.values["jax_enable_x64"]
else:
    import numpy as np
from math import sqrt

from python.lib.robot import move
from python.lib.sensors import observe_range_bearing, inv_observe_range_bearing
from python.lib.transforms import angle_to_rotation_matrix


class EKFSLAM:
    """
    This EKF-SLAM estimator is based on https://jinyongjeong.github.io/images/post/SLAM/lec05_EKF_SLAM/EKF.pdf.
    """

    def __init__(self):
        # state vector (start off not seeing any landmarks).
        # In general X = [R; M], where R (x, y, angle) is the robot pose and M (L_0, ..., L_n) are the landmark
        # positions.
        self.X = np.array([0, 0, 0.], dtype=np.float64)

        # state covariance matrix
        self.P = np.zeros((3, 3), dtype=np.float64)

        # process noise covariance matrix
        self.Q = np.eye(2, 2, dtype=np.float64)

        # measurement noise covariance matrix
        self.R = np.eye(2, 2, dtype=np.float64)

        # X, P, Q and R should be initialised accordingly prior to running the estimator

        # store lookup table of the landmarks
        # Landmark index is calculated from its index in this list. ID of landmark is the value stored in the list.
        self.landmark_lookup = []

    def get_num_landmarks(self):
        """
        Return number of landmarks

        :return:
        """

        return len(self.landmark_lookup)

    def get_landmark_id(self, i):
        """
        Get the ID of the ith landmark

        :param i: landmark index
        :return:
        """

        return self.landmark_lookup[i + 3]

    def new_landmark_range_bearing(self, y_meas, landmark_id=None):
        """
        Append new landmark to the state vector and state covariance matrix

        This assumes a measurement is obtained using a range-bearing sensor

        :param landmark_id: unique value. If None, a unique value is created here
        :param y_meas: measurement of new landmark using range-bearing sensor
        :return:
        """

        if landmark_id is None:
            raise NotImplementedError("Need to still implement this. Must generate a new unique ID.")

        if landmark_id in self.landmark_lookup:
            raise ValueError("Landmark ID already exists. Must be unique.")

        self.landmark_lookup.append(landmark_id)

        # self.X = np.concatenate([self.X, L_est_i])

    @staticmethod
    def jacobian_f_X_r(x_u, x_n, alpha_r, alpha_u, alpha_n):
        """
        Compute the Jacobian of the state transition function (just the robot pose states) w.r.t. robot pose

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

    @staticmethod
    def jacobian_f_N(x_u, x_n, alpha_r, alpha_u, alpha_n):
        """
        Compute the Jacobian of the state transition function (just the robot pose states) w.r.t. input noise perturbation

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

    def state_propagation(self, U):
        """
        Propagate the states and state covariance matrix forward one time step using the control input

    @staticmethod
    @staticmethod
    def state_propagation(X, U, n=None):
        """
        Propagate the state estimates forward one time step using the control input

        :param X: previous states
        :param U: control input (d_x, d_alpha)
        :param n: perturbation to control input (d_x, d_alpha)
        :return:
        """

        # ----------  propagate state vector (update robot pose but leave landmarks unchanged) -------

        r = X[:3]
        if n is None:
            n = np.array([0, 0])
        r_new = move(r, U, n)
        # X[:3] = r_new
        X_new = np.concatenate((r_new, X[3:]))

        return X_new

    def state_and_state_cov_propagation(self, U):
        """
        Propagate the state estimates and state covariance matrix forward one time step using the control input

        :param U: control input (d_x, d_alpha)
        :return:
        """

        # ----------  propagate state vector (update robot pose but leave landmarks unchanged) -------

        self.X = self.state_propagation(self.X, U)

        # ----------- propagate state covariance matrix ------------------

        # The full state cov matrix propagation can be done as follows:
        # P_new = np.dot(np.dot(F_x, P), F_x.T) + np.dot(np.dot(F_n, N), F_n.T)
        # However, this is less efficient that way due to many zeros, since since the landmarks do not move their
        # covariance is always zero and are unaffected by process noise. We can partition P as follows:
        # P = [[P_rr, P_rm], [P_mr, P_mm]]. P_mm will always be zero.

        x_u, alpha_u = U
        x_n, alpha_n = 0, 0     # N     # TODO: do we have access to this?
        alpha_r = self.X[2]     # TODO: should we use the state estimate from the previous timestep rather?

        F_x = self.jacobian_f_X_r(x_u=x_u, x_n=x_n, alpha_r=alpha_r, alpha_u=alpha_u, alpha_n=alpha_n)
        F_n = self.jacobian_f_N(x_u=x_u, x_n=x_n, alpha_r=alpha_r, alpha_u=alpha_u, alpha_n=alpha_n)

        # update covariance of robot pose
        P_rr = self.P[:3, :3]
        P_rr_new = np.dot(np.dot(F_x, P_rr), F_x.T) + np.dot(np.dot(F_n, self.Q), F_n.T)
        self.P[:3, :3] = P_rr_new

        # only update cross variance elements if there are landmarks present
        if self.X.shape[0] > 3:
            # update cross variance of robot pose and landmarks (P_rm)
            # NB: this step has algorithmic complexity of O(n)
            P_rm = self.P[:2, 3:]
            P_rm_new = np.dot(F_x, P_rm)
            self.P[:2, 3:] = P_rm_new

            # update cross variance of landmarks and robot pose (P_mr)
            self.P[3:, :2] = P_rm_new.T

        # transpose to ensure positive semi definite (NB: we could also store just a triangular matrix instead)
        self.P = (self.P + self.P.T) / 2

    def measurement_update(self, y_meas, landmark_index):
        """
        Perform measurement update with selected landmark

        :param y_meas:
        :param landmark_index:
        :return:
        """

        pass
