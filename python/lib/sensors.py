from python.lib import transforms

import os
if os.environ.get("USE_JAX", False):
    import jax.numpy as np
    from jax.config import config
    # check that JAX is set to 64-bit precision
    assert config.values["jax_enable_x64"]
else:
    import numpy as np
from math import sqrt


class RangeBearingSensor:
    @staticmethod
    def observe_range_bearing(X_r, p_world_rect):
        """
        Simulate a range-bearing sensor reading of a landmark in world coordinates

        :param X_r: robot pose (true)
        :param p_world_rect: (x, y) rectangular coordinate position of a landmark in world ref frame
        :return: range-bearing (polar coordinate) measurement of a landmark (local ref frame)
        """

        t = X_r[:2]
        alpha = X_r[2]

        R = transforms.angle_to_rotation_matrix(alpha)
        return transforms.rect_to_polar(transforms.rigid_transform_world_to_local(R, t, p_world_rect))

    @staticmethod
    def inv_observe_range_bearing(X_r, p_local_polar):
        """
        Simulate inverse range-bearing sensor reading of a landmark in world coordinates

        :param X_r: robot pose (true)
        :param p_local_polar: (range, bearing) polar coordinate position of a landmark in local ref frame
        :return: estimated position of landmark in rectangular coordinates (world ref frame)
        """

        t = X_r[:2]
        alpha = X_r[2]

        R = transforms.angle_to_rotation_matrix(alpha)
        return transforms.rigid_transform_local_to_world(R, t, transforms.polar_to_rect(p_local_polar))

    @staticmethod
    def jacobian_H_X_r(x_r, y_r, alpha_r, l_i_x, l_i_y):
        """
        Compute the Jacobian of the range-bearing sensor observation function w.r.t. robot states X_r (x, y, angle)

        :param x_r:
        :param y_r:
        :param alpha_r:
        :param l_i_x:
        :param l_i_y:
        :return:
        """
        s_a = np.sin(alpha_r)
        c_a = np.cos(alpha_r)

        return np.array([
            [((-(l_i_x - x_r) * s_a + (l_i_y - y_r) * c_a) * s_a - (
                        (l_i_x - x_r) * c_a + (l_i_y - y_r) * s_a) * c_a) / sqrt(
                (-(l_i_x - x_r) * s_a + (l_i_y - y_r) * c_a) ** 2 + (
                            (l_i_x - x_r) * c_a + (l_i_y - y_r) * s_a) ** 2), (
                         -(-(l_i_x - x_r) * s_a + (l_i_y - y_r) * c_a) * c_a - (
                             (l_i_x - x_r) * c_a + (l_i_y - y_r) * s_a) * s_a) / sqrt(
                (-(l_i_x - x_r) * s_a + (l_i_y - y_r) * c_a) ** 2 + (
                            (l_i_x - x_r) * c_a + (l_i_y - y_r) * s_a) ** 2), (
                         (2 * (-l_i_x + x_r) * c_a - 2 * (l_i_y - y_r) * s_a) * (
                             -(l_i_x - x_r) * s_a + (l_i_y - y_r) * c_a) / 2 + (
                                     -2 * (l_i_x - x_r) * s_a + 2 * (l_i_y - y_r) * c_a) * (
                                     (l_i_x - x_r) * c_a + (l_i_y - y_r) * s_a) / 2) / sqrt(
                (-(l_i_x - x_r) * s_a + (l_i_y - y_r) * c_a) ** 2 + (
                            (l_i_x - x_r) * c_a + (l_i_y - y_r) * s_a) ** 2)],
            [-((l_i_x - x_r) * s_a - (l_i_y - y_r) * c_a) * c_a / (
                        (-(l_i_x - x_r) * s_a + (l_i_y - y_r) * c_a) ** 2 + (
                            (l_i_x - x_r) * c_a + (l_i_y - y_r) * s_a) ** 2) + (
                         (l_i_x - x_r) * c_a + (l_i_y - y_r) * s_a) * s_a / (
                         (-(l_i_x - x_r) * s_a + (l_i_y - y_r) * c_a) ** 2 + (
                             (l_i_x - x_r) * c_a + (l_i_y - y_r) * s_a) ** 2),
             -((l_i_x - x_r) * s_a - (l_i_y - y_r) * c_a) * s_a / (
                         (-(l_i_x - x_r) * s_a + (l_i_y - y_r) * c_a) ** 2 + (
                             (l_i_x - x_r) * c_a + (l_i_y - y_r) * s_a) ** 2) - (
                         (l_i_x - x_r) * c_a + (l_i_y - y_r) * s_a) * c_a / (
                         (-(l_i_x - x_r) * s_a + (l_i_y - y_r) * c_a) ** 2 + (
                             (l_i_x - x_r) * c_a + (l_i_y - y_r) * s_a) ** 2),
             ((-l_i_x + x_r) * c_a - (l_i_y - y_r) * s_a) * (
                         (l_i_x - x_r) * c_a + (l_i_y - y_r) * s_a) / (
                         (-(l_i_x - x_r) * s_a + (l_i_y - y_r) * c_a) ** 2 + (
                             (l_i_x - x_r) * c_a + (l_i_y - y_r) * s_a) ** 2) + (
                         -(l_i_x - x_r) * s_a + (l_i_y - y_r) * c_a) * (
                         (l_i_x - x_r) * s_a - (l_i_y - y_r) * c_a) / (
                         (-(l_i_x - x_r) * s_a + (l_i_y - y_r) * c_a) ** 2 + (
                             (l_i_x - x_r) * c_a + (l_i_y - y_r) * s_a) ** 2)]])

    @staticmethod
    def jacobian_H_L_i(x_r, y_r, alpha_r, l_i_x, l_i_y):
        """
        Compute the Jacobian of the range-bearing sensor observation function w.r.t. landmark L_i position

        :param x_r:
        :param y_r:
        :param alpha_r:
        :param l_i_x:
        :param l_i_y:
        :return:
        """
        s_a = np.sin(alpha_r)
        c_a = np.cos(alpha_r)

        return np.array([
                [(-(-(l_i_x - x_r) * s_a + (l_i_y - y_r) * c_a) * s_a + (
                            (l_i_x - x_r) * c_a + (l_i_y - y_r) * s_a) * c_a) / sqrt(
                    (-(l_i_x - x_r) * s_a + (l_i_y - y_r) * c_a) ** 2 + (
                                (l_i_x - x_r) * c_a + (l_i_y - y_r) * s_a) ** 2), (
                             (-(l_i_x - x_r) * s_a + (l_i_y - y_r) * c_a) * c_a + (
                                 (l_i_x - x_r) * c_a + (l_i_y - y_r) * s_a) * s_a) / sqrt(
                    (-(l_i_x - x_r) * s_a + (l_i_y - y_r) * c_a) ** 2 + (
                                (l_i_x - x_r) * c_a + (l_i_y - y_r) * s_a) ** 2)],
                [((l_i_x - x_r) * s_a - (l_i_y - y_r) * c_a) * c_a / (
                            (-(l_i_x - x_r) * s_a + (l_i_y - y_r) * c_a) ** 2 + (
                                (l_i_x - x_r) * c_a + (l_i_y - y_r) * s_a) ** 2) - (
                             (l_i_x - x_r) * c_a + (l_i_y - y_r) * s_a) * s_a / (
                             (-(l_i_x - x_r) * s_a + (l_i_y - y_r) * c_a) ** 2 + (
                                 (l_i_x - x_r) * c_a + (l_i_y - y_r) * s_a) ** 2),
                 ((l_i_x - x_r) * s_a - (l_i_y - y_r) * c_a) * s_a / (
                             (-(l_i_x - x_r) * s_a + (l_i_y - y_r) * c_a) ** 2 + (
                                 (l_i_x - x_r) * c_a + (l_i_y - y_r) * s_a) ** 2) + (
                             (l_i_x - x_r) * c_a + (l_i_y - y_r) * s_a) * c_a / (
                             (-(l_i_x - x_r) * s_a + (l_i_y - y_r) * c_a) ** 2 + (
                                 (l_i_x - x_r) * c_a + (l_i_y - y_r) * s_a) ** 2)]])

    @staticmethod
    def jacobian_G_X_r(x_r, y_r, alpha_r, rho, psi):
        """
        Compute the Jacobian of the inverse range-bearing sensor observation function w.r.t. robot position

        :param x_r: x robot state estimate
        :param y_r: y robot state estimate
        :param alpha_r: alpha (angle) robot state estimate
        :param rho: range measurement value
        :param psi: bearing measurement value
        :return:
        """

        return np.array([[1, 0, -rho * np.sin(alpha_r) * np.cos(psi) - rho * np.sin(psi) * np.cos(alpha_r)],
                         [0, 1, -rho * np.sin(alpha_r) * np.sin(psi) + rho * np.cos(alpha_r) * np.cos(psi)]])

    @staticmethod
    def jacobian_G_y_i(x_r, y_r, alpha_r, rho, psi):
        """
        ompute the Jacobian of the inverse range-bearing sensor observation function w.r.t. range-bearing measurement

        :param x_r: x robot state estimate
        :param y_r: y robot state estimate
        :param alpha_r: alpha (angle) robot state estimate
        :param rho: range measurement value
        :param psi: bearing measurement value
        :return:
        """

        return np.array([
            [-np.sin(alpha_r) * np.sin(psi) + np.cos(alpha_r) * np.cos(psi),
             -rho * np.sin(alpha_r) * np.cos(psi) - rho * np.sin(psi) * np.cos(alpha_r)],
            [np.sin(alpha_r) * np.cos(psi) + np.sin(psi) * np.cos(alpha_r),
             -rho * np.sin(alpha_r) * np.sin(psi) + rho * np.cos(alpha_r) * np.cos(psi)]
        ])

