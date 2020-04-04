# Various transforms between reference frames and coordinate systems

import numpy as np


def rigid_transform_local_to_world(R, t, p_local):
    """
    Apply rigid body transform from local reference frame to world reference frame

    :param R: rotation matrix (2x2)
    :param t: translation vector (2x1)
    :param p_local: x, y position in local reference frame
    :return: p_world: x, y position in world reference frame
    """

    p_world = np.dot(R, p_local) + t

    return p_world


def rigid_transform_world_to_local(R, t, p_world):
    """
    Apply rigid body transform from world reference frame to local reference frame

    :param R: rotation matrix (2x2)
    :param t: translation vector (2x1)
    :param p_world: x, y position in world reference frame
    :return: p_local: x, y position in local reference frame
    """

    p_local = np.dot(R.T, (p_world - t))

    return p_local


def rect_to_polar(p_rect):
    """
    Convert (x, y) position vector in rectangular coordinates to (rho, psi) polar coordinate vector

    :param p_rect: (x, y) position vector in rectangular coordinates
    :return: (rho, psi) polar coordinate vector [m], [rad]
    """

    x, y = p_rect
    rho = np.sqrt(x**2 + y**2)
    psi = np.arctan2(y, x)

    return np.array([rho, psi])


def polar_to_rect(p_polar):
    """
    Convert (rho, psi) polar coordinate vector to (x, y) position vector in rectangular coordinates

    :param p_polar: rho, psi) polar coordinate vector
    :return: p_rect: (x, y) position vector in rectangular coordinates
    """

    rho, psi = p_polar
    x = rho * np.cos(psi)
    y = rho * np.sin(psi)

    return np.array([x, y])


def angle_to_rotation_matrix(angle_rad):
    """
    Construct 2x2 rotation matrix from angle

    :param angle_rad: angle [radians]
    :return: rotation matrix
    """

    return np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                     [np.sin(angle_rad), np.cos(angle_rad)]])
