import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from python.lib.transforms import rigid_transform_local_to_world, angle_to_rotation_matrix


def test_rigid_transform_local_to_world_when_rotation_is_zero_and_position_is_zero():
    R = np.array([[1, 0], [0, 1.]])
    t = np.array([0, 0.])
    p_local = np.array([1., 2.])

    assert assert_array_equal(rigid_transform_local_to_world(R, t, p_local), np.array([1., 2.])) is None


def test_rigid_transform_local_to_world_when_rotation_is_zero_and_position_is_non_zero():
    R = angle_to_rotation_matrix(0)
    t = np.array([3.5, -4.])
    p_local = np.array([1., 2.])

    assert assert_array_equal(rigid_transform_local_to_world(R, t, p_local), np.array([4.5, -2.])) is None


def test_rigid_transform_local_to_world_when_rotation_is_180_deg_and_position_is_non_zero():
    R = angle_to_rotation_matrix(180)
    t = np.array([3.5, -4.])
    p_local = np.array([1., 2.])

    assert assert_array_equal(rigid_transform_local_to_world(R, t, p_local), np.array([2.5, -6.])) is None


def test_angle_to_rotation_matrix_if_angle_is_zero():
    assert assert_array_equal(angle_to_rotation_matrix(0), np.array([[1, 0], [0, 1.]])) is None


def test_angle_to_rotation_matrix_if_angle_is_180_deg():
    assert assert_array_almost_equal(angle_to_rotation_matrix(180), np.array([[-1, 0], [0, -1.]])) is None
