from numpy.testing import assert_array_equal
import numpy as np
from pytest import mark

from python.lib.robot import move


# @mark.skip(reason="Getting other tests working first")
def test_robot_move_forward_from_zero_position_when_applying_forward_control_input_without_noise():
    x, y, alpha = 0., 0, 0
    r = np.array([x, y, alpha])
    u = np.array([1, 0.])
    n = np.array([0, 0.])

    r_new = move(r, u, n)

    assert assert_array_equal(r_new, [1, 0, 0.]) is None


def test_robot_move_forward_from_non_zero_position_when_applying_forward_control_input_without_noise():
    x, y, alpha = 1., 3., 0.
    r = np.array([x, y, alpha])
    u = np.array([1, 0.])
    n = np.array([0, 0.])

    r_new = move(r, u, n)

    assert assert_array_equal(r_new, [2., 3., 0.]) is None


def test_robot_move_right_from_non_zero_position_when_applying_right_control_input_without_noise():
    x, y, alpha = 1., 3., 0.
    r = np.array([x, y, alpha])
    u = np.array([1, -90.])
    n = np.array([0, 0.])

    r_new = move(r, u, n)

    assert assert_array_equal(r_new, [1., 2., -90.]) is None


def test_robot_move_left_from_non_zero_position_when_applying_left_control_input_without_noise():
    x, y, alpha = 1., 3., 0.
    r = np.array([x, y, alpha])
    u = np.array([1, 90.])
    n = np.array([0, 0.])

    r_new = move(r, u, n)

    assert assert_array_equal(r_new, [1., 4., 90.]) is None


def test_robot_move_forward_from_non_zero_position_and_45_deg_angle_when_applying_forward_control_input_without_noise():
    x, y, alpha = 1., 3., 45.
    r = np.array([x, y, alpha])
    u = np.array([1, 0.])
    n = np.array([0, 0.])

    r_new = move(r, u, n)

    assert assert_array_equal(r_new, [1.+1./np.sqrt(2), 3.+1./np.sqrt(2), 45.]) is None


def test_robot_move_right_from_non_zero_position_and_45_deg_angle_when_applying_forward_control_input_without_noise():
    x, y, alpha = 1., 3., 45.
    r = np.array([x, y, alpha])
    u = np.array([1, -90.])
    n = np.array([0, 0.])

    r_new = move(r, u, n)

    assert assert_array_equal(r_new, [1.+1./np.sqrt(2), 3.-1./np.sqrt(2), -45.]) is None
