from numpy.testing import assert_array_equal
import numpy as np
from pytest import mark

from python.lib.robot import move


# @mark.skip(reason="Getting other tests working first")
def test_robot_move_forward_when_applying_forward_control_input_without_noise():
    x, y, alpha = 0., 0, 0
    r = np.array([x, y, alpha])
    u = np.array([1, 0, 0.])
    n = np.array([0, 0, 0.])

    r_new = move(r, u, n)

    assert assert_array_equal(r_new, [1, 0, 0.]) is None
