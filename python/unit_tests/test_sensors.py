# import numpy as np
import os
if os.environ.get("USE_JAX", False):
    import jax.numpy as np
    from jax.config import config
    # check that JAX is set to 64-bit precision
    assert config.values["jax_enable_x64"]
else:
    import numpy as np
from numpy.testing import assert_almost_equal

from python.lib.sensors import observe_range_bearing, inv_observe_range_bearing
from python.lib.transforms import angle_to_rotation_matrix


def test_observe_range_bearing_when_robot_is_at_origin_and_landmark_is_directly_in_front():
    alpha = 0.
    # R = angle_to_rotation_matrix(alpha)
    t = np.array([0, 0])
    p_world_rect = np.array([4.5, 0])

    range_, bearing = observe_range_bearing(alpha, t, p_world_rect)

    assert range_ == 4.5
    assert bearing == 0.


def test_observe_range_bearing_when_robot_is_at_origin_and_landmark_is_on_right_side():
    alpha = 0.
    # R = angle_to_rotation_matrix(alpha)
    t = np.array([0, 0])
    p_world_rect = np.array([0., -2.])

    range_, bearing = observe_range_bearing(alpha, t, p_world_rect)

    assert range_ == 2.
    assert bearing == np.deg2rad(-90.)


def test_observe_range_bearing_when_robot_is_not_at_origin_and_landmark_is_on_right_side():
    alpha = 0.
    # R = angle_to_rotation_matrix(alpha)
    t = np.array([1., 0])
    p_world_rect = np.array([1., -2.])

    range_, bearing = observe_range_bearing(alpha, t, p_world_rect)

    assert range_ == 2.
    assert bearing == np.deg2rad(-90.)


def test_inv_observe_range_bearing_when_robot_is_at_origin_and_landmark_is_directly_in_front():
    alpha = 0.
    # R = angle_to_rotation_matrix(alpha)
    t = np.array([0, 0])
    p_local_polar = np.array([4.5, np.deg2rad(0.)])

    x, y = inv_observe_range_bearing(alpha, t, p_local_polar)

    assert assert_almost_equal(x, 4.5) is None
    assert assert_almost_equal(y, 0.) is None


def test_inv_observe_range_bearing_when_robot_is_at_origin_and_landmark_is_on_right_side():
    alpha = 0.
    # R = angle_to_rotation_matrix(alpha)
    t = np.array([0, 0])
    p_local_polar = np.array([2., np.deg2rad(-90.)])

    x, y = inv_observe_range_bearing(alpha, t, p_local_polar)

    assert assert_almost_equal(x, 0.) is None
    assert assert_almost_equal(y, -2.) is None


def test_inv_observe_range_bearing_is_the_inverse_of_observe_range_bearing():
    alpha = 0.
    # R = angle_to_rotation_matrix(alpha)
    t = np.array([0, 0])
    p_world_rect = np.array([4.5, 56.8])

    range_, bearing = observe_range_bearing(alpha, t, p_world_rect)
    x, y = inv_observe_range_bearing(alpha, t, np.array([range_, bearing]))

    assert assert_almost_equal(x, 4.5) is None
    assert assert_almost_equal(y, 56.8) is None
