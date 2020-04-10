# import numpy as np
import os
if os.environ.get("USE_JAX", False):
    import jax.numpy as np
    from jax.config import config
    # check that JAX is set to 64-bit precision
    assert config.values["jax_enable_x64"]
else:
    import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal
import jax

from python.lib.sensors import RangeBearingSensor as rbs


def test_observe_range_bearing_when_robot_is_at_origin_and_landmark_is_directly_in_front():
    x_r = np.array([0, 0, 0])
    p_world_rect = np.array([4.5, 0])

    range_, bearing = rbs.observe_range_bearing(x_r, p_world_rect)

    assert range_ == 4.5
    assert bearing == 0.


def test_observe_range_bearing_when_robot_is_at_origin_and_landmark_is_on_right_side():
    x_r = np.array([0, 0, 0])
    p_world_rect = np.array([0., -2.])

    range_, bearing = rbs.observe_range_bearing(x_r, p_world_rect)

    assert range_ == 2.
    assert bearing == np.deg2rad(-90.)


def test_observe_range_bearing_when_robot_is_not_at_origin_and_landmark_is_on_right_side():
    x_r = np.array([1., 0., 0.])
    p_world_rect = np.array([1., -2.])

    range_, bearing = rbs.observe_range_bearing(x_r, p_world_rect)

    assert range_ == 2.
    assert bearing == np.deg2rad(-90.)


def test_inv_observe_range_bearing_when_robot_is_at_origin_and_landmark_is_directly_in_front():
    x_r = np.array([0, 0, 0])
    p_local_polar = np.array([4.5, np.deg2rad(0.)])

    x, y = rbs.inv_observe_range_bearing(x_r, p_local_polar)

    assert assert_almost_equal(x, 4.5) is None
    assert assert_almost_equal(y, 0.) is None


def test_inv_observe_range_bearing_when_robot_is_at_origin_and_landmark_is_on_right_side():
    x_r = np.array([0, 0, 0])
    p_local_polar = np.array([2., np.deg2rad(-90.)])

    x, y = rbs.inv_observe_range_bearing(x_r, p_local_polar)

    assert assert_almost_equal(x, 0.) is None
    assert assert_almost_equal(y, -2.) is None


def test_inv_observe_range_bearing_is_the_inverse_of_observe_range_bearing():
    x_r = np.array([0, 0, 0])
    p_world_rect = np.array([4.5, 56.8])

    range_, bearing = rbs.observe_range_bearing(x_r, p_world_rect)
    x, y = rbs.inv_observe_range_bearing(x_r, np.array([range_, bearing]))

    assert assert_almost_equal(x, 4.5) is None
    assert assert_almost_equal(y, 56.8) is None


def test_jacobian_H_X_r_at_specific_test_points_compared_to_auto_diff():
    X = np.array([23.5, -14.6, np.deg2rad(45.)])
    p_world_rect = np.array([39., 10.4])    # landmark position

    f = rbs.observe_range_bearing

    H_x = rbs.jacobian_H_X_r(x_r=X[0], y_r=X[1], alpha_r=X[2], l_i_x=p_world_rect[0], l_i_y=p_world_rect[1])

    J = jax.jacfwd(f, argnums=0)(X, p_world_rect)

    assert assert_array_almost_equal(J, H_x) is None


def test_jacobian_H_L_i_at_specific_test_points_compared_to_auto_diff():
    X = np.array([23.5, -14.6, np.deg2rad(45.)])
    p_world_rect = np.array([39., 10.4])    # landmark position

    f = rbs.observe_range_bearing

    H_l = rbs.jacobian_H_L_i(x_r=X[0], y_r=X[1], alpha_r=X[2], l_i_x=p_world_rect[0], l_i_y=p_world_rect[1])

    J = jax.jacfwd(f, argnums=1)(X, p_world_rect)

    assert assert_array_almost_equal(J, H_l) is None


def test_jacobian_G_X_r_at_specific_test_points_compared_to_auto_diff():
    X = np.array([23.5, -14.6, np.deg2rad(45.)])
    p_local_polar = np.array([11.6, .25])    # range-bearing measurement

    f = rbs.inv_observe_range_bearing

    G_x = rbs.jacobian_G_X_r(x_r=X[0], y_r=X[1], alpha_r=X[2], rho=p_local_polar[0], psi=p_local_polar[1])

    J = jax.jacfwd(f, argnums=0)(X, p_local_polar)

    assert assert_array_almost_equal(J, G_x) is None


def test_jacobian_G_y_i_at_specific_test_points_compared_to_auto_diff():
    X = np.array([23.5, -14.6, np.deg2rad(45.)])
    p_local_polar = np.array([11.6, .25])    # range-bearing measurement

    f = rbs.inv_observe_range_bearing

    G_x = rbs.jacobian_G_y_i(x_r=X[0], y_r=X[1], alpha_r=X[2], rho=p_local_polar[0], psi=p_local_polar[1])

    J = jax.jacfwd(f, argnums=1)(X, p_local_polar)

    assert assert_array_almost_equal(J, G_x) is None
