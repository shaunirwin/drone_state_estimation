import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

from python.lib.ekf import EKFSLAM


def test_ekf_state_propagation_initialised_correctly():
    est = EKFSLAM()

    assert est.X.shape == (3,)
    assert est.P.shape == (3, 3)
    assert est.N.shape == (2, 2)


def test_ekf_state_propagation_forward_control_input_updates_state_vector_x_position():
    est = EKFSLAM()
    est.X[:3] = [0, 0, 0]
    est.P = np.zeros((3, 3)) * 1.
    est.N = np.array([[0.2, 0], [0, 0.3]])

    est.state_propagation([1, 0])

    assert assert_array_almost_equal(est.X, [1, 0, 0.]) is None

    est.state_propagation([1, 0])

    assert assert_array_almost_equal(est.X, [2, 0, 0.]) is None


def test_ekf_state_propagation_angled_control_input_updates_state_vector_x_and_y_position():
    est = EKFSLAM()
    est.X[:3] = [0, 0, 0.]
    est.P = np.zeros((3, 3)) * 1.
    est.N = np.array([[0.2, 0], [0, 0.3]])

    est.state_propagation([1, np.deg2rad(45.)])

    assert assert_array_almost_equal(est.X, [1./np.sqrt(2), 1./np.sqrt(2), np.deg2rad(45.)]) is None

    est.state_propagation([1, 0])

    assert assert_array_almost_equal(est.X, [2./np.sqrt(2), 2./np.sqrt(2), np.deg2rad(45.)]) is None


def test_ekf_state_propagation_state_covariance_matrix_remains_zero_if_zero_process_noise():
    est = EKFSLAM()
    est.X[:3] = [0, 0, 0]
    est.P = np.zeros((3, 3)) * 1.
    est.N = np.zeros((2, 2))

    est.state_propagation([1, 0])

    assert assert_array_equal(est.P, np.zeros((3, 3))) is None


def test_ekf_state_propagation_forward_control_input_updates_state_covariance():
    est = EKFSLAM()
    est.X[:3] = [0, 0, 0]
    est.P = np.zeros((3, 3)) * 1.
    est.N = np.array([[0.2, 0], [0, 0.3]])

    est.state_propagation([1, 0])

    # TODO: need to double check that these values are as expected

    assert assert_array_almost_equal(est.P, np.array([[0.2, 0, 0],
                                                      [0, 0.3, 0.3],
                                                      [0, 0.3, 0.3]])) is None

    est.state_propagation([1, 0])

    assert assert_array_almost_equal(est.P, np.array([[0.4, 0, 0],
                                                      [0, 1.5, 0.9],
                                                      [0, 0.9, 0.6]])) is None
