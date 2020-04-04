import numpy as np
from numpy.testing import assert_array_almost_equal

from python.lib.ekf import ekf_covariance_propagation


def test_ekf_covariance_propagation():
    F_x = np.eye(3, 3)
    P = np.eye(3, 3)
    F_n = np.eye(3, 3)
    N = np.eye(3, 3)

    assert assert_array_almost_equal(ekf_covariance_propagation(F_x, P, F_n, N), 2 * np.eye(3, 3)) is None
