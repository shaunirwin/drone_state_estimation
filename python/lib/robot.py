# robot motion

import os
if os.environ.get("USE_JAX", False):
    import jax.numpy as np
    from jax.config import config
    # check that JAX is set to 64-bit precision
    assert config.values["jax_enable_x64"]
else:
    import numpy as np

from python.lib.transforms import rigid_transform_local_to_world, angle_to_rotation_matrix


def move(r, u, n):
    """
    Move robot using control input and perturbation

    :param r: current robot pose in world ref frame [x; y; alpha]
    :param u: control signal in local ref frame [d_x; d_alpha]
    :param n: perturbation to control input in local ref frame [d_x; d_y; d_alpha]
    :return: new robot pose in world ref frame [x; y; alpha]
    """

    # current pose in world ref frame
    x, y, alpha = r

    # control signal in local ref frame
    d_x_local_u, d_alpha_local_u = u
    d_x_local_n, d_alpha_local_n = n
    d_x_local = d_x_local_u + d_x_local_n
    d_p_local = np.array([d_x_local, 0])
    d_alpha_local = d_alpha_local_u + d_alpha_local_n

    # # convert control input position + noise to world reference frame
    # d_x_local = u[0] + n[0]
    # d_y_local = u[1] + n[1]
    # d_alpha = u[2] + n[2]

    alpha_new = alpha + d_alpha_local
    R_new = angle_to_rotation_matrix(alpha_new)

    # move to new position
    d_x_world, d_y_world = np.dot(R_new, d_p_local)

    # d_x_world, d_y_world = rigid_transform_local_to_world(angle_to_rotation_matrix(alpha + d_alpha_local),
    #                                                       [d_x_local, 0],
    #                                                       [0, 0])

    r_new = np.array([x + d_x_world, y + d_y_world, alpha_new])
    # r_new = [d_x_world, d_y_world, alpha + d_alpha]

    return r_new
