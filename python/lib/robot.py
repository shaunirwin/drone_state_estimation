# robot motion

from python.lib.transforms import rigid_transform_local_to_world, angle_to_rotation_matrix


def move(r, u, n):
    """
    Move robot using control input and perturbation

    :param r: current robot pose in world ref frame [x; y; alpha]
    :param u: control signal in local ref frame [d_x; d_y, d_alpha]
    :param n: perturbation to control input in local ref frame [d_x; d_y; d_alpha]
    :return: new robot pose in world ref frame [x; y; alpha]
    """

    # current pose in world ref frame
    x, y, alpha = r

    # convert control input position + noise to world reference frame
    d_x_local = u[0] + n[0]
    d_y_local = u[1] + n[1]
    d_alpha = u[2] + n[2]
    d_x_world, d_y_world = rigid_transform_local_to_world(angle_to_rotation_matrix(alpha),
                                                          [x, y],
                                                          [d_x_local, d_y_local])

    r_new = [x + d_x_world, y + d_y_world, alpha + d_alpha]

    return r_new
