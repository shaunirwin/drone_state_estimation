from python.lib import transforms


def observe_range_bearing(alpha, t, p_world_rect):
    """
    Simulate a range-bearing sensor reading of a landmark in world coordinates

    :param alpha: rotation angle robot (true)
    :param t: (x, y) rectangular coordinate position of robot (true)
    :param p_world_rect: (x, y) rectangular coordinate position of a landmark in world ref frame
    :return: range-bearing (polar coordinate) measurement of a landmark (local ref frame)
    """

    R = transforms.angle_to_rotation_matrix(alpha)
    return transforms.rect_to_polar(transforms.rigid_transform_world_to_local(R, t, p_world_rect))


def inv_observe_range_bearing(alpha, t, p_local_polar):
    """
    Simulate inverse range-bearing sensor reading of a landmark in world coordinates

    :param alpha: rotation angle of robot (true)
    :param t: (x, y) rectangular coordinate position of robot (true)
    :param p_local_polar: (range, bearing) polar coordinate position of a landmark in local ref frame
    :return: estimated position of landmark in rectangular coordinates (world ref frame)
    """

    R = transforms.angle_to_rotation_matrix(alpha)
    return transforms.rigid_transform_local_to_world(R, t, transforms.polar_to_rect(p_local_polar))
