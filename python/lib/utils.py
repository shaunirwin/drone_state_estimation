import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


def confidence_ellipse(mean, cov, ax, n_std=3.0, facecolor='g', **kwargs):
    """
    Plot a confidence ellipse

    Modified from https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html

    :param mean: center of confidence ellipse
    :param cov: covariance matrix
    :param ax: The axes object to draw the ellipse into.
    :param n_std: number of standard deviations to plot confidence ellipse at
    :param facecolor:
    :param kwargs:
    :return: matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """

    if mean.shape != (2,):
        raise ValueError("Only works with 2D mean")

    if cov.shape != (2, 2):
        raise ValueError("Only works with 2D covariance matrix")

    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = mean[0]

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = mean[1]

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)

    return ax.add_patch(ellipse)
