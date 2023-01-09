"""Useful functions."""

import functools

import numpy as np

from config import CONFIG


def get_cv2_xy(x, y):
    """Transform x environment position into x opencv position.

    The origin of the environment is the anti-missiles battery, placed in the
    bottom center. But in the render method, the origin is in the top left
    corner. It is also good to note that in python-opencv, coordinates are
    written (y, x) and not (x, y) like for the environment.

    Args:
        x (float): x environment coordinate.
        y (float): y environment coordinate.

    Returns:
        y (int): x opencv coordinate.

        x (int): x opencv coordinate.
    """
    return int(CONFIG.HEIGHT - y), int(x + (CONFIG.WIDTH / 2))


def rgetattr(obj, attr, *args):
    """Recursive getattr function."""

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))


def rsetattr(obj, attr, val):
    """Recursive setattr function."""
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def angle_diff(a, b):
    """
    Turn angle in degrees from a to b
    """
    a = np.mod(a, 360)
    b = np.mod(b, 360)
    d = a - b
    if np.abs(360 - np.abs(d)) < np.abs(d):
        d = np.abs(d) - 360
    return d


def pov_transform(frame, poses):
    """
    Transform poses to the given frame
    poses and are given as np array in global frame with columns
     [x, y, heading(degrees)]
    """
    poses[0:2, :] = poses[0:2, :] - frame[0:2]
    poses[2, :] = np.mod(poses[2, :] - frame[0:2], 360)
    return poses
