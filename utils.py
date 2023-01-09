"""Useful functions."""

import functools

import numpy as np
import cv2
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


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, -angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result



def draw_sprite(image, sprite, pose):
    """

    Args:
        image: np array, RGB matrix of image
        sprite: np array, RGB matrix of sprite smaller than image
        pose: np array, [x,y,azimuth] vector of entity to draw

    Returns:
        image: image after drawing sprite
    """
    if pose[0] < 0 \
            or pose[0] > image.shape[1] \
            or pose[1] < 0 \
            or pose[1] > image.shape[0]:
        return image

    sprite_width = sprite.shape[0]
    sprite_height = sprite.shape[1]

    x1 = (pose[0] - (sprite_width / 2)).astype(int)
    x2 = (pose[0] + (sprite_width / 2)).astype(int)
    y1 = (pose[1] - (sprite_height / 2)).astype(int)
    y2 = (pose[1] + (sprite_height / 2)).astype(int)
    offset_x1 = 0
    offset_x2 = sprite_width
    offset_y1 = 0
    offset_y2 = sprite_height
    sprite = rotate_image(sprite, pose[2])

    if x1 < 0:
        offset_x1 = abs(x1)
        x1 = 0
    if x2 > image.shape[1] - 1:
        offset_x2 -= x2 - (image.shape[0] - 1)
        x2 = image.shape[1] - 1
    if y1 < 0:
        offset_y1 = abs(y1)
        y1 = 0
    if y2 > image.shape[0] - 1:
        offset_y2 -= y2 - (image.shape[1] - 1)
        y2 = image.shape[0] - 1

    sprite = sprite[offset_y1:offset_y2, offset_x1:offset_x2, :]
    alpha_sprite = sprite[:, :, 3] / 255.0
    alpha_sprite = np.expand_dims(alpha_sprite, -1)
    sprite = alpha_sprite * sprite[:, :, 0:3]
    sprite += (1 - alpha_sprite) * image[y1:y2, x1:x2, :]
    image[y1:y2, x1:x2, :] = sprite

    return image
