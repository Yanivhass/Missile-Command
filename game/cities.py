"""Cities."""

import sys

import cv2
import numpy as np

from config import CONFIG
from utils import get_cv2_xy


class EnemyCities():
    """Cities class.

    Attributes:
        MAX_HEALTH (float): value corresponding to the max health of a city.
            Each time an enemy missile destroys a city, it loses 1 point of
            heath. At 0, the city is completely destroyed.
    """
    MAX_HEALTH = 1.0

    def __init__(self, number, pose):
        """Initialize cities.

        # Attributes:
        #     cities (numpy array): of size (N, 3) with N the number of cities.
        #         The features are: (0) x position, (1) y position and (2)
        #         integrity level (0 if destroyed else 1).
        # """

        self.pose = pose
        self.health = health*np.ones((number))



    def get_remaining_cities(self):
        """Compute healthy cities number.

        Returns:opencv draw multiple circles
            nb_remaining_cities (int): the number of remaining cities.
        """
        # return np.sum(self.cities[:, 2] == self.MAX_HEALTH)
        return self.health

    def reset(self, pose = (0, CONFIG.WIDTH//2), health = CONFIG.ENNEMY_CITIES.MAX_HEALTH):
        """Reset cities.

        Integrity is reset to 1 for all cities.

        Warning:
            To fully initialize a Cities object, init function and reset
            function musts be called.
        """
        self.pose = pose
        self.health[:] = health


    def step(self, action, observation):
        """Go from current step to next one.

        Destructions by enemy missiles are checked in the main environment
        class.

        Args:
            action (int): (0) do nothing, (1) target up, (2) target down, (3)
                target left, (4) target right, (5) fire missile.

        returns:
            observation: None.

            reward: None.

            done (bool): True if the episode is finished, i.d. all cities are
                destroyed. False otherwise.

            info: None.
        """

        res_cities = np.where(observation['enemy_cities']['health'] != 0)[0].tolist()
        done = True if len(res_cities) == 0 else False

        return None, None, done, res_cities

    def render(self, observation):
        """Render cities.

        Todo:
            Include the integrity level.

        Args:
            observation (numpy.array): the current environment observation
                representing the pixels. See the object description in the main
                environment class for information.
        """
        for x, y, integrity in zip(self.cities[:, 0],
                                   self.cities[:, 1],
                                   self.cities[:, 2]):
            if integrity > 0:
                cv2.circle(
                    img=observation,
                    center=(get_cv2_xy(x, y)),
                    radius=int(CONFIG.CITIES.RADIUS),
                    color=CONFIG.COLORS.CITY,
                    thickness=-1,
                )
