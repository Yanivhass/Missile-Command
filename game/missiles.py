"""Enemy missiles."""

import random

import cv2
import numpy as np

from config import CONFIG
from utils import get_cv2_xy

# class EnemyMissiles():
#
#     """Go from current step to next one.
#
#            The missile launched is done in the main environment class.
#
#            Args:
#                action (int): (0) do nothing, (1) fire missile,
#                (2) target up, (3) target down, (4), target left, (5) target right,
#                (6) target_left_up, (7) target_left_down, (8) target_right_up, (9) target_right_down
#
#            Returns:
#                observation: None.
#
#                reward: None.
#
#                done: None.
#
#                info (dict): additional information of the current time step. It
#                    contains key "can_fire" with associated value "True" if the
#                    anti-missile battery can fire a missile and "False" otherwise.
#            """
#
#     def __init__(self, number = CONFIG.ENEMY_MISSILES.NUM_OF_BATTERIES,
#                  pose = np.array([CONFIG.ENNEMY_BATTERY.INIT_POS_RANGE[0]*CONFIG.WIDTH, 0, CONFIG.ENNEMY_BATTERY.SPEED, CONFIG.ENNEMY_BATTERY.LAUNCH_THETA]),
#                  health = CONFIG.ENEMY_MISSILES.MAX_HEALTH):
#         # self.pose = np.tile(pose, (number, 1))
#         self.pose = pose
#         self.health = health*np.ones(number)
#         self.launch = np.zeros(number, dtype=bool)
#         self.enemy_tar = np.zeros(number, dtype=bool)
#         self.enemy_atc = np.zeros(number, dtype=bool)
#         self.targets = {
#             'missiles': np.zeros((number, 2), dtype=int)
#             # 'fr_bats': np.zeros((number), dtype=int),
#             # 'fr_missiles': np.zeros((number), dtype=int)
#         }
#                     # a value can be in the range 0: CONFIG.FRIENDLY_BATTERY.NUMBER*CONFIG.FRIENDLY_MISSILES.NUMBER
#
#
#
#     def reset(self, pose = np.array([CONFIG.ENNEMY_BATTERY.INIT_POS_RANGE[0]*CONFIG.WIDTH, 0, CONFIG.ENNEMY_BATTERY.SPEED, CONFIG.ENNEMY_BATTERY.LAUNCH_THETA]),
#                  health = CONFIG.ENEMY_MISSILES.MAX_HEALTH):
#         self.pose[:] = pose
#         self.health[:] = health
#         self.launch[:] = False
#         self.enemy_tar[:] = False
#         self.enemy_atc[:] = False
#         # self.targets['fr_bats'][:] = 0
#         self.targets['missiles'][:] = 0
#
#
#     def _compute_velocity(self, target):
#         """Launch a new missile.
#
#         - 0) Generate initial and final positions.
#         - 1) Compute speed vectors.
#         - 2) Add the new missile.
#         """
#         # Generate initial and final positions
#         # ------------------------------------------
#
#         # Initial n Final position
#         x0, y0, x1, y1 = self.pose[0], self.pose[1], target[0], target[1]
#
#         # Compute speed vectors
#         # ------------------------------------------
#         # Compute norm
#         norm = np.sqrt(np.square(x1 - x0) + np.square(y1 - y0))
#
#         # Compute unit vectors
#         ux = (x1 - x0) / norm
#         uy = (y1 - y0) / norm
#
#         # Compute speed vectors
#         vx = CONFIG.ENEMY_MISSILES.SPEED * ux
#         vy = CONFIG.ENEMY_MISSILES.SPEED * uy
#
#         # Add the new missile
#         # ------------------------------------------
#
#         # Create the missile
#         missile_pose = np.array(
#             [[x0, y0, x0, y0, x1, y1, vx, vy]],
#             dtype=CONFIG.DTYPE,
#         )
#
#         # Add it to the others
#         self.enemy_missiles = np.vstack(
#             (self.enemy_missiles, new_missile))
#
#         # Increase number of launched missiles
#         self.nb_missiles_launched += 1
#
#     def step(self, action, target):
#
#         self.vel = self.vel_map[action]
#         self.pos = 0
#
#
# class FriendlyMissiles():
#
#     def __init__(self, number = CONFIG.FRIENDLY_MISSILES.NUM_OF_BATTERIES, pose=np.array([0, CONFIG.FRIENDLY_BATTERY.INIT_HEIGHT_RANGE[0],
#                                                                                           CONFIG.FRIENDLY_BATTERY.SPEED, CONFIG.FRIENDLY_BATTERY.LAUNCH_THETA]), health=CONFIG.FRIENDLY_MISSILES.MAX_HEALTH):
#         # self.pose = np.tile(pose, (number, 1))
#         self.pose = pose
#         self.health = health * np.ones(number)
#         self.launch = np.zeros(number, dtype=bool)
#         self.enemy_tar = np.zeros(number, dtype=bool)
#         self.enemy_atc = np.zeros(number, dtype=bool)
#         self.targets = {
#             'bats': np.zeros((number), dtype=int),            ######## enemy bats is inferrior to city types ...........
#             'missiles': np.zeros((number, 2), dtype=int)
#             # 'city_bats': np.zeros((number), dtype=int),
#             # 'city_missiles': np.zeros((number), dtype=int)
#         }
#
#     def reset(self, pose=np.array([0, CONFIG.FRIENDLY_BATTERY.INIT_HEIGHT_RANGE[0],
#                                 CONFIG.FRIENDLY_BATTERY.SPEED, CONFIG.FRIENDLY_BATTERY.LAUNCH_THETA]), health=CONFIG.FRIENDLY_MISSILES.MAX_HEALTH):
#         self.pose[:] = pose
#         self.health[:] = health
#         self.launch[:] = False
#         self.enemy_tar[:] = False
#         self.enemy_atc[:] = False
#         self.targets['bats'][:] = 0
#         self.targets['missiles'][:] = 0
#         # self.targets['city_bats'][:] = 0
#         # self.targets['city_missiles'][:] = 0
#
#     def step(self, action, target):
#
#         # self.pose += np.array(self.vel_map[action])
#         # self.vel = np.array(self.vel_map[action])
#         action = 0
#
#         return action


class Missile():

    def __init__(self, pose, health):
        self.pose = pose  # [x,y,z,heading angle in degrees]
        self.health = health
        self.launched = False
        self.target = []
        self.target_xy = np.zeros(2, 1)
        self.guided = False


    def reset(self, pose, health):
        self.pose[:] = pose
        self.health[:] = health
        self.launched = False
        # self.enemy_tar[:] = False
        # self.enemy_atc[:] = False
        # self.targets['bats'][:] = 0
        # self.targets['missiles'][:] = 0
        # self.targets['city_bats'][:] = 0
        # self.targets['city_missiles'][:] = 0

    def step(self):
        """
        The missile steers itself towards it's target
        UNNECESSARY, UPDATED IN FATHER CLASSS "ENTITY"
        """

        if self.guided:
            # self.target_xy =
            relative_distance = self.pose[0:2]-self.target_xy
            target_angle = np.arctan2(relative_distance[0], relative_distance[1]) * 180/ np.pi  # angle to target in degrees
            angle_delta = target_angle - self.pose[3]

        heading_angle = self.pose[3] * np.pi / 180.0  # heading in radians
        delta_pose = self.pose[2] * np.array([np.cos(heading_angle), np.sin(heading_angle)])
        self.pose[0:2] += delta_pose



class Cities():

    def __init__(self, number = CONFIG.ENNEMY_CITIES.NUM_OF_BATTERIES, pose=np.array([0, CONFIG.ENNEMY_CITIES.INIT_HEIGHT_RANGE[0],
                                                                                      CONFIG.ENNEMY_CITIES.SPEED, CONFIG.ENNEMY_CITIES.LAUNCH_THETA]), launch = 0, health=CONFIG.ENNEMY_CITIES.MAX_HEALTH):
        # self.pose = np.tile(pose, (number, 1))
        self.pose = pose
        self.health = health * np.ones(number)
        self.launch = launch*np.ones(number, dtype=bool)
        self.enemy_tar = np.zeros(number, dtype=bool)
        self.enemy_atc = np.zeros(number, dtype=bool)

        self.attackers = {
            'fr_missiles': np.zeros((number, 2), dtype=int)
            # 'fr_bats': np.zeros((number), dtype=int),
            # 'fr_missiles': np.zeros((number), dtype=int)
        }

    def reset(self, pose=np.array([0, CONFIG.ENNEMY_CITIES.INIT_HEIGHT_RANGE[0],
                                CONFIG.ENNEMY_CITIES.SPEED, CONFIG.ENNEMY_CITIES.LAUNCH_THETA]), health=CONFIG.ENNEMY_CITIES.MAX_HEALTH):
        self.pose[:] = pose
        self.health[:] = health
        self.launch[:] = 1
        self.enemy_tar[:] = False
        self.enemy_atc[:] = False
        # self.targets[:] = 0
        # self.attacker['fr_bats'][:] = 0
        self.attacker['fr_missiles'][:] = 0

    def step(self, action, target):

        # self.pose += np.array(self.vel_map[action])
        # self.vel = np.array(self.vel_map[action])
        action = 0

        return action


# class EnemyMissiles():
#     """Enemy missiles class.
#
#     Enemy missiles are created by the environment.
#     """
#
#     def __init__(self, battery_pos):
#         """Initialize missiles.
#
#         Attributes:
#             enemy_missiles (numpy array): of size (N, 8)  with N the number of
#                 enemy missiles present in the environment. The features are:
#                 (0) initial x position, (1) initial y position, (2) current x
#                 position, (3) current y position, (4) final x position, (5)
#                 final y position, (6) horizontal speed vx and (7) vertical
#                 speed vy.
#             nb_missiles_launched (int): the number of enemy missiles launched
#                 in the environment.
#         """
#         self.pos_x = battery_pos[0]
#         self.pos_y = battery_pos[1]
#
#     def _launch_missile(self):
#         """Launch a new missile.
#
#         - 0) Generate initial and final positions.
#         - 1) Compute speed vectors.
#         - 2) Add the new missile.
#         """
#         # Generate initial and final positions
#         # ------------------------------------------
#
#         # Initial position
#         x0 = self.pos_x
#         y0 = self.pos_y
#
#         # Final position
#         x1 = random.uniform(-0.5 * CONFIG.WIDTH, 0.5 * CONFIG.WIDTH)
#         y1 = 0.0
#
#         # Compute speed vectors
#         # ------------------------------------------
#
#         # Compute norm
#         norm = np.sqrt(np.square(x1 - x0) + np.square(y1 - y0))
#
#         # Compute unit vectors
#         ux = (x1 - x0) / norm
#         uy = (y1 - y0) / norm
#
#         # Compute speed vectors
#         vx = CONFIG.ENEMY_MISSILES.SPEED * ux
#         vy = CONFIG.ENEMY_MISSILES.SPEED * uy
#
#         # Add the new missile
#         # ------------------------------------------
#
#         # Create the missile
#         new_missile = np.array(
#             [[x0, y0, x0, y0, x1, y1, vx, vy]],
#             dtype=CONFIG.DTYPE,
#         )
#
#         # Add it to the others
#         self.enemy_missiles = np.vstack(
#             (self.enemy_missiles, new_missile))
#
#         # Increase number of launched missiles
#         self.nb_missiles_launched += 1
#
#     def reset(self):
#         """Reset enemy missiles.
#
#         Warning:
#             To fully initialize a EnemyMissiles object, init function and reset
#             function must be called.
#         """
#         # self.enemy_missiles = np.zeros((0, 8), dtype=CONFIG.DTYPE)
#         # self.nb_missiles_launched = 0
#
#     def step(self, action):
#         """Go from current step to next one.
#
#         - 0) Moving missiles.
#         - 1) Potentially launch a new missile.
#         - 2) Remove missiles that hit the ground.
#
#         Collisions with friendly missiles and / or cities are checked in the
#         main environment class.
#
#         Notes:
#             From one step to another, a missile could exceed its final
#             position, so we need to do some verification. This issue is due to
#             the discrete nature of environment, decomposed in time steps.
#
#         Args:
#             action (int): (0) do nothing, (1) target up, (2) target down, (3)
#                 target left, (4) target right, (5) fire missile.
#
#         returns:
#             observation: None.
#
#             reward: None.
#
#             done (bool): True if the episode is finished, i.d. there are no
#                 more enemy missiles in the environment and no more enemy
#                 missiles to be launch. False otherwise.
#
#             info: None.
#         """
#         # Moving missiles
#         # ------------------------------------------
#
#         # Compute horizontal and vertical distances to targets
#         dx = np.abs(self.enemy_missiles[:, 4] - self.enemy_missiles[:, 2])
#         dy = np.abs(self.enemy_missiles[:, 5] - self.enemy_missiles[:, 3])
#
#         # Take the minimum between the actual speed and the distance to target
#         movement_x = np.minimum(np.abs(self.enemy_missiles[:, 6]), dx)
#         movement_y = np.minimum(np.abs(self.enemy_missiles[:, 7]), dy)
#
#         # Keep the right sign
#         movement_x *= np.sign(self.enemy_missiles[:, 6])
#         movement_y *= np.sign(self.enemy_missiles[:, 7])
#
#         # Step t to step t+1
#         self.enemy_missiles[:, 2] += movement_x
#         self.enemy_missiles[:, 3] += movement_y
#
#         # Potentially launch a new missile
#         # ------------------------------------------
#
#         if self.nb_missiles_launched < CONFIG.ENEMY_MISSILES.NUMBER:
#             if random.random() <= CONFIG.ENEMY_MISSILES.PROBA_IN:
#                 self._launch_missile()
#
#         # Remove missiles that hit the ground
#         # ------------------------------------------
#
#         missiles_out_indices = np.squeeze(np.argwhere(
#             (self.enemy_missiles[:, 2] == self.enemy_missiles[:, 4]) &
#             (self.enemy_missiles[:, 3] == self.enemy_missiles[:, 5])
#         ))
#
#         self.enemy_missiles = np.delete(
#             self.enemy_missiles, missiles_out_indices, axis=0)
#
#         done = self.enemy_missiles.shape[0] == 0 and \
#             self.nb_missiles_launched == CONFIG.ENEMY_MISSILES.NUMBER
#         return None, None, done, None
#
#     def render(self, observation):
#         """Render enemy missiles.
#
#         For each enemy missiles, draw a line of its trajectory and the actual
#         missile.
#
#         Args:
#             observation (numpy.array): the current environment observation
#                 representing the pixels. See the object description in the main
#                 environment class for information.
#         """
#         for x0, y0, x, y in zip(self.enemy_missiles[:, 0],
#                                 self.enemy_missiles[:, 1],
#                                 self.enemy_missiles[:, 2],
#                                 self.enemy_missiles[:, 3]):
#             cv2.line(
#                 img=observation,
#                 pt1=(get_cv2_xy(x0, y0)),
#                 pt2=(get_cv2_xy(x, y)),
#                 color=CONFIG.COLORS.ENEMY_MISSILE,
#                 thickness=1,
#             )
#
#             cv2.circle(
#                 img=observation,
#                 center=(get_cv2_xy(x, y)),
#                 radius=int(CONFIG.ENEMY_MISSILES.RADIUS),
#                 color=CONFIG.COLORS.ENEMY_MISSILE,
#                 thickness=-1,
#             )
