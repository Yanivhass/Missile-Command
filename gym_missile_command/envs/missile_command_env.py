"""Main environment class."""
import contextlib
import sys
import cv2
import numpy as np
# import gym
import utils
# from gym import spaces
# from gym.utils import seeding
# from gym.spaces.utils import flatten_space, flatten, unflatten
from PIL import Image

from config import CONFIG

# Import Pygame and remove welcome message
with contextlib.redirect_stdout(None):
    import pygame

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from gymnasium.spaces.utils import flatten_space, flatten, unflatten
from gymnasium.spaces import Discrete, Box
import os
import random

import ray
from ray import air, tune
from ray.rllib.env.env_context import EnvContext


def out_of_bounds(points):
    below = points[:, 1] < 0
    above = points[:, 1] > CONFIG.HEIGHT
    right = points[:, 0] > CONFIG.WIDTH
    left = points[:, 0] < 0
    return below | above | right | left


def get_missiles_to_launch(missiles_list, launching_unit_id):
    """

    Args:
        missiles_list:
        launching_unit_id:

    Returns:
        missiles_to_launch:
    """
    ready_missiles = np.argwhere(missiles_list[:, 7] == 0)
    ready_missiles = missiles_list[ready_missiles, :]
    # _, idx = np.unique(ready_missiles[:, 1], return_index=True)
    # ready_missiles = ready_missiles[idx, :]
    missiles_to_launch = ready_missiles[np.isin(ready_missiles[:, 8], launching_unit_id)]
    missiles_to_launch = missiles_to_launch[:, 0]  # id of missiles to launch
    return missiles_to_launch


def get_movement(velocity, angles, action):
    """
    Args:
        velocity: movement speed
        angles: current heading angle of entity, measured as angle from positive x-axis. degrees.
        action: int, movement direction [0 = left, 1 = straight, 2 = right]

    Returns:
        delta: position change during current timestep
        velocity:
        angles:
    """
    angles[np.argwhere(action == 0)] = angles[np.argwhere(action == 0)] + 10  # turn left 10 degrees
    angles[np.argwhere(action == 2)] = angles[np.argwhere(action == 2)] - 10  # turn right 10 degrees
    angles = np.mod(angles, 360)
    delta_pose = (velocity * np.array([np.cos(np.deg2rad(angles)), np.sin(np.deg2rad(angles))])).transpose()
    return [delta_pose, velocity, angles]


class MissileCommandEnv(gym.Env):
    """Missile Command Gym environment.

    Attributes:
        ENEMY_BATTERIES (int): number of enemy batteries
        ENEMY_MISSILES (int): number of enemy missiles - each can be in one of the NB_ACTIONS states
        FRIENDLY_BATTERIES (int): number of firiendly batteries
        FRIENDLY_MISSILES (int): number of friendly missiles - each can be in one of the NB_ACTIONS states
        NB_ACTIONS (int): the 10 possible missiles actions. (0) do nothing, (1) target
            up, (2) target down, (3) target left, (4) target right, (5) fire
            missile, (6) target left up, (7) target left down, (8) target right up, (9) target right down

        metadata (dict): OpenAI Gym dictionary with the "render.modes" key.
    """
    # En_NB_ACTIONS = CONFIG.ENEMY_MISSILES.NB_ACTIONS
    # Fr_NB_ACTIONS = CONFIG.FRIENDLY_MISSILES.NB_ACTIONS
    metadata = {"render_modes": ["human", "rgb_array"],
                'video.frames_per_second': CONFIG.FPS}

    def __init__(self, env_config: EnvContext):
        """Initialize MissileCommand environment.

        Args:
            custom_config (dict): optional, custom configuration dictionary
                with configuration attributes (strings) as keys (for example
                "FRIENDLY_MISSILES.NUMBER") and values as... Well, values (for
                example 42).

        Attributes:
            action_space (gym.spaces.discrete.Discrete): OpenAI Gym action
                space.
            enemy_batteries (Batteries): Batteries game object.
            enemy_missiles (EnemyMissiles): EnemyMissiles game object - Derived from EnemyBatteries.
            enemy_cities (Cities): Cities game object.

            friendly_batteries (FriendlyBatteries): Batteries game object.
            friendly_missiles (FriendlyMissiles): FriendlyMissiles game object - - Derived from FriendlyBatteries.

            observation (numpy.array): of size (CONFIG.WIDTH, CONFIG.HEIGHT,
                3). The observation of the current timestep, representing the
                position and velocity of enemy missiles.

            observation_space (gym.spaces.Box): OpenAI Gym observation space.

            reward_timestep (float): reward of the current timestep.
            reward_total (float): reward sum from first timestep to current
                one.
            timestep (int): current timestep, starts from 0.

            clock (pygame.time.Clock): Pygame clock.
            display (pygame.Surface): pygame surface, only for the render
                method.
        """
        super(MissileCommandEnv, self).__init__()
        self.terrain = None
        self.city_sprite = None
        self.missile_sprite = None
        self.bomb_sprite = None
        self.battery_sprite = None
        self.airplane_sprite = None
        attackers_count = CONFIG.ATTACKERS.QUANTITY
        attackers_missile_count = CONFIG.ATTACKERS.QUANTITY * CONFIG.ATTACKERS.MISSILES_PER_UNIT
        defenders_count = CONFIG.DEFENDERS.QUANTITY
        defenders_missile_count = CONFIG.DEFENDERS.QUANTITY * CONFIG.DEFENDERS.MISSILES_PER_UNIT
        cities_count = CONFIG.CITIES.QUANTITY
        # columns are [0-id, 1-x, 2-y, 3-velocity, 4-direction, 5-health, 6-target_id, 7-num of missiles, 8-fuel]
        self.attackers = np.zeros((attackers_count, 9))
        # columns are [0-id, 1-x, 2-y, 3-velocity, 4-direction, 5-health, 6-target_id, 7-launched, 8-parent_id, 9-fuel]
        self.attackers_missiles = np.zeros((attackers_missile_count, 10))
        # columns are [0-id, 1-x, 2-y, 3-velocity, 4-direction, 5-health, 6-target_id, 7-num of missiles]
        self.defenders = np.zeros((defenders_count, 8))
        # columns are [0-id, 1-x, 2-y, 3-velocity, 4-direction, 5-health, 6-target_id, 7-launched, 8-parent_id, 9-fuel]
        self.defenders_missiles = np.zeros((defenders_missile_count, 10))
        # columns are [id, x, y, health]
        self.cities = np.zeros((cities_count, 4))

        '''
        Action space for the game
        '''
        num_of_targets = defenders_count + cities_count
        attacker_targets = np.ones((1, attackers_missile_count)) * num_of_targets
        # attacker_targets[:,1] = num_of_targets
        num_of_targets = attackers_count + attackers_missile_count
        defender_targets = np.ones((1, defenders_missile_count)) * num_of_targets
        # defender_targets[:, 1] = num_of_targets
        self.action_dictionary = spaces.Dict(
            # Actions, currently only for attackers
            {'attackers': spaces.Dict(
                {
                    # movement [0 = left, 1 = straight, 2 = right]
                    'movement': spaces.MultiDiscrete([3] * attackers_count),
                    'target': spaces.MultiDiscrete([defenders_count + cities_count] * attackers_count),
                    'fire': spaces.MultiBinary([attackers_count])

                }),
                'defenders': spaces.Dict(
                    {
                        # movement [0 = left, 1 = straight, 2 = right]
                        'movement': spaces.MultiDiscrete([3] * defenders_count),
                        'target': spaces.MultiDiscrete([attackers_count] * defenders_count),
                        'fire': spaces.MultiBinary([defenders_count])
                    })
            }
        )
        self.action_space = spaces.MultiDiscrete([3, (defenders_count + cities_count), 2])
        # self.action_space = flatten_space(self.action_dictionary)
        # pose_boxmin = pose_boxmax = np.zeros((1, 4))
        pose_boxmin = np.array([0, 0, 0, 0])
        pose_boxmax = np.array([CONFIG.WIDTH, CONFIG.HEIGHT, 100, 360])

        # self.observation_dictionary = \
        self.observation_space = \
            spaces.Dict(
                {
                    # state-space for the defending team batteries
                    'defenders': spaces.Dict({
                        'pose': spaces.Box(
                            np.tile(pose_boxmin[0:2], (defenders_count, 1)),
                            np.tile(pose_boxmax[0:2], (defenders_count, 1)),
                            shape=(defenders_count, 2)),
                        'velocity': spaces.Box(
                            np.tile(pose_boxmin[2], (defenders_count, 1)),
                            np.tile(pose_boxmax[2], (defenders_count, 1)),
                            shape=(defenders_count, 1)),
                        'direction': spaces.Box(
                            np.tile(pose_boxmin[3], (defenders_count, 1)),
                            np.tile(pose_boxmax[3], (defenders_count, 1)),
                            shape=(defenders_count, 1)),
                        # pose holds the position of each battery
                        'health': spaces.Box(0, 1, shape=(defenders_count, 1)),
                        # health - the hp each battery has
                        'missiles': spaces.Dict({
                            # 0 - Ready, 1 - Launched
                            'launched': spaces.MultiBinary(defenders_missile_count),
                            # Each missile's target is the entity number
                            'target': spaces.MultiDiscrete(
                                np.ones((1, defenders_missile_count)) * (attackers_count + attackers_missile_count)),
                            # pose is [x,y,velocity,heading angle]
                            'pose': spaces.Box(
                                np.tile(pose_boxmin[0:2], (defenders_missile_count, 1)),
                                np.tile(pose_boxmax[0:2], (defenders_missile_count, 1)),
                                shape=(defenders_missile_count, 2)),
                            'velocity': spaces.Box(
                                np.tile(pose_boxmin[2], (defenders_missile_count, 1)),
                                np.tile(pose_boxmax[2], (defenders_missile_count, 1)),
                                shape=(defenders_missile_count, 1)),
                            'direction': spaces.Box(
                                np.tile(pose_boxmin[3], (defenders_missile_count, 1)),
                                np.tile(pose_boxmax[3], (defenders_missile_count, 1)),
                                shape=(defenders_missile_count, 1)),
                            # Missiles health is binary
                            'health': spaces.MultiBinary(defenders_missile_count),
                            'fuel': spaces.Box(low=0, high=np.inf, shape=(defenders_missile_count, 1))
                        }),
                    }),
                    # cities are immobile and have no action
                    'cities': spaces.Dict({
                        'pose': spaces.Box(
                            np.tile(pose_boxmin[0:2], (cities_count, 1)),
                            np.tile(pose_boxmax[0:2], (cities_count, 1)),
                            shape=(cities_count, 2)),
                        'health': spaces.Box(0, 1, shape=(cities_count, 1)),
                    }),

                    # state-space for the attacking team bombers
                    'attackers': spaces.Dict({
                        'pose': spaces.Box(
                            np.tile(pose_boxmin[0:2], (attackers_count, 1)),
                            np.tile(pose_boxmax[0:2], (attackers_count, 1)),
                            shape=(attackers_count, 2)),
                        'velocity': spaces.Box(
                            np.tile(pose_boxmin[2], (attackers_count, 1)),
                            np.tile(pose_boxmax[2], (attackers_count, 1)),
                            shape=(attackers_count, 1)),
                        'direction': spaces.Box(
                            np.tile(pose_boxmin[3], (attackers_count, 1)),
                            np.tile(pose_boxmax[3], (attackers_count, 1)),
                            shape=(attackers_count, 1)),
                        'health': spaces.Box(0, 1, shape=(attackers_count, 1)),
                        'fuel': spaces.Box(low=0, high=np.inf, shape=(attackers_count, 1)),
                        'missiles': spaces.Dict({
                            # 0 - Ready, 1 - Launched
                            'launched': spaces.MultiBinary(attackers_missile_count),
                            # Each missile's target is the entity number
                            'target': spaces.MultiDiscrete(
                                np.ones((1, attackers_missile_count)) * (defenders_count + cities_count)),
                            # pose is [x,y,z,heading angle]
                            'pose': spaces.Box(
                                np.tile(pose_boxmin[0:2], (attackers_missile_count, 1)),
                                np.tile(pose_boxmax[0:2], (attackers_missile_count, 1)),
                                shape=(attackers_missile_count, 2)),
                            'velocity': spaces.Box(
                                np.tile(pose_boxmin[2], (attackers_missile_count, 1)),
                                np.tile(pose_boxmax[2], (attackers_missile_count, 1)),
                                shape=(attackers_missile_count, 1)),
                            'direction': spaces.Box(
                                np.tile(pose_boxmin[3], (attackers_missile_count, 1)),
                                np.tile(pose_boxmax[3], (attackers_missile_count, 1)),
                                shape=(attackers_missile_count, 1)),
                            # Missiles health is binary
                            'health': spaces.MultiBinary(attackers_missile_count),
                            'fuel': spaces.Box(low=0, high=np.inf, shape=(attackers_missile_count, 1))
                        })
                    })

                })
        self.observation_dictionary = self.observation_space
        # self.map = spaces.Box(0, 255, shape=(CONFIG.WIDTH, CONFIG.HEIGHT))
        # self.observation_space = flatten_space(self.observation_dictionary)
        # self.observation = self.observation_dictionary.sample()
        # # self.observation_vec = flatten(self.observation)
        # self.observation_vec = flatten(self.observation_dictionary, self.observation)
        # Set the seed for reproducibility
        self.seed(CONFIG.SEED)
        # Initialize the state
        self.reset()

        # Initializing objects
        # ------------------------------------------
        # No display while no render
        self.clock = None
        self.display = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _compute_sensor_observation(self, sensor_type):
        """
        Compute observation. Current game graphics.

        """
        if self.terrain is None:
            self.terrain = cv2.imread("../sprites/terrain.png", cv2.IMREAD_UNCHANGED)
            self.terrain[:,:,0:3] = cv2.cvtColor(self.terrain[:,:,0:3], cv2.COLOR_RGB2BGR)
            if self.terrain is None:
                raise Exception("Failed to load sprite at " + "../sprites/terrain.png")
            dim = (CONFIG.WIDTH, CONFIG.HEIGHT)
            self.terrain = cv2.resize(self.terrain, dim, interpolation=cv2.INTER_AREA)  # resize image
        if self.airplane_sprite is None:
            self.airplane_sprite = cv2.imread("../sprites/airplane.png", cv2.IMREAD_UNCHANGED)
            self.airplane_sprite[:,:,0:3] = cv2.cvtColor(self.airplane_sprite[:,:,0:3], cv2.COLOR_RGB2BGR)
            if self.airplane_sprite is None:
                raise Exception("Failed to load sprite at " + "../sprites/airplane.png")
            dim = (CONFIG.ATTACKERS.WIDTH, CONFIG.ATTACKERS.HEIGHT)
            self.airplane_sprite = cv2.resize(self.airplane_sprite, dim, interpolation=cv2.INTER_AREA)  # resize image
        if self.missile_sprite is None:
            self.missile_sprite = cv2.imread("../sprites/missile.png", cv2.IMREAD_UNCHANGED)
            self.missile_sprite[:,:,0:3] = cv2.cvtColor(self.missile_sprite[:,:,0:3], cv2.COLOR_RGB2BGR)
            if self.missile_sprite is None:
                raise Exception("Failed to load sprite at " + "../sprites/missile.png")
            dim = (CONFIG.DEFENDERS.MISSILES.WIDTH, CONFIG.DEFENDERS.MISSILES.HEIGHT)
            self.missile_sprite = cv2.resize(self.missile_sprite, dim, interpolation=cv2.INTER_AREA)  # resize image
        if self.bomb_sprite is None:
            self.bomb_sprite = cv2.imread("../sprites/bomb.png", cv2.IMREAD_UNCHANGED)
            self.bomb_sprite[:,:,0:3] = cv2.cvtColor(self.bomb_sprite[:,:,0:3], cv2.COLOR_RGB2BGR)
            if self.bomb_sprite is None:
                raise Exception("Failed to load sprite at " + "../sprites/bomb.png")
            dim = (CONFIG.ATTACKERS.MISSILES.WIDTH, CONFIG.ATTACKERS.MISSILES.HEIGHT)
            self.bomb_sprite = cv2.resize(self.bomb_sprite, dim, interpolation=cv2.INTER_AREA)  # resize image
        if self.battery_sprite is None:
            self.battery_sprite = cv2.imread("../sprites/battery.png", cv2.IMREAD_UNCHANGED)
            self.battery_sprite[:,:,0:3] = cv2.cvtColor(self.battery_sprite[:,:,0:3], cv2.COLOR_RGB2BGR)
            if self.battery_sprite is None:
                raise Exception("Failed to load sprite at " + "../sprites/battery.png")
            dim = (CONFIG.DEFENDERS.WIDTH, CONFIG.DEFENDERS.HEIGHT)
            self.battery_sprite = cv2.resize(self.battery_sprite, dim, interpolation=cv2.INTER_AREA)  # resize image
        if self.city_sprite is None:
            self.city_sprite = cv2.imread("../sprites/city.png", cv2.IMREAD_UNCHANGED)
            self.city_sprite[:,:,0:3] = cv2.cvtColor(self.city_sprite[:,:,0:3], cv2.COLOR_RGB2BGR)
            if self.battery_sprite is None:
                raise Exception("Failed to load sprite at " + "../sprites/battery.png")
            dim = (CONFIG.CITIES.WIDTH, CONFIG.CITIES.HEIGHT)
            self.city_sprite = cv2.resize(self.city_sprite, dim, interpolation=cv2.INTER_AREA)  # resize image
    # self.airplane_dim = (CONFIG.ATTACKERS.WIDTH, CONFIG.ATTACKERS.HEIGHT)
    # self.airplane = cv2.resize(self.airplane, self.airplane_dim, interpolation=cv2.INTER_AREA)  # resize image
    # self.alpha_airplane = self.airplane[:, :, 3] / 255.0
    # self.alpha_back = 1.0 - self.alpha_airplane
    # self.airplane_width = self.airplane.shape[0]
    # self.airplane_height = self.airplane.shape[1]

    # Reset observation


        if sensor_type == 'vision':
            # im = self.observation['sensors'][sensor_type].astype('uint8')
            # im = np.zeros((CONFIG.WIDTH, CONFIG.HEIGHT, 3), dtype=np.uint8)
            # im[:, :, 0] = CONFIG.COLORS.BACKGROUND[0]
            # im[:, :, 1] = CONFIG.COLORS.BACKGROUND[1]
            # im[:, :, 2] = CONFIG.COLORS.BACKGROUND[2]
            im = np.array(self.terrain[:,:,0:3])

            filled = -1
            pose = np.zeros((3,))
            for i in range(self.attackers.shape[0]):
                if self.attackers[i, 5] > 0:
                    # cv2.circle(im, self.attackers[i, 1:3].astype(int),
                    #            CONFIG.ATTACKERS.RADIUS, CONFIG.ATTACKERS.COLOR, filled)
                    im = utils.draw_sprite(image=im, sprite=self.airplane_sprite, pose=self.attackers[i, [1, 2, 4]])
                    cv2.circle(im, self.attackers[i, 1:3].astype(int),
                               CONFIG.ATTACKERS.RANGE, CONFIG.ATTACKERS.COLOR, 1)
            for i in range(self.attackers_missiles.shape[0]):
                if (self.attackers_missiles[i, 5] > 0) & (self.attackers_missiles[i, 7] > 0):
                    im = utils.draw_sprite(image=im, sprite=self.bomb_sprite, pose=self.attackers_missiles[i, [1, 2, 4]])
                    # cv2.circle(im, self.attackers_missiles[i, 1:3].astype(int),
                    #            CONFIG.ATTACKERS.MISSILES.RADIUS, CONFIG.ATTACKERS.MISSILES.COLOR, filled)
            for i in range(self.defenders.shape[0]):
                if self.defenders[i, 5] > 0:
                    pose[0:2] = self.defenders[i, [1, 2]]
                    pose[2] = 0
                    im = utils.draw_sprite(image=im, sprite=self.battery_sprite, pose=pose)
                    # cv2.circle(im, self.defenders[i, 1:3].astype(int),
                    #            CONFIG.DEFENDERS.RADIUS, CONFIG.DEFENDERS.COLOR, filled)
                    cv2.circle(im, self.defenders[i, 1:3].astype(int),
                               CONFIG.DEFENDERS.RANGE, CONFIG.DEFENDERS.COLOR, 1)
            for i in range(self.defenders_missiles.shape[0]):
                if (self.defenders_missiles[i, 5] > 0) & (self.defenders_missiles[i, 7] > 0):
                    im = utils.draw_sprite(image=im, sprite=self.missile_sprite, pose=self.defenders_missiles[i, [1, 2, 4]])
                    # cv2.circle(im, self.defenders_missiles[i, 1:3].astype(int),
                    #            CONFIG.DEFENDERS.MISSILES.RADIUS, CONFIG.DEFENDERS.MISSILES.COLOR, filled)
            for i in range(self.cities.shape[0]):
                if self.cities[i, 3] > 0:
                    pose[0:2] = self.cities[i, [1, 2]]
                    pose[2] = 0
                    im = utils.draw_sprite(image=im, sprite=self.city_sprite,
                                           pose=pose)
                    # cv2.circle(im, self.cities[i, 1:3].astype(int),
                    #            CONFIG.CITIES.RADIUS, CONFIG.CITIES.COLOR, filled)

            return im

    def _process_observation(self):
        """Process observation.

        This function could be implemented into the agent model, but for
        commodity this environment can do it directly.

        The interpolation mode INTER_AREA seems to give the best results. With
        other methods, every objects could not be seen at all timesteps.

        Returns:
            processed_observation (numpy.array): of size
                (CONFIG.OBSERVATION.HEIGHT, CONFIG.OBSERVATION.WIDTH, 3), the
                resized (or not) observation.
        """
        processed_observation = cv2.resize(
            self.map,
            (CONFIG.SCREEN_WIDTH, CONFIG.SCREEN_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )
        return processed_observation.astype(CONFIG.IMAGE_DTYPE)


    def state_to_dict(self):
        obs = self.observation_dictionary.sample()
        obs['attackers']['pose'] = self.attackers[:, 1:3]
        obs['attackers']['velocity'] = self.attackers[:, 3]
        obs['attackers']['direction'] = self.attackers[:, 4]
        obs['attackers']['health'] = self.attackers[:, 5]
        obs['attackers']['fuel'] = self.attackers[:, 8]
        obs['attackers']['missiles']['pose'] = self.attackers_missiles[:, 1:3]
        obs['attackers']['missiles']['velocity'] = self.attackers_missiles[:, 3]
        obs['attackers']['missiles']['direction'] = self.attackers_missiles[:, 4]
        obs['attackers']['missiles']['health'] = self.attackers_missiles[:, 5].astype(int)
        obs['attackers']['missiles']['target'] = self.attackers_missiles[:, 6].astype(int)
        obs['attackers']['missiles']['launched'] = self.attackers_missiles[:, 7].astype(int)
        obs['attackers']['missiles']['fuel'] = self.attackers_missiles[:, 9]

        obs['defenders']['pose'] = self.defenders[:, 1:3]
        obs['defenders']['velocity'] = self.defenders[:, 3]
        obs['defenders']['direction'] = self.defenders[:, 4]
        obs['defenders']['health'] = self.defenders[:, 5]
        # obs['defenders']['fuel'] = self.defenders[:, 8]
        obs['defenders']['missiles']['pose'] = self.defenders_missiles[:, 1:3]
        obs['defenders']['missiles']['velocity'] = self.defenders_missiles[:, 3]
        obs['defenders']['missiles']['direction'] = self.defenders_missiles[:, 4]
        obs['defenders']['missiles']['health'] = self.defenders_missiles[:, 5].astype(int)
        obs['defenders']['missiles']['target'] = self.defenders_missiles[:, 6].astype(int)
        obs['defenders']['missiles']['launched'] = self.defenders_missiles[:, 7].astype(int)
        obs['defenders']['missiles']['fuel'] = self.defenders_missiles[:, 9]

        obs['cities']['pose'] = self.cities[:, 1:3]
        obs['cities']['health'] = self.cities[:, 3]
        self.observation = flatten(self.observation_dictionary, obs)
        return self.observation

    def action_to_dict(self,action):
        obs = self.action_dictionary.sample()
        obs['attackers']['movement'] = self.attackers[:, 1:3]
        obs['attackers']['target'] = self.attackers[:, 3]
        obs['attackers']['target'] = self.attackers[:, 3]


        obs['defenders']['pose'] = self.defenders[:, 1:3]
        obs['defenders']['velocity'] = self.defenders[:, 3]
        obs['defenders']['direction'] = self.defenders[:, 4]

        obs['cities']['health'] = self.cities[:, 3]
        self.observation = flatten(self.observation_dictionary, obs)
        return self.observation


    def reset(self, seed=None,options={}):
        """Reset the environment.

        Returns:
            observation (numpy.array): the processed observation.
        """
        # Reset timestep and rewards
        self.timestep = 0
        self.reward_total = 0.0
        self.reward_timestep = 0.0

        # Defender's units
        bat_pose_x = np.random.uniform(CONFIG.DEFENDERS.INIT_POS_RANGE[0] * CONFIG.WIDTH,
                                       CONFIG.DEFENDERS.INIT_POS_RANGE[1] * CONFIG.WIDTH,
                                       size=CONFIG.DEFENDERS.QUANTITY)
        bat_pose_y = np.random.uniform(CONFIG.DEFENDERS.INIT_HEIGHT_RANGE[0] * CONFIG.WIDTH,
                                       CONFIG.DEFENDERS.INIT_HEIGHT_RANGE[1] * CONFIG.WIDTH,
                                       size=CONFIG.DEFENDERS.QUANTITY)
        # defender team units
        # columns are [0-id, 1-x, 2-y, 3-velocity, 4-direction, 5-health, 6-target_id, 7-num of missiles 8-fuel]
        id_offset = 0
        self.defenders[:, 0] = np.arange(CONFIG.DEFENDERS.QUANTITY)
        self.defenders[:, 1] = bat_pose_x
        self.defenders[:, 2] = bat_pose_y
        self.defenders[:, 3] = CONFIG.DEFENDERS.SPEED * CONFIG.SPEED_MODIFIER
        self.defenders[:, 4] = CONFIG.DEFENDERS.LAUNCH_THETA
        self.defenders[:, 5] = 1
        self.defenders[:, 7] = CONFIG.DEFENDERS.MISSILES_PER_UNIT
        id_offset += CONFIG.DEFENDERS.QUANTITY

        # defender cities
        init_height = np.array(CONFIG.CITIES.INIT_HEIGHT_RANGE) * CONFIG.HEIGHT
        init_pose = np.array(CONFIG.CITIES.INIT_POS_RANGE) * CONFIG.WIDTH
        city_pose_x = np.random.uniform(init_pose[0], init_pose[1], CONFIG.CITIES.QUANTITY)
        city_pose_y = np.random.uniform(init_height[0], init_height[1], CONFIG.CITIES.QUANTITY)

        # columns are [id, x, y, health]
        self.cities[:, 0] = np.arange(CONFIG.CITIES.QUANTITY) + id_offset
        self.cities[:, 1] = city_pose_x
        self.cities[:, 2] = city_pose_y
        self.cities[:, 3] = 1
        id_offset += CONFIG.CITIES.QUANTITY

        # defender missiles

        missiles_id = np.arange(CONFIG.DEFENDERS.QUANTITY * CONFIG.DEFENDERS.MISSILES_PER_UNIT) + id_offset
        self.defenders_missiles = np.zeros((CONFIG.DEFENDERS.QUANTITY, 10))
        # columns are columns are [0-id, 1-x, 2-y, 3-velocity, 4-direction, 5-health, 6-target_id, 7-launched, 8-parent_id 9-fuel]
        self.defenders_missiles[:, 1] = bat_pose_x
        self.defenders_missiles[:, 2] = bat_pose_y
        self.defenders_missiles[:, 3] = CONFIG.DEFENDERS.SPEED * CONFIG.SPEED_MODIFIER
        self.defenders_missiles[:, 4] = CONFIG.DEFENDERS.LAUNCH_THETA
        self.defenders_missiles[:, 5] = 1
        self.defenders_missiles[:, 6] = 0
        self.defenders_missiles[:, 7] = 0
        self.defenders_missiles[:, 8] = np.arange(CONFIG.DEFENDERS.QUANTITY)  # parent id
        self.defenders_missiles[:, 9] = CONFIG.DEFENDERS.MISSILES.FUEL
        self.defenders_missiles = np.tile(self.defenders_missiles, (CONFIG.DEFENDERS.MISSILES_PER_UNIT, 1))
        self.defenders_missiles[:, 0] = missiles_id

        # Attacker's units
        init_height = np.array(CONFIG.ATTACKERS.INIT_HEIGHT_RANGE) * CONFIG.HEIGHT
        init_pose = np.array(CONFIG.ATTACKERS.INIT_POS_RANGE) * CONFIG.WIDTH

        bat_pose_x = np.random.uniform(init_pose[0], init_pose[1], CONFIG.ATTACKERS.QUANTITY)
        bat_pose_y = np.random.uniform(init_height[0], init_height[1], CONFIG.ATTACKERS.QUANTITY)

        # columns are [0-id, 1-x, 2-y, 3-velocity, 4-direction, 5-health, 6-target_id, 7-num of missiles 8-fuel]
        id_offset = 0
        self.attackers[:, 0] = np.arange(CONFIG.ATTACKERS.QUANTITY)
        self.attackers[:, 1] = bat_pose_x
        self.attackers[:, 2] = bat_pose_y
        self.attackers[:, 3] = CONFIG.ATTACKERS.SPEED * CONFIG.SPEED_MODIFIER
        self.attackers[:, 4] = CONFIG.ATTACKERS.LAUNCH_THETA
        self.attackers[:, 5] = 1
        self.attackers[:, 7] = CONFIG.ATTACKERS.MISSILES_PER_UNIT
        self.attackers[:, 8] = CONFIG.ATTACKERS.FUEL
        id_offset += CONFIG.ATTACKERS.QUANTITY

        missiles_id = np.arange(CONFIG.ATTACKERS.QUANTITY * CONFIG.ATTACKERS.MISSILES_PER_UNIT) + id_offset
        # columns are [0-id, 1-x, 2-y, 3-velocity, 4-direction, 5-health, 6-target_id, 7-launched, 8-parent_id 9-fuel]
        self.attackers_missiles = np.zeros((CONFIG.ATTACKERS.QUANTITY, 10))
        self.attackers_missiles[:, 1] = bat_pose_x
        self.attackers_missiles[:, 2] = bat_pose_y
        self.attackers_missiles[:, 3] = CONFIG.ATTACKERS.SPEED * CONFIG.SPEED_MODIFIER
        self.attackers_missiles[:, 4] = CONFIG.ATTACKERS.LAUNCH_THETA
        self.attackers_missiles[:, 5] = 1
        self.attackers_missiles[:, 7] = 0  # launched
        self.attackers_missiles[:, 8] = np.arange(CONFIG.ATTACKERS.QUANTITY)  # parent id
        self.attackers_missiles[:, 9] = CONFIG.ATTACKERS.MISSILES.FUEL
        self.attackers_missiles = np.tile(self.attackers_missiles, (CONFIG.ATTACKERS.MISSILES_PER_UNIT, 1))
        self.attackers_missiles[:, 0] = missiles_id

        return self.state_to_dict()


    def step(self, action):
        """
        Go from current step to next one. Missile command step includes:
        1. Update movements according to given actions
        2. Check for collisions
        3. Update entities according to collisions
        4. Calculate step reward


        Args:


        Returns:
            observation (numpy.array): the processed observation.

            reward (float): reward of the current time step.

            done (bool): True if the episode is finished, False otherwise.

            info (dict): additional information on the current time step.
        """
        action = self.action_dictionary.sample() ######### DEBUG
        # Reset current reward
        # ------------------------------------------
        self.reward_timestep = 0.0
        reward_battery_destroyed = 0
        reward_city_destroyed = 0
        reward_bomber_destroyed = 0
        reward_missiles_launched = 0

        # Step functions: update state
        # ------------------------------------------
        # all entities [id, x, y, health]
        all_defenders = np.vstack(
            (self.defenders[:, [0, 1, 2, 5]], self.cities, self.defenders_missiles[:, [0, 1, 2, 5]]))
        all_attackers = np.vstack((self.attackers[:, [0, 1, 2, 5]], self.attackers_missiles[:, [0, 1, 2, 5]]))

        # Launch missiles
        attackers_range = CONFIG.ATTACKERS.RANGE
        defenders_range = CONFIG.DEFENDERS.RANGE

        # attackers

        # launch attacker missiles
        # chose one ready missile from each living attacker and
        # launch for parent who chose to fire
        attackers_firing = np.argwhere(action['attackers']['fire'])
        alive = self.attackers[attackers_firing, 5] > 0
        attackers_firing = attackers_firing[alive]  # live attackers that chose to fire
        unlaunched = self.attackers_missiles[:, 7] == 0
        # choose the first unlaunched missile
        _, missiles_to_launch = np.unique(self.attackers_missiles[unlaunched, 8].astype(int), return_index=True)
        missiles_to_launch = self.attackers_missiles[unlaunched, :][missiles_to_launch, 0]  # missile id
        parents = self.attackers_missiles[np.isin(self.attackers_missiles[:, 0], missiles_to_launch), 8]
        # parents = parents[missiles_to_launch]
        # missiles with living parents who chose to fire
        missiles_to_launch = missiles_to_launch[np.isin(parents, attackers_firing)]  # parents who chose to fire
        # convert back to binary array of missile
        missiles_to_launch = np.isin(self.attackers_missiles[:, 0], missiles_to_launch)
        parents = self.attackers_missiles[missiles_to_launch, 8].astype(int)
        target_id = action['attackers']['target']
        target_id = target_id[parents]

        # confirm target is within range and alive
        in_range = \
            np.linalg.norm(all_defenders[target_id, 1:3] - self.attackers_missiles[missiles_to_launch, 1:3],
                           axis=-1) <= attackers_range
        alive = all_defenders[target_id, 3] > 0
        missiles_to_launch[missiles_to_launch] = in_range & alive
        target_id = target_id[in_range & alive]
        if np.any(missiles_to_launch):
            self.attackers_missiles[missiles_to_launch, 7] = True  # set missiles to launched
            self.attackers_missiles[missiles_to_launch, 6] = target_id  # set target
            y = all_defenders[target_id, 2] - self.attackers_missiles[missiles_to_launch, 2]
            x = all_defenders[target_id, 1] - self.attackers_missiles[missiles_to_launch, 1]
            direction = np.mod(np.rad2deg(np.arctan2(y, x)), 360)  # set direction
            self.attackers_missiles[missiles_to_launch, 4] = direction
            self.attackers_missiles[missiles_to_launch, 3] = \
                CONFIG.ATTACKERS.MISSILES.SPEED * CONFIG.SPEED_MODIFIER
            reward_missiles_launched += np.sum(missiles_to_launch) * CONFIG.REWARD.MISSILE_LAUNCHED
        # defenders
        # chose one ready missile from each living attacker and
        # launch for parent who chose to fire
        defenders_firing = np.argwhere(action['defenders']['fire'])
        alive = self.defenders[defenders_firing, 5] > 0
        defenders_firing = defenders_firing[alive]  # live attackers that chose to fire
        unlaunched = self.defenders_missiles[:, 7] == 0
        # choose the first unlaunched missile
        _, missiles_to_launch = np.unique(self.defenders_missiles[unlaunched, 8].astype(int), return_index=True)
        missiles_to_launch = self.defenders_missiles[unlaunched, :][missiles_to_launch, 0]  # missile id
        parents = self.defenders_missiles[np.isin(self.defenders_missiles[:, 0], missiles_to_launch), 8]
        # parents = parents[missiles_to_launch]
        # missiles with living parents who chose to fire
        missiles_to_launch = missiles_to_launch[np.isin(parents, defenders_firing)]  # parents who chose to fire
        # convert back to binary array of missile
        missiles_to_launch = np.isin(self.defenders_missiles[:, 0], missiles_to_launch)
        parents = self.defenders_missiles[missiles_to_launch, 8].astype(int)
        target_id = action['defenders']['target']
        target_id = target_id[parents]

        # confirm target is within range and alive
        in_range = \
            np.linalg.norm(all_attackers[target_id, 1:3] - self.defenders_missiles[missiles_to_launch, 1:3],
                           axis=-1) <= defenders_range
        alive = all_attackers[target_id, 3] > 0
        missiles_to_launch[missiles_to_launch] = in_range & alive
        target_id = target_id[in_range & alive]
        if np.any(missiles_to_launch):
            self.defenders_missiles[missiles_to_launch, 7] = True  # set missiles to launched
            self.defenders_missiles[missiles_to_launch, 6] = target_id  # set target
            y = all_attackers[target_id, 2] - self.defenders_missiles[missiles_to_launch, 2]
            x = all_attackers[target_id, 1] - self.defenders_missiles[missiles_to_launch, 1]
            direction = np.mod(np.rad2deg(np.arctan2(y, x)), 360)  # set direction
            self.defenders_missiles[missiles_to_launch, 4] = direction
            self.defenders_missiles[missiles_to_launch, 3] = \
                CONFIG.DEFENDERS.MISSILES.SPEED * CONFIG.SPEED_MODIFIER

        # Roll movements
        # attackers units
        [delta_pose, velocity, angles] = get_movement(self.attackers[:, 3], self.attackers[:, 4],
                                                      action['attackers']['movement'])
        self.attackers[:, 1:3] += delta_pose
        self.attackers[:, 4] = angles

        # terminate if out of bounds
        oob = out_of_bounds(self.attackers[:, 1:3])
        if np.any(oob):
            num_of_destroyed = self.destroy_units_by_id(side="attackers", unit_ids=self.attackers[oob, 0])
            reward_bomber_destroyed = num_of_destroyed * CONFIG.REWARD.DESTROYED_BOMBER

        # decrement fuel
        self.attackers[:, 8] -= 1
        launched = self.attackers_missiles[:, 7] == 1
        self.attackers_missiles[launched, 9] -= 1
        # self.defenders[:, 8] -= 1
        launched = self.defenders_missiles[:, 7] == 1
        self.defenders_missiles[launched, 9] -= 1

        # terminate if out of fuel
        oof = np.argwhere(self.attackers[:, 8] == 0)
        self.destroy_units_by_id(side="attackers", unit_ids=self.attackers[oof, 0])
        # self.attackers_missiles[np.argwhere(self.attackers_missiles[:, 9] == 0), [3, 5]] = 0
        # oof = np.argwhere(self.defenders[:, 8] == 0)
        # self.destroy_units_by_id(side="defenders", unit_ids=self.defenders[oof, 0])
        self.defenders_missiles[np.argwhere(self.defenders_missiles[:, 9] == 0), [3, 5]] = 0

        # defenders units
        [delta_pose, velocity, angles] = get_movement(self.defenders[:, 3], self.defenders[:, 4],
                                                      action['defenders']['movement'])
        self.defenders[:, 1:3] += delta_pose
        self.defenders[:, 4] = angles

        # terminate if out of bounds
        oob = out_of_bounds(self.defenders[:, 1:3])
        if np.any(oob):
            # self.defenders[oob, 1:3] -= delta_pose[oob]
            self.destroy_units_by_id(side="defenders", unit_ids=self.defenders[oob, 0])

        # attackers missiles - non launched
        attackers_non_launched = np.argwhere(np.logical_not(self.attackers_missiles[:, 7]))
        non_launched_parents = self.attackers_missiles[attackers_non_launched, 8].astype(int)
        self.attackers_missiles[attackers_non_launched, 1:5] = self.attackers[non_launched_parents, 1:5]

        # defenders missiles - non launched
        defenders_non_launched = np.argwhere(np.logical_not(self.defenders_missiles[:, 7]))
        non_launched_parents = self.defenders_missiles[defenders_non_launched, 8].astype(int)
        self.defenders_missiles[defenders_non_launched, 1:5] = self.defenders[non_launched_parents, 1:5]

        # attackers missiles - launched
        attackers_launched = np.argwhere((self.attackers_missiles[:, 7] == 1) & (self.attackers_missiles[:, 5] > 0))
        if attackers_launched.shape[0] > 0:
            angles = self.attackers_missiles[attackers_launched, 4] * np.pi / 180.0  # degrees to rads
            delta = (self.attackers_missiles[attackers_launched, 3] * np.array(
                [np.cos(angles), np.sin(angles)])).T
            self.attackers_missiles[attackers_launched, 1:3] += \
                delta.reshape(self.attackers_missiles[attackers_launched, 1:3].shape)
            target_id = self.attackers_missiles[attackers_launched, 6].astype(int)
            y = all_defenders[target_id, 2] - self.attackers_missiles[attackers_launched, 2]
            x = all_defenders[target_id, 1] - self.attackers_missiles[attackers_launched, 1]
            direction = np.mod(np.rad2deg(np.arctan2(y, x)), 360)  # set direction
            self.attackers_missiles[attackers_launched, 4] = direction
            oob = out_of_bounds(self.attackers_missiles[:, 1:3])
            if np.any(oob):
                # self.attackers_missiles[oob, 1:3] -= delta[oob,:].reshape(
                #     self.attackers_missiles[oob, 1:3].shape)
                self.attackers_missiles[oob, 3] = 0  # velocity,health = 0
                self.attackers_missiles[oob, 5] = 0

        # defenders missiles - launched
        defenders_launched = np.argwhere((self.defenders_missiles[:, 7] == 1) & (self.defenders_missiles[:, 5] > 0))
        if defenders_launched.shape[0] > 0:
            angles = self.defenders_missiles[defenders_launched, 4] * np.pi / 180.0  # degrees to rads
            delta = (self.defenders_missiles[defenders_launched, 3] * np.array(
                [np.cos(angles), np.sin(angles)])).T
            self.defenders_missiles[defenders_launched, 1:3] += \
                delta.reshape(self.defenders_missiles[defenders_launched, 1:3].shape)
            target_id = self.defenders_missiles[defenders_launched, 6].astype(int)
            y = all_attackers[target_id, 2] - self.defenders_missiles[defenders_launched, 2]
            x = all_attackers[target_id, 1] - self.defenders_missiles[defenders_launched, 1]
            direction = np.mod(np.rad2deg(np.arctan2(y, x)), 360)  # set direction
            self.defenders_missiles[defenders_launched, 4] = direction
            oob = out_of_bounds(self.defenders_missiles[:, 1:3])
            if np.any(oob):
                # self.defenders_missiles[oob, 1:3] -= delta[oob,:].reshape(self.defenders_missiles[oob, 1:3].shape)
                self.defenders_missiles[oob, 3] = 0  # velocity,health = 0
                self.defenders_missiles[oob, 5] = 0

        # Check for collisions
        # ------------------------------------------
        att_exp_rad = CONFIG.ATTACKERS.MISSILES.EXPLOSION_RADIUS
        def_exp_rad = CONFIG.DEFENDERS.MISSILES.EXPLOSION_RADIUS
        defenders_launched = np.any((self.defenders_missiles[:, 7] == 1) & (self.defenders_missiles[:, 5] > 0))
        attackers_launched = np.any((self.attackers_missiles[:, 7] == 1) & (self.attackers_missiles[:, 5] > 0))

        # get range from launched defender missiles to attackers and their launched missiles
        if np.any(defenders_launched):
            defenders_launched = np.argwhere((self.defenders_missiles[:, 7] == 1) & (self.defenders_missiles[:, 5] > 0))
            target_id = self.defenders_missiles[defenders_launched, 6].astype(int)
            targets_xy = all_attackers[target_id, 1:3]
            hits = np.linalg.norm(targets_xy - self.defenders_missiles[defenders_launched, 1:3], axis=-1) <= def_exp_rad
            if np.any(hits):
                defenders_hit = defenders_launched[hits]
                self.defenders_missiles[defenders_hit, 3] = 0
                self.defenders_missiles[defenders_hit, 5] = 0
                attackers_hit = np.isin(self.attackers[:, 0], target_id[hits])
                if np.any(attackers_hit):
                    num_of_destroyed = self.destroy_units_by_id(side="attackers", unit_ids=self.attackers[attackers_hit, 0])
                    reward_bomber_destroyed = num_of_destroyed * CONFIG.REWARD.DESTROYED_BOMBER
                attackers_hit = np.isin(self.attackers_missiles[:, 0], target_id[hits])
                if np.any(attackers_hit):
                    self.attackers_missiles[attackers_hit, [3, 5]] = 0  # velocity, health = 0

        # get range from launched attacker missiles to defender units and cities
        if np.any(attackers_launched):
            attackers_launched = np.argwhere((self.attackers_missiles[:, 7] == 1) & (self.attackers_missiles[:, 5] > 0))
            target_id = self.attackers_missiles[attackers_launched, 6].astype(int)
            targets_xy = all_defenders[target_id, 1:3]
            hits = np.linalg.norm(targets_xy - self.attackers_missiles[attackers_launched, 1:3], axis=-1) <= att_exp_rad
            if np.any(hits):
                attackers_hit = attackers_launched[hits]
                self.attackers_missiles[attackers_hit, 3] = 0
                self.attackers_missiles[attackers_hit, 5] = 0
                defenders_hit = np.isin(self.defenders[:, 0], target_id[hits])
                if np.any(defenders_hit):
                    self.destroy_units_by_id(side="defenders", unit_ids=self.defenders[defenders_hit, 0])
                    reward_battery_destroyed = np.sum(defenders_hit) * CONFIG.REWARD.DESTROYED_AA_BATTERY
                # Defenders missiles cannot be hit by attackers
                # defenders_hit = np.isin(self.defenders_missiles[:, 0], target_id[hits])
                # if np.any(defenders_hit):
                #     self.defenders_missiles[defenders_hit, [3, 5]] = 0  # velocity, health = 0
                defenders_hit = np.isin(self.cities[:, 0], target_id[hits])
                if np.any(defenders_hit):
                    self.cities[defenders_hit, 3] = 0  # health = 0
                    reward_city_destroyed = np.sum(defenders_hit) * CONFIG.REWARD.DESTROYED_CITY

        # Calculate rewards
        # ------------------------------------------
        cities_reward = 0

        # Check if episode is finished
        # ------------------------------------------
        done = np.all(self.attackers[:, 5] == 0) \
               | (np.all(self.defenders[:, 5] == 0) & np.all(self.cities[:, 3] == 0))

        # Render every objects
        # ------------------------------------------

        # self.map = \
        #     self._compute_sensor_observation('vision')

        # Return everything
        # ------------------------------------------

        self.timestep += 1
        # self.reward_total += friendly_battery_reward + enemy_battery_reward + cities_reward + self.reward_timestep
        self.reward_timestep = reward_battery_destroyed + \
                               reward_bomber_destroyed + \
                               reward_city_destroyed + \
                               reward_missiles_launched
        self.reward_total += self.reward_timestep
        self.observation = self.state_to_dict()
        return self.observation, self.reward_timestep, done, {}


    def render(self, mode="rgb_array"):
        """Render the environment.

        This function renders the environment observation. To check what the
        processed observation looks like, it can also renders it.

        Args:
            mode (str): the render mode. Possible values are "rgb_array" and
                "processed_observation".
        """
        frame = None
        if not self.display:
            pygame.display.init()
            # pygame.mouse.set_visible(False)
            self.clock = pygame.time.Clock()
            pygame.display.set_caption("MissileCommand")

        # Display the normal observation
        if mode == "rgb_array":
            self.map = \
                self._compute_sensor_observation('vision')
            self.display = pygame.display.set_mode(
                (CONFIG.SCREEN_WIDTH, CONFIG.SCREEN_HEIGHT))
            frame = cv2.resize(self.map, (CONFIG.SCREEN_WIDTH, CONFIG.SCREEN_HEIGHT),
                               interpolation=cv2.INTER_AREA,
                               )
            surface = pygame.surfarray.make_surface(frame)
            surface = pygame.transform.rotate(surface, 90)

        # Display the processed observation
        elif mode == "processed_observation":
            self.display = pygame.display.set_mode((
                CONFIG.OBSERVATION.RENDER_PROCESSED_HEIGHT,
                CONFIG.OBSERVATION.RENDER_PROCESSED_WIDTH,
            ))
            surface = pygame.surfarray.make_surface(
                self._process_observation())
            surface = pygame.transform.scale(
                surface,
                (CONFIG.OBSERVATION.RENDER_PROCESSED_HEIGHT,
                 CONFIG.OBSERVATION.RENDER_PROCESSED_WIDTH),
            )

        self.display.blit(surface, (0, 0))
        pygame.display.flip()  # update()

        # Limix max FPS
        self.clock.tick(CONFIG.FPS)
        frame = cv2.flip(frame, 0)
        return frame


    def close(self):
        """Close the environment."""
        if self.display:
            pygame.quit()


    def destroy_units_by_id(self, side, unit_ids):
        # destroy unit and unlaunched missiles
        # return number of destroyed units(that were not previously destroyed)
        num_of_destroyed = 0
        if side == "attackers":
            num_of_destroyed = np.sum(self.attackers[unit_ids.astype(int), 5] > 0)
            self.attackers[unit_ids.astype(int), 3] = 0  # velocity, health = 0
            self.attackers[unit_ids.astype(int), 5] = 0
            missiles_unlaunched = self.attackers_missiles[:, 7] == 0
            missile_ids = np.isin(self.attackers_missiles[missiles_unlaunched, 8], unit_ids).astype(int)
            self.attackers_missiles[missile_ids, 3] = 0  # velocity = 0
            self.attackers_missiles[missile_ids, 5] = 0  # health = 0
        elif side == "defenders":
            num_of_destroyed = np.sum(self.defenders[unit_ids.astype(int), 5] > 0)
            self.defenders[unit_ids.astype(int), 3] = 0  # velocity, health = 0
            self.defenders[unit_ids.astype(int), 5] = 0  # velocity, health = 0
            missiles_unlaunched = self.defenders_missiles[:, 7] == 0
            missile_ids = np.isin(self.defenders_missiles[missiles_unlaunched, 8], unit_ids).astype(int)
            self.defenders_missiles[missile_ids, 3] = 0  # velocity = 0
            self.defenders_missiles[missile_ids, 5] = 0  # health = 0
        else:
            print("Warning: destroy_units_by_id side should be \"attackers\" or \"defenders\"")
        return num_of_destroyed


if __name__ == "__main__":
    # Create the environment
    env = gym.make("gym_missile_command:missile-command-v0")

    # Reset it
    observation = env.reset()

    t = 1
