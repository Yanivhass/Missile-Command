"""Main environment class."""
import contextlib
import sys
import cv2
import gym
import numpy as np
from gym import spaces
from PIL import Image

from config import CONFIG
from game.Entities import Unit, City, Missile
# from game.batteries import EnemyBattery, FriendlyBattery, CityBattery
from game.cities import EnemyCities
# from gym_missile_command.game.missile import EnemyMissiles, FriendlyMissiles
from game.target import Target
from utils import rgetattr, rsetattr

# Import Pygame and remove welcome message
with contextlib.redirect_stdout(None):
    import pygame


def get_missiles_to_launch(missiles_list, launching_unit_id):
    ready_missiles = np.argwhere(not missiles_list[:, 8])
    ready_missiles = missiles_list[ready_missiles, :]
    _, idx = np.unique(ready_missiles[:, 1], return_index=True)
    ready_missiles = ready_missiles[idx, :]
    missiles_to_launch = ready_missiles[np.isin(ready_missiles[:, 1], launching_unit_id)]
    missiles_to_launch = missiles_to_launch[:, 0]  # id of missiles to launch
    return missiles_to_launch

def get_movement(velocity, angles, action):
    angles[np.argwhere(action == -1)] = angles[np.argwhere(action == -1)] + 10 # turn left 10 degrees
    angles[np.argwhere(action == 1)] = angles[np.argwhere(action == 1)] - 10 # turn right 10 degrees
    angles = np.mod(angles,360)
    angles = angles * np.pi / 180.0  # degreess to rads
    delta = (velocity * np.array([np.cos(angles), np.sin(angles)])).transpose()
    return [delta, velocity, angles]

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
    En_NB_ACTIONS = CONFIG.ENEMY_MISSILES.NB_ACTIONS
    Fr_NB_ACTIONS = CONFIG.FRIENDLY_MISSILES.NB_ACTIONS
    metadata = {"render_modes": ["human", "rgb_array"],
                'video.frames_per_second': CONFIG.FPS}

    def __init__(self, custom_config={}):
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
        '''
        Action space for the game
        '''
        num_of_targets = CONFIG.DEFENDERS.QUANTITY + CONFIG.CITIES.QUANTITY
        attacker_targets = np.ones((1, CONFIG.ATTACKERS.MISSILES_PER_UNIT * CONFIG.ATTACKERS.QUANTITY)) * num_of_targets
        # attacker_targets[:,1] = num_of_targets
        num_of_targets = CONFIG.ATTACKERS.QUANTITY
        defender_targets = np.ones((1, CONFIG.DEFENDERS.MISSILES_PER_UNIT * CONFIG.DEFENDERS.QUANTITY)) * num_of_targets
        # defender_targets[:, 1] = num_of_targets
        self.action_space = spaces.Dict(
            # Actions, currently only for attackers
            {'attackers': spaces.Dict(
                {
                    # 'batteries': spaces.MultiBinary(CONFIG.ATTACKERS.QUANTITY),
                    # movement [-1 = left, 0 = straight, 1 = right]
                    'movement': spaces.MultiDiscrete([-1, 0, 1] * CONFIG.ATTACKERS.QUANTITY),
                    # Actions for the missiles:
                    'missiles': spaces.Dict(
                        {
                            # 'launch: 0 - None, 1 - launch,
                            'launch': spaces.MultiBinary(
                                CONFIG.ATTACKERS.MISSILES_PER_UNIT * CONFIG.ATTACKERS.QUANTITY),
                            # target ID each missile is currently aiming at (including unfired missiles)
                            'enemy_tar': spaces.MultiDiscrete(attacker_targets),
                            # which enemy is being attacked
                        }
                    )
                }),
                'defenders': spaces.Dict(
                    {
                        # 'batteries': spaces.MultiBinary(CONFIG.ATTACKERS.QUANTITY),
                        # movement [-1 = left, 0 = straight, 1 = right]
                        'movement': spaces.MultiDiscrete([-1, 0, 1] * CONFIG.DEFENDERS.QUANTITY),
                        # Actions for the missiles:
                        'missiles': spaces.Dict(
                            {
                                # 'launch: 0 - None, 1 - launch,
                                'launch': spaces.MultiBinary(
                                    CONFIG.DEFENDERS.MISSILES_PER_UNIT * CONFIG.DEFENDERS.QUANTITY),
                                # target ID each missile is currently aiming at (including unfired missiles)
                                'enemy_tar': spaces.MultiDiscrete(defender_targets),
                                # which enemy is being attacked
                            }
                        )})
            }
        )

        pose_boxmin = pose_boxmax = np.zeros((CONFIG.DEFENDERS.QUANTITY, 4))
        pose_boxmin[:] = np.array([0, 0, -1, -1])
        pose_boxmax[:] = np.array([CONFIG.WIDTH, CONFIG.HEIGHT, 1, 1])

        self.observation_space = \
            spaces.Dict(
                {
                    # state-space for the defending team batteries
                    'defenders': spaces.Dict({
                        'pose': spaces.Box(pose_boxmin, pose_boxmax, shape=(CONFIG.DEFENDERS.QUANTITY, 4)),
                        # pose holds the position of each battery
                        'health': spaces.Box(0, 1, shape=(CONFIG.DEFENDERS.QUANTITY, 1)),
                        # health - the hp each battery has
                        'missiles': spaces.Dict({

                            # 0 - Ready, 1 - Launched
                            'launched': spaces.MultiBinary(CONFIG.DEFENDERS.QUANTITY *
                                                           CONFIG.DEFENDERS.MISSILES_PER_UNIT),
                            # Each missile's target is the entity number
                            'target': spaces.MultiDiscrete(
                                [CONFIG.ATTACKERS.QUANTITY] * CONFIG.DEFENDERS.QUANTITY),
                            # pose is [x,y,velocity,heading angle]
                            'pose': spaces.Box(pose_boxmin, pose_boxmax, shape=(CONFIG.DEFENDERS.QUANTITY, 4)),
                            # Missiles health is binary
                            'health': spaces.MultiBinary(CONFIG.DEFENDERS.QUANTITY *
                                                         CONFIG.DEFENDERS.MISSILES_PER_UNIT)
                        }),
                    }),
                    # cities are immobile and have no action
                    'cities': spaces.Dict({
                        'pose': spaces.Box(pose_boxmin, pose_boxmax, shape=(CONFIG.CITIES.QUANTITY, 4)),
                        'health': spaces.Box(0, 1, shape=(CONFIG.CITIES.QUANTITY, 1)),
                    }),

                    # state-space for the attacking team bombers
                    'attackers': spaces.Dict({
                        'pose': spaces.Box(pose_boxmin, pose_boxmax, shape=(CONFIG.ATTACKERS.QUANTITY, 4)),
                        'health': spaces.Box(0, 1, shape=(CONFIG.ATTACKERS.QUANTITY, 1)),
                        'missiles': spaces.Dict({

                            # 0 - Ready, 1 - Launched
                            'launched': spaces.MultiBinary(CONFIG.ATTACKERS.QUANTITY *
                                                           CONFIG.ATTACKERS.MISSILES_PER_UNIT),
                            # Each missile's target is the entity number
                            'target': spaces.MultiDiscrete(
                                [CONFIG.DEFENDERS.QUANTITY] * CONFIG.ATTACKERS.QUANTITY),
                            # pose is [x,y,z,heading angle]
                            'pose': spaces.Box(pose_boxmin, pose_boxmax, shape=(CONFIG.ATTACKERS.QUANTITY, 4)),
                            # Missiles health is binary
                            'health': spaces.MultiBinary(CONFIG.ATTACKERS.QUANTITY *
                                                         CONFIG.ATTACKERS.MISSILES_PER_UNIT),
                        }),
                    }),
                    'sensors': spaces.Dict({
                        'vision': spaces.Box(0, 255, shape=(CONFIG.WIDTH, CONFIG.HEIGHT))
                    })
                })

        # self.observation = {
        #     'enemy_bat': {
        #         'pose': np.zeros((CONFIG.DEFENDERS.QUANTITY, 4)),
        #         'health': np.ones((CONFIG.DEFENDERS.QUANTITY)),
        #         'missiles': {
        #             'launch': np.zeros(
        #                 (CONFIG.DEFENDERS.QUANTITY, CONFIG.DEFENDERS.QUANTITY)),
        #             # 0 - batt, 1 - Missile
        #             'enemy_tar': np.zeros(
        #                 (CONFIG.DEFENDERS.QUANTITY, CONFIG.ENEMY_MISSILES.NUM_OF_BATTERIES)),
        #             'enemy_atc': np.zeros(
        #                 (CONFIG.ENNEMY_BATTERY.NUM_OF_BATTERIES, CONFIG.ENEMY_MISSILES.NUM_OF_BATTERIES)),
        #             'pose': np.zeros(
        #                 (CONFIG.ENNEMY_BATTERY.NUM_OF_BATTERIES, CONFIG.ENEMY_MISSILES.NUM_OF_BATTERIES, 4)),
        #             'health': np.ones((CONFIG.ENNEMY_BATTERY.NUM_OF_BATTERIES, CONFIG.ENEMY_MISSILES.NUM_OF_BATTERIES)),
        #             'targets': np.zeros(
        #                 (CONFIG.ENNEMY_BATTERY.NUM_OF_BATTERIES, CONFIG.ENEMY_MISSILES.NUM_OF_BATTERIES, 2), dtype=int)
        #             #     {
        #             #     'missiles': np.zeros((CONFIG.ENNEMY_BATTERY.NUMBER, CONFIG.ENEMY_MISSILES.NUMBER, 2), dtype=int)
        #             #     # 'fr_bats': np.zeros((CONFIG.ENNEMY_BATTERY.NUMBER, CONFIG.ENEMY_MISSILES.NUMBER), dtype=int),
        #             #     # 'fr_missiles': np.zeros((CONFIG.ENNEMY_BATTERY.NUMBER, CONFIG.ENEMY_MISSILES.NUMBER), dtype=int)
        #             # },
        #         },
        #     },
        #     'enemy_cities': {
        #         'pose': np.zeros((CONFIG.ENNEMY_CITIES_BATTERY.NUM_OF_BATTERIES, 4)),
        #         'health': np.ones((CONFIG.ENNEMY_CITIES_BATTERY.NUM_OF_BATTERIES)),
        #         'missiles': {
        #             'launch': np.zeros(
        #                 (CONFIG.ENNEMY_CITIES_BATTERY.NUM_OF_BATTERIES, CONFIG.ENNEMY_CITIES.NUM_OF_BATTERIES)),
        #             'enemy_tar': np.zeros(
        #                 (CONFIG.ENNEMY_CITIES_BATTERY.NUM_OF_BATTERIES, CONFIG.ENNEMY_CITIES.NUM_OF_BATTERIES)),
        #             'enemy_atc': np.zeros(
        #                 (CONFIG.ENNEMY_CITIES_BATTERY.NUM_OF_BATTERIES, CONFIG.ENNEMY_CITIES.NUM_OF_BATTERIES)),
        #             'pose': np.zeros(
        #                 (CONFIG.ENNEMY_CITIES_BATTERY.NUM_OF_BATTERIES, CONFIG.ENNEMY_CITIES.NUM_OF_BATTERIES, 4)),
        #             'health': np.ones(
        #                 (CONFIG.ENNEMY_CITIES_BATTERY.NUM_OF_BATTERIES, CONFIG.ENNEMY_CITIES.NUM_OF_BATTERIES)),
        #             # 'attacker': np.zeros((CONFIG.ENNEMY_CITIES_BATTERY.NUMBER, CONFIG.ENNEMY_CITIES.NUMBER), dtype=int)
        #             # 'attackers': {
        #             #     'fr_bats': np.zeros((CONFIG.ENNEMY_CITIES_BATTERY.NUMBER, CONFIG.ENNEMY_CITIES.NUMBER), dtype=int),
        #             #     'fr_missiles': np.zeros((CONFIG.ENNEMY_CITIES_BATTERY.NUMBER, CONFIG.ENNEMY_CITIES.NUMBER), dtype=int)
        #             # },
        #         },
        #     },
        #     'friends_bat': {
        #         'pose': np.zeros((CONFIG.FRIENDLY_BATTERY.NUM_OF_BATTERIES, 4)),
        #         'health': np.ones((CONFIG.FRIENDLY_BATTERY.NUM_OF_BATTERIES)),
        #         'missiles': {
        #             'launch': np.zeros(
        #                 (CONFIG.FRIENDLY_BATTERY.NUM_OF_BATTERIES, CONFIG.FRIENDLY_MISSILES.NUM_OF_BATTERIES)),
        #             'enemy_tar': np.zeros(
        #                 (CONFIG.FRIENDLY_BATTERY.NUM_OF_BATTERIES, CONFIG.FRIENDLY_MISSILES.NUM_OF_BATTERIES)),
        #             'enemy_atc': np.zeros(
        #                 (CONFIG.FRIENDLY_BATTERY.NUM_OF_BATTERIES, CONFIG.FRIENDLY_MISSILES.NUM_OF_BATTERIES)),
        #             'pose': np.zeros(
        #                 (CONFIG.FRIENDLY_BATTERY.NUM_OF_BATTERIES, CONFIG.FRIENDLY_MISSILES.NUM_OF_BATTERIES, 4)),
        #             'health': np.ones(
        #                 (CONFIG.FRIENDLY_BATTERY.NUM_OF_BATTERIES, CONFIG.FRIENDLY_MISSILES.NUM_OF_BATTERIES)),
        #             'targets': np.zeros(
        #                 (CONFIG.FRIENDLY_BATTERY.NUM_OF_BATTERIES, CONFIG.FRIENDLY_MISSILES.NUM_OF_BATTERIES, 2),
        #                 dtype=int)
        #             #     {
        #             #     # 'bats': np.zeros((CONFIG.FRIENDLY_BATTERY.NUMBER, CONFIG.FRIENDLY_MISSILES.NUMBER), dtype=int),
        #             #     'missiles': np.zeros((CONFIG.FRIENDLY_BATTERY.NUMBER, CONFIG.FRIENDLY_MISSILES.NUMBER, 2), dtype=int)
        #             #
        #             # },
        #
        #         },
        #     },
        #     'sensors': {
        #         'vision': np.zeros((CONFIG.WIDTH, CONFIG.HEIGHT, 3)),
        #     }
        # }

        #############  Init Enemy and Friendly batteries and missiles #############################################
        # Defender's units
        bat_pose_x = np.random.uniform(CONFIG.DEFENDERS.INIT_POS_RANGE[0] * CONFIG.WIDTH,
                                       CONFIG.DEFENDERS.INIT_POS_RANGE[1] * CONFIG.WIDTH,
                                       size=CONFIG.DEFENDERS.QUANTITY)
        bat_pose_y = np.random.uniform(CONFIG.DEFENDERS.INIT_HEIGHT_RANGE[0] * CONFIG.WIDTH,
                                       CONFIG.DEFENDERS.INIT_HEIGHT_RANGE[1] * CONFIG.WIDTH,
                                       size=CONFIG.DEFENDERS.QUANTITY)
        # self.defenders = [Unit(
        #     pose=[bat_pose_x[ind], bat_pose_y[ind], CONFIG.DEFENDERS.SPEED, CONFIG.DEFENDERS.LAUNCH_THETA],
        #     health=1.0, missiles_count=CONFIG.DEFENDERS.MISSILES_PER_UNIT) for ind in
        #     range(CONFIG.DEFENDERS.QUANTITY)]

        # defender team units
        # columns are [id, x, y, velocity, direction, health, num of missiles, target_id]
        id_offset = 0
        self.defenders = np.zeros((CONFIG.DEFENDERS.QUANTITY, 8))
        self.defenders[:, 0] = np.arange(CONFIG.DEFENDERS.QUANTITY)
        self.defenders[:, 1] = bat_pose_x
        self.defenders[:, 2] = bat_pose_y
        self.defenders[:, 3] = CONFIG.DEFENDERS.SPEED
        self.defenders[:, 4] = CONFIG.DEFENDERS.LAUNCH_THETA
        self.defenders[:, 5] = 1
        self.defenders[:, 6] = CONFIG.DEFENDERS.MISSILES_PER_UNIT
        id_offset += CONFIG.DEFENDERS.QUANTITY

        # defender cities
        init_height = np.array(CONFIG.CITIES.INIT_HEIGHT_RANGE) * CONFIG.HEIGHT
        init_pose = np.array(CONFIG.CITIES.INIT_POS_RANGE) * CONFIG.WIDTH
        city_pose_x = np.random.uniform(init_pose[0], init_pose[1], CONFIG.CITIES.QUANTITY)
        city_pose_y = np.random.uniform(init_height[0], init_height[1], CONFIG.CITIES.QUANTITY)

        # self.cities = [City(pose=[bat_pose_x[ind], bat_pose_y[ind], CONFIG.CITIES.SPEED,
        #                           CONFIG.CITIES.LAUNCH_THETA],
        #                     health=1.0) for ind in
        #                range(CONFIG.QUANTITY.NUM_OF_BATTERIES)]
        # columns are [id, x, y, health]
        self.cities = np.zeros((CONFIG.CITIES.QUANTITY, 5))
        self.cities[:, 0] = np.arange(CONFIG.CITIES.QUANTITY) + id_offset
        self.cities[:, 1] = city_pose_x
        self.cities[:, 2] = city_pose_y
        self.cities[:, 3] = 1
        id_offset += CONFIG.CITIES.QUANTITY

        # defender missiles

        missiles_id = np.arange(CONFIG.DEFENDERS.QUANTITY*CONFIG.DEFENDERS.MISSILES_PER_UNIT) + id_offset
        # columns are [id, parent_id, x, y, velocity, direction, health, target_id, launched]
        self.defenders_missiles = np.zeros((CONFIG.DEFENDERS.QUANTITY, 8))
        self.defenders_missiles[:, 0] = np.arange(CONFIG.DEFENDERS.QUANTITY)  # parent id
        self.defenders_missiles[:, 1] = bat_pose_x
        self.defenders_missiles[:, 2] = bat_pose_y
        self.defenders_missiles[:, 3] = CONFIG.DEFENDERS.SPEED
        self.defenders_missiles[:, 4] = CONFIG.DEFENDERS.LAUNCH_THETA
        self.defenders_missiles[:, 5] = 1
        self.defenders_missiles[:, 6] = 0
        self.defenders_missiles = np.tile(self.defenders_missiles, (CONFIG.DEFENDERS.MISSILES_PER_UNIT,1))
        self.defenders_missiles = [missiles_id, self.defenders_missiles]

        # Attacker's units
        init_height = np.array(CONFIG.ATTACKERS.INIT_HEIGHT_RANGE) * CONFIG.HEIGHT
        init_pose = np.array(CONFIG.ATTACKERS.INIT_POS_RANGE) * CONFIG.WIDTH

        bat_pose_x = np.random.uniform(init_pose[0], init_pose[1], CONFIG.ATTACKERS.QUANTITY)
        bat_pose_y = np.random.uniform(init_height[0], init_height[1], CONFIG.ATTACKERS.QUANTITY)

        # columns are [id, x, y, velocity, direction, health, num of missiles, target_id]
        id_offset = 0
        self.attackers = np.zeros((CONFIG.ATTACKERS.QUANTITY, 8))
        self.attackers[:, 0] = np.arange(CONFIG.ATTACKERS.QUANTITY)
        self.attackers[:, 1] = bat_pose_x
        self.attackers[:, 2] = bat_pose_y
        self.attackers[:, 3] = CONFIG.ATTACKERS.SPEED
        self.attackers[:, 4] = CONFIG.ATTACKERS.LAUNCH_THETA
        self.attackers[:, 5] = 1
        self.attackers[:, 6] = CONFIG.ATTACKERS.MISSILES_PER_UNIT
        id_offset += CONFIG.ATTACKERS.QUANTITY

        missiles_id = np.arange(CONFIG.DEFENDERS.QUANTITY * CONFIG.DEFENDERS.MISSILES_PER_UNIT) + id_offset
        # columns are [id, parent_id, x, y, velocity, direction, health, target_id, launched]
        self.attackers_missiles = np.zeros((CONFIG.DEFENDERS.QUANTITY, 8))
        self.attackers_missiles[:, 0] = np.arange(CONFIG.DEFENDERS.QUANTITY)  # parent id
        self.attackers_missiles[:, 1] = bat_pose_x
        self.attackers_missiles[:, 2] = bat_pose_y
        self.attackers_missiles[:, 3] = CONFIG.DEFENDERS.SPEED
        self.attackers_missiles[:, 4] = CONFIG.DEFENDERS.LAUNCH_THETA
        self.attackers_missiles[:, 5] = 1
        self.attackers_missiles[:, 6] = 0
        self.attackers_missiles = np.tile(self.attackers_missiles, (CONFIG.DEFENDERS.MISSILES_PER_UNIT, 1))
        self.attackers_missiles = [missiles_id, self.attackers_missiles]

        # cities_pose = np.zeros((CONFIG.ENNEMY_CITIES.NUMBER, 2))
        # cities_pose[:, 0] = np.random.uniform(CONFIG.WIDTH // 2, CONFIG.WIDTH, size=CONFIG.ENNEMY_CITIES.NUMBER)
        # cities_health = np.ones((CONFIG.ENNEMY_CITIES.NUMBER))
        # self.enemy_cities = EnemyCities(CONFIG.ENNEMY_CITIES.NUMBER, pose=cities_pose, health=cities_health)
        #################################################################################################################

        # self.observation_space = spaces.Box(
        #     low=0,
        #     high=255,
        #     shape=(CONFIG.OBSERVATION.HEIGHT, CONFIG.OBSERVATION.WIDTH, 3),
        #     dtype=np.uint8,
        # )

        # Custom configuration
        # ------------------------------------------
        # For each custom attributes
        for key, value in custom_config.items():

            # Check if attributes is valid
            try:
                _ = rgetattr(CONFIG, key)
            except AttributeError as e:
                print("Invalid custom configuration: {}".format(e))
                sys.exit(1)

            # Modify it
            rsetattr(CONFIG, key, value)

        # Initializing objects
        # ------------------------------------------
        # No display while no render
        self.clock = None
        self.display = None

    # def _collisions_cities(self):
    #     """Check for cities collisions.
    #
    #     Check cities destroyed by enemy missiles.
    #     """
    #     # Cities
    #     enemy_cities_pos = self.enemy_cities.pose
    #     enemy_cities_health = self.enemy_cities.health
    #
    #     # Enemy missiles current position
    #     for en_bat in self.enemy_batteries:
    #         en_missiles_pose = en_bat.mi
    #
    #
    #     # Enemy missiles current positions
    #     enemy_m = self.enemy_missiles.enemy_missiles[:, [2, 3]]
    #
    #     # Align cities and enemy missiles
    #     cities_dup = np.repeat(cities, enemy_m.shape[0], axis=0)
    #     enemy_m_dup = np.tile(enemy_m, reps=[cities.shape[0], 1])
    #
    #     # Compute distances
    #     dx = enemy_m_dup[:, 0] - cities_dup[:, 0]
    #     dy = enemy_m_dup[:, 1] - cities_dup[:, 1]
    #     distances = np.sqrt(np.square(dx) + np.square(dy))
    #
    #     # Get cities destroyed by enemy missiles
    #     exploded = distances <= (
    #         CONFIG.ENEMY_MISSILES.RADIUS + CONFIG.CITIES.RADIUS)
    #     exploded = exploded.astype(int)
    #     exploded = np.reshape(exploded, (cities.shape[0], enemy_m.shape[0]))
    #
    #     # Get destroyed cities
    #     cities_out = np.argwhere(
    #         (np.sum(exploded, axis=1) >= 1) &
    #         (cities[:, 2] > 0.0)
    #     )
    #
    #     # Update timestep reward
    #     self.reward_timestep += CONFIG.REWARD.DESTROYED_CITY * \
    #         cities_out.shape[0]
    #
    #     # Destroy the cities
    #     self.cities.cities[cities_out, 2] -= 1

    # def _collisions_missiles(self):
    #     """Check for missiles collisions."""
    #
    #     ################# find submitted enemies and corresponding friends ... #################
    #     for en_bat in self.enemy_batteries:
    #
    #         launched_enemy_missiles = np.where(en_bat.missiles.launch ==1)[0]
    #         target_fr_missiles = en_bat.missiles.targets['missiles'][launched_enemy_missiles]
    #
    #         launched_pose = en_bat.missiles.pose[launched_enemy_missiles, :2]
    #         target_pose
    #
    #         friendly_bats_indices = list(target_fr_missiles[0])
    #
    #         for fr_bat in self.friendly_batteries:
    #
    #
    #
    #             launch_en_pose = en_bat['missiles'][launched_enemy_missiles][0:2]
    #             target_fr_pose = en_bat['missiles'][target_fr_missiles][0:2]
    #
    #     live_launched_fr, live_en_cities = live_launched_fr_missile[0].shape[0], live_enemy_cities[0].shape[0]
    #     if live_launched_fr > 0 and live_en_cities > 0:
    #         fr_missiles_pos = friends_missiless['pose'][live_launched_fr_missile][:, :2]
    #         fr_missiles_targets = friends_missiless['targets'][live_launched_fr_missile]
    #         live_en_cities_pos = observation['enemy_cities']['pose'][live_enemy_cities][:, :2]
    #
    #     # Check enemy missiles destroyed by friendly exploding missiles.
    #     # """
    #     # # Friendly exploding missiles
    #     # friendly_exploding = self.friendly_missiles.missiles_explosion
    #     #
    #     # # Enemy missiles current positions
    #     # enemy_missiles = self.enemy_missiles.enemy_missiles[:, [2, 3]]
    #     #
    #     # # Align enemy missiles and friendly exploding ones
    #     # enemy_m_dup = np.repeat(enemy_missiles,
    #     #                         friendly_exploding.shape[0],
    #     #                         axis=0)
    #     # friendly_e_dup = np.tile(friendly_exploding,
    #     #                          reps=[enemy_missiles.shape[0], 1])
    #     #
    #     # # Compute distances
    #     # dx = friendly_e_dup[:, 0] - enemy_m_dup[:, 0]
    #     # dy = friendly_e_dup[:, 1] - enemy_m_dup[:, 1]
    #     # distances = np.sqrt(np.square(dx) + np.square(dy))
    #     #
    #     # # Get enemy missiles inside an explosion radius
    #     # inside_radius = distances <= (
    #     #     friendly_e_dup[:, 2] + CONFIG.ENEMY_MISSILES.RADIUS)
    #     # inside_radius = inside_radius.astype(int)
    #     # inside_radius = np.reshape(
    #     #     inside_radius,
    #     #     (enemy_missiles.shape[0], friendly_exploding.shape[0]),
    #     # )
    #     #
    #     # # Remove theses missiles
    #     # missiles_out = np.argwhere(np.sum(inside_radius, axis=1) >= 1)
    #     # self.enemy_missiles.enemy_missiles = np.delete(
    #     #     self.enemy_missiles.enemy_missiles,
    #     #     np.squeeze(missiles_out),
    #     #     axis=0,
    #     # )
    #     #
    #     # # Compute current reward
    #     # nb_missiles_destroyed = missiles_out.shape[0]
    #     # self.reward_timestep += CONFIG.REWARD.DESTROYED_ENEMEY_MISSILES * \
    #     #     nb_missiles_destroyed

    def _compute_sensor_observation(self, sensor_type):
        """
        Compute observation. Current game graphics.

        """

        enemy_bats = self.enemy_batteries
        friendly_bats = self.friendly_batteries
        enemy_cities = self.enemy_cities

        # Reset observation
        if sensor_type == 'vision':
            im = self.observation['sensors'][sensor_type].astype('uint8')
            # im = np.zeros((CONFIG.WIDTH, CONFIG.HEIGHT, 3), dtype=np.uint8)
            im[:, :, 0] = CONFIG.COLORS.BACKGROUND[0]
            im[:, :, 1] = CONFIG.COLORS.BACKGROUND[1]
            im[:, :, 2] = CONFIG.COLORS.BACKGROUND[2]

            # Draw objects
            for idx, enemy_bat in enumerate(enemy_bats):
                enemy_bat.render(im, self.observation['enemy_bat']['missiles']['health'][idx, :])
            for idx, friend_bat in enumerate(friendly_bats):
                friend_bat.render(im, self.observation['friends_bat']['missiles']['health'][idx, :])
            for idx, city_bat in enumerate(enemy_cities):
                city_bat.render(im, self.observation['enemy_cities']['missiles']['health'][idx, :])

            return im

            # self.batteries.render(self.observation)
            # self.cities.render(self.observation)
            # self.enemy_missiles.render(self.observation)
            # self.friendly_missiles.render(self.observation)
            # self.target.render(self.observation)

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
            self.observation['sensors']['vision'],
            (CONFIG.OBSERVATION.HEIGHT, CONFIG.OBSERVATION.WIDTH),
            interpolation=cv2.INTER_AREA,
        )
        return processed_observation.astype(CONFIG.DTYPE)

    def _extract_observation(self):

        for fr_bat_ind in range(CONFIG.ATTACKERS.QUANTITY):
            fr_battery = self.friendly_batteries[fr_bat_ind]
            self.observation['attackers']['pose'][fr_bat_ind] = fr_battery.pose  # [0:2]
            self.observation['attackers']['health'][fr_bat_ind] = fr_battery.health
            self.observation['attackers']['missiles']['pose'][fr_bat_ind] = fr_battery.missiles.pose  # [:, 0:2]
            self.observation['attackers']['missiles']['health'][fr_bat_ind] = fr_battery.missiles.health
            self.observation['attackers']['missiles']['launch'][fr_bat_ind] = fr_battery.missiles.launch
            self.observation['attackers']['missiles']['enemy_tar'][fr_bat_ind] = fr_battery.missiles.enemy_tar
            self.observation['attackers']['missiles']['enemy_atc'][fr_bat_ind] = fr_battery.missiles.enemy_atc
            self.observation['attackers']['missiles']['targets'][fr_bat_ind] = fr_battery.missiles.targets['missiles']

        for en_bat_ind in range(CONFIG.DEFENDERS.QUANTITY):
            en_battery = self.enemy_batteries[en_bat_ind]
            self.observation['defenders']['pose'][en_bat_ind] = en_battery.pose  # [0:2]
            self.observation['defenders']['health'][en_bat_ind] = en_battery.health
            self.observation['defenders']['missiles']['pose'][en_bat_ind] = en_battery.missiles.pose  # [:, 0:2]
            self.observation['defenders']['missiles']['health'][en_bat_ind] = en_battery.missiles.health
            self.observation['defenders']['missiles']['launch'][en_bat_ind] = en_battery.missiles.launch
            self.observation['defenders']['missiles']['enemy_tar'][en_bat_ind] = en_battery.missiles.enemy_tar
            self.observation['defenders']['missiles']['enemy_atc'][en_bat_ind] = en_battery.missiles.enemy_atc
            self.observation['defenders']['missiles']['targets'][en_bat_ind] = en_battery.missiles.targets['missiles']

        for city_bat_ind in range(CONFIG.CITIES.QUANTITY):
            en_cities_bat = self.enemy_cities[city_bat_ind]
            self.observation['cities']['pose'][city_bat_ind] = en_cities_bat.pose
            self.observation['cities']['health'][city_bat_ind] = en_cities_bat.health

        obs = self.observation
        return obs

    def reset(self):
        """Reset the environment.

        Returns:
            observation (numpy.array): the processed observation.
        """
        # Reset timestep and rewards
        self.timestep = 0
        self.reward_total = 0.0
        self.reward_timestep = 0.0

        #############  Iinit Enemy and Friendly batteries and missiles #############################################
        #  Enemy batteries and missiles
        bat_pose_x = np.random.uniform(CONFIG.DEFENDERS.INIT_POS_RANGE[0] * CONFIG.WIDTH,
                                       CONFIG.DEFENDERS.INIT_POS_RANGE[1] * CONFIG.WIDTH,
                                       size=CONFIG.DEFENDERS.QUANTITY)
        bat_pose_y = np.random.uniform(CONFIG.DEFENDERS.INIT_HEIGHT_RANGE[0] * CONFIG.WIDTH,
                                       CONFIG.DEFENDERS.INIT_HEIGHT_RANGE[1] * CONFIG.WIDTH,
                                       size=CONFIG.DEFENDERS.QUANTITY)

        self.defenders = [Unit(id=ind,
                               pose=[bat_pose_x[ind], bat_pose_y[ind], CONFIG.DEFENDERS.SPEED,
                                     CONFIG.DEFENDERS.LAUNCH_THETA],
                               health=1.0, missiles_count=CONFIG.DEFENDERS.MISSILES_PER_UNIT) for ind in
                          range(CONFIG.DEFENDERS.QUANTITY)]

        # Attacker units and missiles
        Init_Height = np.array(CONFIG.ATTACKERS.INIT_HEIGHT_RANGE) * CONFIG.HEIGHT
        Init_Pose = np.array(CONFIG.ATTACKERS.INIT_POS_RANGE) * CONFIG.WIDTH
        bat_pose_x = np.random.uniform(Init_Pose[0], Init_Pose[1], CONFIG.ATTACKERS.QUANTITY)
        bat_pose_y = np.random.uniform(Init_Height[0], Init_Height[1], CONFIG.ATTACKERS.QUANTITY)

        self.attackers = [Unit(id=ind,
                               pose=[bat_pose_x[ind], bat_pose_y[ind], CONFIG.ATTACKERS.SPEED,
                                     CONFIG.ATTACKERS.LAUNCH_THETA], health=1.0,
                               missiles_count=CONFIG.ATTACKERS.QUANTITY) for ind in
                          range(CONFIG.ATTACKERS.QUANTITY)]
        # self.target = Target()

        # Enemy cities
        Init_Height = np.array(CONFIG.CITIES.INIT_HEIGHT_RANGE) * CONFIG.HEIGHT
        Init_Pose = np.array(CONFIG.CITIES.INIT_POS_RANGE) * CONFIG.WIDTH
        if CONFIG.CITIES.QUANTITY == 1:
            bat_pose_x, bat_pose_y = [Init_Pose[0]], [Init_Height[0]]
        else:
            bat_pose_x = np.random.uniform(Init_Pose[0], Init_Pose[1], CONFIG.CITIES.QUANTITY)
            bat_pose_y = np.random.uniform(Init_Height[0], Init_Height[1],
                                           CONFIG.CITIES.QUANTITY)

        self.enemy_cities = [City(id=ind + CONFIG.ATTACKERS.QUANTITY,
                                  pose=[bat_pose_x[ind], bat_pose_y[ind], CONFIG.CITIES.SPEED,
                                        CONFIG.CITIES.LAUNCH_THETA], health=1.0) for ind in
                             range(CONFIG.CITIES.QUANTITY)]
        #################################################################################################################

        return self._extract_observation()

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
        # Reset current reward
        # ------------------------------------------

        self.reward_timestep = 0.0

        # Step functions: update state
        # ------------------------------------------
        friendly_batteries_reward = 0
        attackers_units_pose = np.zeros(len(self.attackers), 2)
        attackers_missiles_pose = np.zeros(len(self.attackers) * CONFIG.ATTACKERS.MISSILES_PER_UNIT, 2)
        defenders_units_pose = np.zeros(len(self.defenders), 2)
        defenders_missiles_pose = np.zeros(len(self.defenders) * CONFIG.DEFENDERS.MISSILES_PER_UNIT, 2)
        cities_pose = self.cities[:].pose[0:2]

        # Launch missiles
        # attackers
        attackers_launch_id = np.argwhere(action['attackers']['missiles']['launch'])  # id of units with launch action
        missiles_to_launch = get_missiles_to_launch(self.attackers_missiles, attackers_launch_id)
        self.attackers_missiles[missiles_to_launch, 8] = True  # set missiles to launched

        # defenders
        defenders_launch_id = np.argwhere(action['defenders']['missiles']['launch'])  # id of units with launch action
        missiles_to_launch = get_missiles_to_launch(self.defenders_missiles, defenders_launch_id)
        self.defenders_missiles[missiles_to_launch, 8] = True  # set missiles to launched

        # Roll movements
        # attackers units
        delta_attackers = get_movement(self.attackers[:3], self.attackers[:4], action['attackers']['movement'])
        self.attackers[:, 1:5] += delta_attackers
        # defenders units
        delta_defenders = get_movement(self.defenders[:3], self.defenders[:4], action['defenders']['movement'])
        self.defenders[:, 1:5] += delta_defenders
        # attackers missiles - non launched
        non_launched = np.argwhere(not self.attackers_missiles[:, 8])
        non_launched_parents = self.attackers_missiles[non_launched, 1]
        attackers_missiles_delta = delta_attackers[non_launched_parents]
        self.attackers_missiles[non_launched, 1:4] += attackers_missiles_delta
        # defenders missiles - non launched
        # attackers missiles - launched
        # defenders missiles - launched
        # Check for collisions
        # ------------------------------------------


        self.collisions(self.observation)
        # self._collisions_cities()

        # Calculate rewards
        # ------------------------------------------
        cities_reward = 0

        # Check if episode is finished
        # ------------------------------------------

        done = False  # done_enemy_cities or done_enemy_batteries

        # Render every objects
        # ------------------------------------------

        self.observation['sensors']['vision'] = \
            self._compute_sensor_observation('vision')

        # Return everything
        # ------------------------------------------

        self.timestep += 1
        self.reward_total += friendly_battery_reward + enemy_battery_reward + cities_reward + self.reward_timestep
        self.observation['sensors']['vision'] = self._process_observation()
        return self.observation, self.reward_timestep, done, {}

    def get_entities_indexes(self, friends_missiless, observation):
        live_non_launched_fr_missile = np.where(
            np.logical_and(friends_missiless['health'] == 1, friends_missiless['launch'] == 0) == True)
        live_launched_fr_missile = np.where(
            np.logical_and(friends_missiless['health'] == 1, friends_missiless['launch'] == 1) == True)

        live_enemy_cities = np.where(enenmy_cities['health'] == 1)
        target_enemy_cities = np.where(
            np.logical_and(np.logical_and(enenmy_cities['health'] == 1, enenmy_cities['launch'] == 1),
                           enenmy_cities['enemy_atc'] == True) == True)
        non_target_enemy_cities = np.where(
            np.logical_and(np.logical_and(enenmy_cities['health'] == 1, enenmy_cities['launch'] == 1),
                           enenmy_cities['enemy_atc'] == False) == True)
        return live_non_launched_fr_missile, live_launched_fr_missile, target_enemy_cities, non_target_enemy_cities

    def render(self, mode="rgb_array"):
        """Render the environment.

        This function renders the environment observation. To check what the
        processed observation looks like, it can also renders it.

        Args:
            mode (str): the render mode. Possible values are "rgb_array" and
                "processed_observation".
        """
        if not self.display:
            pygame.init()
            # pygame.mouse.set_visible(False)
            self.clock = pygame.time.Clock()
            pygame.display.set_caption("MissileCommand")

        # Display the normal observation
        if mode == "rgb_array":
            self.display = pygame.display.set_mode(
                (CONFIG.WIDTH, CONFIG.HEIGHT))
            obs = self.observation['sensors'][
                'vision']  # np.array(Image.fromarray(self.observation['sensors']['vision']))
            surface = pygame.surfarray.make_surface(obs)
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
        return self.observation

    def close(self):
        """Close the environment."""
        if self.display:
            pygame.quit()


if __name__ == "__main__":
    # Create the environment
    env = gym.make("gym_missile_command:missile-command-v0")

    # Reset it
    observation = env.reset()

    t = 1
