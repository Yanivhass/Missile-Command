"""Main environment class."""
import contextlib
import sys
import cv2
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from PIL import Image

from config import CONFIG

# from game.Entities import Unit, City
# from game.batteries import EnemyBattery, FriendlyBattery, CityBattery
# from game.cities import EnemyCities
# from gym_missile_command.game.missile import EnemyMissiles, FriendlyMissiles
# from game.target import Target
# from utils import rgetattr, rsetattr

# Import Pygame and remove welcome message
with contextlib.redirect_stdout(None):
    import pygame


# def check_hits(missiles_xy, targets_xy, targeting_matrix, explosion_radius):
#     """
#
#     Args:
#         missiles_xy: A [n,2] matrix of x,y coordinates
#         targets_xy: A [m,2] matrix of x,y coordinates
#         targeting_matrix: a binary [m,n] matrix, [i,j] is true
#             if entity i is the target of missiles j. False otherwise.
#         explosion_radius: a scalar. If missiles is in explosion radius of it's target
#             it is detonated
#
#     Returns:
#         hits: a [n,1] matrix of how many missiles can hit each target
#     """
#     # distance matrix where [i,j] is the distance from the i'th target to the j'th missiles
#     dist_matrix = np.linalg.norm(targets_xy[:, None, :] - missiles_xy[None, :, :], axis=-1)
#     dist_matrix = np.where(dist_matrix <= explosion_radius, 1, 0)
#     dist_matrix = dist_matrix & targeting_matrix  # only detonate on assigned target
#     hits = np.sum(dist_matrix, axis=1)
#     return hits, detonated


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
    angles[np.argwhere(action == -1)] = angles[np.argwhere(action == -1)] + 10  # turn left 10 degrees
    angles[np.argwhere(action == 1)] = angles[np.argwhere(action == 1)] - 10  # turn right 10 degrees
    angles = np.mod(angles, 360)
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
    # En_NB_ACTIONS = CONFIG.ENEMY_MISSILES.NB_ACTIONS
    # Fr_NB_ACTIONS = CONFIG.FRIENDLY_MISSILES.NB_ACTIONS
    metadata = {"render_modes": ["human", "rgb_array"],
                'video.frames_per_second': CONFIG.FPS}

    def __init__(self):
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
        attackers_count = CONFIG.ATTACKERS.QUANTITY
        attackers_missile_count = CONFIG.ATTACKERS.QUANTITY * CONFIG.ATTACKERS.MISSILES_PER_UNIT
        defenders_count = CONFIG.DEFENDERS.QUANTITY
        defenders_missile_count = CONFIG.DEFENDERS.QUANTITY * CONFIG.DEFENDERS.MISSILES_PER_UNIT
        cities_count = CONFIG.CITIES.QUANTITY
        # columns are [id, x, y, velocity, direction, health, num of missiles, target_id]
        self.attackers = np.zeros((attackers_count, 8))
        # columns are [id, parent_id, x, y, velocity, direction, health, target_id, launched]
        self.attackers_missiles = np.zeros((attackers_missile_count, 8))
        # columns are [id, x, y, velocity, direction, health, num of missiles, target_id]
        self.defenders = np.zeros((defenders_count, 8))
        # columns are [id, parent_id, x, y, velocity, direction, health, target_id, launched]
        self.defenders_missiles = np.zeros((defenders_missile_count, 8))
        # columns are [id, x, y, health]
        self.cities = np.zeros((cities_count, 5))

        '''
        Action space for the game
        '''
        num_of_targets = defenders_count + cities_count
        attacker_targets = np.ones((1, attackers_missile_count)) * num_of_targets
        # attacker_targets[:,1] = num_of_targets
        num_of_targets = attackers_count + attackers_missile_count
        defender_targets = np.ones((1, defenders_missile_count)) * num_of_targets
        # defender_targets[:, 1] = num_of_targets
        self.action_space = spaces.Dict(
            # Actions, currently only for attackers
            {'attackers': spaces.Dict(
                {
                    # 'batteries': spaces.MultiBinary(CONFIG.ATTACKERS.QUANTITY),
                    # movement [0 = left, 1 = straight, 2 = right]
                    'movement': spaces.MultiDiscrete([3] * attackers_count),
                    # Actions for the missiles:
                    'missiles': spaces.Dict(
                        {
                            # 'launch: 0 - None, 1 - launch,
                            'launch': spaces.MultiBinary(attackers_missile_count),
                            # target ID each missile is currently aiming at (including unfired missiles)
                            'enemy_tar': spaces.MultiDiscrete(
                                np.ones((1, attackers_missile_count)) * (defenders_count + cities_count)),
                            # which enemy is being attacked
                        }
                    )
                }),
                'defenders': spaces.Dict(
                    {
                        # 'batteries': spaces.MultiBinary(CONFIG.ATTACKERS.QUANTITY),
                        # movement [0 = left, 1 = straight, 2 = right]
                        'movement': spaces.MultiDiscrete([3] * defenders_count),
                        # Actions for the missiles:
                        'missiles': spaces.Dict(
                            {
                                # 'launch: 0 - None, 1 - launch,
                                'launch': spaces.MultiBinary(defenders_missile_count),
                                # target ID each missile is currently aiming at (including unfired missiles)
                                'enemy_tar': spaces.MultiDiscrete(
                                    np.ones((1, defenders_missile_count)) * (
                                            attackers_count + attackers_missile_count)),
                                # which enemy is being attacked
                            }
                        )})
            }
        )

        # pose_boxmin = pose_boxmax = np.zeros((1, 4))
        pose_boxmin = np.array([0, 0, 0, 0])
        pose_boxmax = np.array([CONFIG.WIDTH, CONFIG.HEIGHT, 100, 360])

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
                            'health': spaces.MultiBinary(defenders_missile_count)
                        }),
                    }),
                    # cities are immobile and have no action
                    'cities': spaces.Dict({
                        'pose': spaces.Box(
                            np.tile(pose_boxmin, (cities_count, 1)),
                            np.tile(pose_boxmax, (cities_count, 1)),
                            shape=(cities_count, 4)),
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
                        }),
                    }),
                    'sensors': spaces.Dict({
                        'vision': spaces.Box(0, 255, shape=(CONFIG.WIDTH, CONFIG.HEIGHT))
                    })
                })

        self.observation = self.observation_space.sample()
        # Set the seed for reproducibility
        self.seed(CONFIG.SEED)
        # Initialize the state
        self.reset()

        # self.observation_space = spaces.Box(
        #     low=0,
        #     high=255,
        #     shape=(CONFIG.OBSERVATION.HEIGHT, CONFIG.OBSERVATION.WIDTH, 3),
        #     dtype=np.uint8,
        # )

        # Custom configuration
        # ------------------------------------------
        # For each custom attributes
        # for key, value in custom_config.items():
        #
        #     # Check if attributes is valid
        #     try:
        #         _ = rgetattr(CONFIG, key)
        #     except AttributeError as e:
        #         print("Invalid custom configuration: {}".format(e))
        #         sys.exit(1)
        #
        #     # Modify it
        #     rsetattr(CONFIG, key, value)

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

        enemy_bats = self.enemy_batteries
        friendly_bats = self.friendly_batteries
        enemy_cities = self.enemy_cities

        # Reset observation
        if sensor_type == 'vision':
            # im = self.observation['sensors'][sensor_type].astype('uint8')
            im = np.zeros((CONFIG.WIDTH, CONFIG.HEIGHT, 3), dtype=np.uint8)
            im[:, :, 0] = CONFIG.COLORS.BACKGROUND[0]
            im[:, :, 1] = CONFIG.COLORS.BACKGROUND[1]
            im[:, :, 2] = CONFIG.COLORS.BACKGROUND[2]

            # Draw objects
            # for idx, enemy_bat in enumerate(enemy_bats):
            #     enemy_bat.render(im, self.observation['enemy_bat']['missiles']['health'][idx, :])
            # for idx, friend_bat in enumerate(friendly_bats):
            #     friend_bat.render(im, self.observation['friends_bat']['missiles']['health'][idx, :])
            # for idx, city_bat in enumerate(enemy_cities):
            #     city_bat.render(im, self.observation['enemy_cities']['missiles']['health'][idx, :])

            im[self.attackers[:, 1], self.attackers[:, 2], :] = CONFIG.ATTACKERS.COLOR
            im[self.attackers_missiles[:, 1], self.attackers_missiles[:, 2], :] = CONFIG.ATTACKERS.MISSILES.COLOR
            im[self.defenders[:, 1], self.defenders[:, 2], :] = CONFIG.DEFENDERS.COLOR
            im[self.defenders_missiles[:, 1], self.defenders_missiles[:, 2], :] = CONFIG.DEFENDERS.MISSILES.COLOR
            im[self.cities[:, 1], self.attackers[:, 2], :] = CONFIG.CITIES.COLOR

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
        self.observation['attackers']['pose'] = self.attackers[:, 1:3]
        self.observation['attackers']['velocity'] = self.attackers[:, 3]
        self.observation['attackers']['direction'] = self.attackers[:, 4]
        self.observation['attackers']['health'] = self.attackers[:, 5]
        self.observation['attackers']['missiles']['pose'] = self.attackers_missiles[:, 1:3]
        self.observation['attackers']['missiles']['velocity'] = self.attackers_missiles[:, 3]
        self.observation['attackers']['missiles']['direction'] = self.attackers_missiles[:, 4]
        self.observation['attackers']['missiles']['health'] = self.attackers_missiles[:, 5]
        self.observation['attackers']['missiles']['target'] = self.attackers_missiles[:, 6]
        self.observation['attackers']['missiles']['launched'] = self.attackers_missiles[:, 7]

        self.observation['defenders']['pose'] = self.defenders[:, 1:3]
        self.observation['defenders']['velocity'] = self.defenders[:, 3]
        self.observation['defenders']['direction'] = self.defenders[:, 4]
        self.observation['defenders']['health'] = self.defenders[:, 5]
        self.observation['defenders']['missiles']['pose'] = self.defenders_missiles[:, 1:3]
        self.observation['defenders']['missiles']['velocity'] = self.defenders_missiles[:, 3]
        self.observation['defenders']['missiles']['direction'] = self.defenders_missiles[:, 4]
        self.observation['defenders']['missiles']['health'] = self.defenders_missiles[:, 5]
        self.observation['defenders']['missiles']['target'] = self.defenders_missiles[:, 6]
        self.observation['defenders']['missiles']['launched'] = self.defenders_missiles[:, 7]

        self.observation['cities']['pose'] = self.cities[:, 1:3]
        self.observation['cities']['health'] = self.cities[:, 3]

        return self.observation

    def reset(self):
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
        # columns are [0-id, 1-x, 2-y, 3-velocity, 4-direction, 5-health, 6-target_id, 7-num of missiles]
        id_offset = 0
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

        # columns are [id, x, y, health]
        self.cities[:, 0] = np.arange(CONFIG.CITIES.QUANTITY) + id_offset
        self.cities[:, 1] = city_pose_x
        self.cities[:, 2] = city_pose_y
        self.cities[:, 3] = 1
        id_offset += CONFIG.CITIES.QUANTITY

        # defender missiles

        missiles_id = np.arange(CONFIG.DEFENDERS.QUANTITY * CONFIG.DEFENDERS.MISSILES_PER_UNIT) + id_offset
        self.defenders_missiles = np.zeros((CONFIG.DEFENDERS.QUANTITY, 9))
        # columns are [0-id, 1-x, 2-y, 3-velocity, 4-direction, 5-health, 6-target_id, 7-launched, 8-parent_id]
        self.defenders_missiles[:, 1] = bat_pose_x
        self.defenders_missiles[:, 2] = bat_pose_y
        self.defenders_missiles[:, 3] = CONFIG.DEFENDERS.SPEED
        self.defenders_missiles[:, 4] = CONFIG.DEFENDERS.LAUNCH_THETA
        self.defenders_missiles[:, 5] = 1
        self.defenders_missiles[:, 6] = 0
        self.defenders_missiles[:, 8] = np.arange(CONFIG.DEFENDERS.QUANTITY)  # parent id
        self.defenders_missiles = np.tile(self.defenders_missiles, (CONFIG.DEFENDERS.MISSILES_PER_UNIT, 1))
        self.defenders_missiles[:, 0] = missiles_id

        # Attacker's units
        init_height = np.array(CONFIG.ATTACKERS.INIT_HEIGHT_RANGE) * CONFIG.HEIGHT
        init_pose = np.array(CONFIG.ATTACKERS.INIT_POS_RANGE) * CONFIG.WIDTH

        bat_pose_x = np.random.uniform(init_pose[0], init_pose[1], CONFIG.ATTACKERS.QUANTITY)
        bat_pose_y = np.random.uniform(init_height[0], init_height[1], CONFIG.ATTACKERS.QUANTITY)

        # columns are [id, x, y, velocity, direction, health, num of missiles, target_id]
        id_offset = 0
        self.attackers[:, 0] = np.arange(CONFIG.ATTACKERS.QUANTITY)
        self.attackers[:, 1] = bat_pose_x
        self.attackers[:, 2] = bat_pose_y
        self.attackers[:, 3] = CONFIG.ATTACKERS.SPEED
        self.attackers[:, 4] = CONFIG.ATTACKERS.LAUNCH_THETA
        self.attackers[:, 5] = 1
        self.attackers[:, 6] = CONFIG.ATTACKERS.MISSILES_PER_UNIT
        id_offset += CONFIG.ATTACKERS.QUANTITY

        missiles_id = np.arange(CONFIG.ATTACKERS.QUANTITY * CONFIG.ATTACKERS.MISSILES_PER_UNIT) + id_offset
        # columns are [0-id, 1-x, 2-y, 3-velocity, 4-direction, 5-health, 6-target_id, 7-launched, 8-parent_id]
        self.attackers_missiles = np.zeros((CONFIG.ATTACKERS.QUANTITY, 9))
        self.attackers_missiles[:, 1] = bat_pose_x
        self.attackers_missiles[:, 2] = bat_pose_y
        self.attackers_missiles[:, 3] = CONFIG.ATTACKERS.SPEED
        self.attackers_missiles[:, 4] = CONFIG.ATTACKERS.LAUNCH_THETA
        self.attackers_missiles[:, 5] = 1
        self.attackers_missiles[:, 7] = 0  # target id
        self.attackers_missiles[:, 7] = 0  # launched
        self.attackers_missiles[:, 8] = np.arange(CONFIG.ATTACKERS.QUANTITY)  # parent id
        self.attackers_missiles = np.tile(self.attackers_missiles, (CONFIG.ATTACKERS.MISSILES_PER_UNIT, 1))
        self.attackers_missiles[:, 0] = missiles_id

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
        attackers_count = CONFIG.ATTACKERS.QUANTITY
        defenders_count = CONFIG.DEFENDERS.QUANTITY
        # Reset current reward
        # ------------------------------------------

        self.reward_timestep = 0.0

        # Step functions: update state
        # ------------------------------------------

        # Launch missiles
        # attackers

        missiles_to_launch = np.argwhere(action['attackers']['missiles']['launch'])  # missiles with launch action
        # missiles_to_launch = get_missiles_to_launch(self.attackers_missiles, attackers_launch_id)
        self.attackers_missiles[missiles_to_launch, 8] = True  # set missiles to launched

        # defenders
        missiles_to_launch = np.argwhere(action['defenders']['missiles']['launch']) # issiles with launch action
        # missiles_to_launch = get_missiles_to_launch(self.defenders_missiles, defenders_launch_id)
        self.defenders_missiles[missiles_to_launch, 8] = True  # set missiles to launched

        # Roll movements
        # attackers units
        delta_attackers = get_movement(self.attackers[:, 3], self.attackers[:, 4], action['attackers']['movement'])
        self.attackers[:, 1:5] += delta_attackers
        # defenders units
        delta_defenders = get_movement(self.defenders[:, 3], self.defenders[:, 4], action['defenders']['movement'])
        self.defenders[:, 1:5] += delta_defenders
        # attackers missiles - non launched
        attackers_non_launched = np.argwhere(not self.attackers_missiles[:, 8])
        non_launched_parents = self.attackers_missiles[attackers_non_launched, 1]
        attackers_missiles_delta = delta_attackers[non_launched_parents]
        self.attackers_missiles[attackers_non_launched, 1:4] += attackers_missiles_delta
        # defenders missiles - non launched
        defenders_non_launched = np.argwhere(not self.defenders_missiles[:, 8])
        non_launched_parents = self.attackers_missiles[defenders_non_launched, 1]
        defenders_missiles_delta = delta_defenders[non_launched_parents]
        self.defenders_missiles[defenders_non_launched, 1:4] += defenders_missiles_delta
        # attackers missiles - launched
        attackers_launched = np.argwhere(not self.attackers_missiles[:, 8])
        angles = self.attackers_missiles[attackers_launched, 4] * np.pi / 180.0  # degreess to rads
        delta = (self.attackers_missiles[attackers_launched, 3] * np.array(
            [np.cos(angles), np.sin(angles)])).transpose()
        self.attackers_missiles[:, 1:2] += delta
        # defenders missiles - launched
        defenders_launched = np.argwhere(not self.defenders_missiles[:, 8])
        angles = self.defenders_missiles[defenders_launched, 4] * np.pi / 180.0  # degreess to rads
        delta = (self.defenders_missiles[defenders_launched, 3] * np.array(
            [np.cos(angles), np.sin(angles)])).transpose()
        self.defenders_missiles[:, 1:2] += delta

        # Check for collisions
        # ------------------------------------------
        att_exp_rad = CONFIG.ATTACKERS.MISSILES.EXPLOSION_RADIUS
        def_exp_rad = CONFIG.DEFENDERS.MISSILES.EXPLOSION_RADIUS
        # get range from launched defender missiles to attackers and their launched missiles
        all_attackers = np.vstack((self.attackers, self.attackers_missiles))
        target_id = self.defenders_missiles[defenders_launched, 7]
        targets_xy = all_attackers[target_id, 1:2]
        hits = np.linalg.norm(targets_xy - self.defenders_missiles[defenders_launched, 1:2], axis=-1) <= def_exp_rad
        self.attackers[np.isin(self.attackers[:, 0], target_id[hits]), [3, 5]] = 0  # velocity, health = 0
        self.attackers_missiles[
            np.isin(self.attackers_missiles[:, 0], target_id[hits]), [3, 5]] = 0  # velocity, health = 0
        # get range from launched attacker missiles to defender units and cities
        all_defenders = np.vstack((self.defenders, self.defenders_missiles, self.cities))
        target_id = self.attackers_missiles[attackers_launched, 7]
        targets_xy = all_defenders[target_id, 1:2]
        hits = np.linalg.norm(targets_xy - self.attackers_missiles[attackers_launched, 1:2], axis=-1) <= att_exp_rad
        self.defenders[np.isin(self.attackers[:, 0], target_id[hits]), [3, 5]] = 0  # velocity, health = 0
        self.defenders_missiles[
            np.isin(self.attackers_missiles[:, 0], target_id[hits]), [3, 5]] = 0  # velocity, health = 0
        self.cities[np.isin(self.cities[:, 0], target_id[hits]), 3] = 0  # health = 0

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
        # self.reward_total += friendly_battery_reward + enemy_battery_reward + cities_reward + self.reward_timestep
        self.reward_total = 0
        self.observation['sensors']['vision'] = self._process_observation()
        return self.observation, self.reward_timestep, done, {}

    # def get_entities_indexes(self, friends_missiless, observation):
    #     live_non_launched_fr_missile = np.where(
    #         np.logical_and(friends_missiless['health'] == 1, friends_missiless['launch'] == 0) == True)
    #     live_launched_fr_missile = np.where(
    #         np.logical_and(friends_missiless['health'] == 1, friends_missiless['launch'] == 1) == True)
    #
    #     live_enemy_cities = np.where(enenmy_cities['health'] == 1)
    #     target_enemy_cities = np.where(
    #         np.logical_and(np.logical_and(enenmy_cities['health'] == 1, enenmy_cities['launch'] == 1),
    #                        enenmy_cities['enemy_atc'] == True) == True)
    #     non_target_enemy_cities = np.where(
    #         np.logical_and(np.logical_and(enenmy_cities['health'] == 1, enenmy_cities['launch'] == 1),
    #                        enenmy_cities['enemy_atc'] == False) == True)
    #     return live_non_launched_fr_missile, live_launched_fr_missile, target_enemy_cities, non_target_enemy_cities

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
