"""Main environment class."""
import contextlib
import sys
import cv2
import gym
import numpy as np
from gym import spaces
from PIL import Image

from config import CONFIG
from game.batteries import EnemyBattery, FriendlyBattery, CityBattery
from game.cities import EnemyCities
# from gym_missile_command.game.missile import EnemyMissiles, FriendlyMissiles
from game.target import Target
from utils import rgetattr, rsetattr

# Import Pygame and remove welcome message
with contextlib.redirect_stdout(None):
    import pygame


class MissileCommandEnv(gym.Env, EnemyBattery, FriendlyBattery):
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
        self.action_space = spaces.Dict(
            # Actions of friendly units
            {'friends': spaces.Dict(
                { 'batteries': spaces.MultiBinary(CONFIG.FRIENDLY_BATTERY.NUMBER),
                  # For the batteries there are only two actions for each battery: a missile has been
                  # launched or not

                  # Actions for the missiles:
                  'missiles':  spaces.Dict(
                      {
                        # 'launch: 0 - None, 1 - launch,
                        'launch':  spaces.MultiDiscrete([[2]*CONFIG.FRIENDLY_MISSILES.NUMBER]*CONFIG.FRIENDLY_BATTERY.NUMBER),
                        # 0 - battery target, 1 - missile - target
                        'enemy_tar':   spaces.MultiDiscrete(
                            [[2]*CONFIG.FRIENDLY_MISSILES.NUMBER]*CONFIG.FRIENDLY_BATTERY.NUMBER),
                        # 0 - non-attacker, 1- attacker
                        'enemy_atc': spaces.MultiDiscrete(
                            [[2] * CONFIG.FRIENDLY_MISSILES.NUMBER] * CONFIG.FRIENDLY_BATTERY.NUMBER),
                        # which enemy is being attacked
                        'actions': spaces.MultiDiscrete([[CONFIG.FRIENDLY_MISSILES.NB_ACTIONS]*CONFIG.FRIENDLY_MISSILES.NUMBER]*CONFIG.FRIENDLY_BATTERY.NUMBER),
                        # A set which holds all of the possible actions
                        'targets': spaces.MultiDiscrete([[[CONFIG.ENNEMY_CITIES_BATTERY.NUMBER, CONFIG.ENNEMY_CITIES.NUMBER]]*CONFIG.FRIENDLY_MISSILES.NUMBER]*CONFIG.FRIENDLY_BATTERY.NUMBER)
                        # what is the target (which battery / city)


                        #     spaces.Dict(
                        #     {
                        #         # 'bats': spaces.MultiDiscrete([[CONFIG.ENNEMY_BATTERY.NUMBER] * CONFIG.FRIENDLY_MISSILES.NUMBER] * CONFIG.FRIENDLY_BATTERY.NUMBER),
                        #         'missiles': spaces.MultiDiscrete([[[CONFIG.ENNEMY_CITIES_BATTERY.NUMBER, CONFIG.ENNEMY_CITIES.NUMBER]]*CONFIG.FRIENDLY_MISSILES.NUMBER]*CONFIG.FRIENDLY_BATTERY.NUMBER)
                        # })
                      }
                  )}),
            # The enemies have a very similar action space
             'enemies': spaces.Dict(
                 {'batteries': spaces.MultiBinary(CONFIG.ENNEMY_BATTERY.NUMBER),
                  'missiles': spaces.Dict(
                      {
                      # 'launch: 0 - None, 1 - launch,
                       'launch': spaces.MultiDiscrete([[2]*CONFIG.ENEMY_MISSILES.NUMBER]*CONFIG.ENNEMY_BATTERY.NUMBER),
                          # 0 - target bat, 1 - target missile
                        'enemy_tar': spaces.MultiDiscrete(
                              [[2] * CONFIG.ENEMY_MISSILES.NUMBER] * CONFIG.ENNEMY_BATTERY.NUMBER),
                        'enemy_atc': spaces.MultiDiscrete(
                              [[2] * CONFIG.ENEMY_MISSILES.NUMBER] * CONFIG.ENNEMY_BATTERY.NUMBER),
                       'actions': spaces.MultiDiscrete([[CONFIG.ENEMY_MISSILES.NB_ACTIONS] * CONFIG.ENEMY_MISSILES.NUMBER]*CONFIG.ENNEMY_BATTERY.NUMBER),
                       'targets': spaces.MultiDiscrete([[[CONFIG.FRIENDLY_BATTERY.NUMBER, CONFIG.FRIENDLY_MISSILES.NUMBER]] * CONFIG.ENEMY_MISSILES.NUMBER] * CONFIG.ENNEMY_BATTERY.NUMBER)
                           # spaces.Dict(
                           # {
                           #  'missiles': spaces.MultiDiscrete([[[CONFIG.FRIENDLY_BATTERY.NUMBER, CONFIG.FRIENDLY_MISSILES.NUMBER]] * CONFIG.ENEMY_MISSILES.NUMBER] * CONFIG.ENNEMY_BATTERY.NUMBER)
                           #  # 'fr_bats': spaces.MultiDiscrete([[CONFIG.FRIENDLY_BATTERY.NUMBER] * CONFIG.ENEMY_MISSILES.NUMBER] * CONFIG.ENNEMY_BATTERY.NUMBER),
                           #  # 'fr_missiles': spaces.MultiDiscrete([[CONFIG.FRIENDLY_MISSILES.NUMBER] * CONFIG.ENEMY_MISSILES.NUMBER] * CONFIG.ENNEMY_BATTERY.NUMBER)
                           #  })
                       }
                  )}),
            # cities also have similar action space
             'cities': spaces.Dict(
                 {
                     'batteries': spaces.MultiBinary(CONFIG.ENNEMY_CITIES_BATTERY.NUMBER),
                     'missiles': spaces.Dict(
                         {
                             'launch': spaces.MultiDiscrete([[2]*CONFIG.ENNEMY_CITIES.NUMBER]*CONFIG.ENNEMY_CITIES_BATTERY.NUMBER),
                             'enemy_tar': spaces.MultiDiscrete(
                                 [[2] * CONFIG.ENNEMY_CITIES.NUMBER] * CONFIG.ENNEMY_CITIES_BATTERY.NUMBER),
                             'enemy_atc': spaces.MultiDiscrete(
                                 [[2] * CONFIG.ENNEMY_CITIES.NUMBER] * CONFIG.ENNEMY_CITIES_BATTERY.NUMBER),
                             'actions': spaces.MultiDiscrete([[2]*CONFIG.ENNEMY_CITIES.NUMBER]*CONFIG.ENNEMY_CITIES_BATTERY.NUMBER),

                         })
                   })
            }
        )

        pose_boxmin = pose_boxmax = np.zeros((CONFIG.ENNEMY_BATTERY.NUMBER, 4))
        pose_boxmin[:] = np.array([0, 0, -1, -1])
        pose_boxmax[:] = np.array([CONFIG.WIDTH, CONFIG.HEIGHT, 1, 1])

        self.observation_space = \
            spaces.Dict(
            {
                # state-space for the enemy batteries
            'enemy_bat':  spaces.Dict( {
                'pose': spaces.Box(pose_boxmin, pose_boxmax, shape=(CONFIG.ENNEMY_BATTERY.NUMBER, 4)),
                # pose holds the position of each battery
                'health': spaces.Box(0, 1, shape=(CONFIG.ENNEMY_BATTERY.NUMBER, 1)),
                # health - the hp each battery has

                # state space of the missiles is the same as the action space
                'missiles': spaces.Dict( {
                    # 0 - None, 1 - Missile, 2 - Bats
                    'launch': spaces.MultiDiscrete([[2]*CONFIG.ENEMY_MISSILES.NUMBER]*CONFIG.ENNEMY_BATTERY.NUMBER),
                    'enemy_tar': spaces.MultiDiscrete([[2] * CONFIG.ENEMY_MISSILES.NUMBER] * CONFIG.ENNEMY_BATTERY.NUMBER),
                    'enemy_atc': spaces.MultiDiscrete([[2] * CONFIG.ENEMY_MISSILES.NUMBER] * CONFIG.ENNEMY_BATTERY.NUMBER),
                    'pose': spaces.Box(pose_boxmin, pose_boxmax, shape=(CONFIG.ENNEMY_BATTERY.NUMBER, 4)),
                    'health': spaces.Box(0, 1, shape=(CONFIG.ENNEMY_BATTERY.NUMBER, CONFIG.ENEMY_MISSILES.NUMBER)),
                    'targets': spaces.MultiDiscrete([[[CONFIG.FRIENDLY_BATTERY.NUMBER, CONFIG.FRIENDLY_MISSILES.NUMBER]] * CONFIG.ENEMY_MISSILES.NUMBER] * CONFIG.ENNEMY_BATTERY.NUMBER),
                        # spaces.Dict(
                        # {
                        #  'missiles': spaces.MultiDiscrete([[[CONFIG.FRIENDLY_BATTERY.NUMBER, CONFIG.FRIENDLY_MISSILES.NUMBER]] * CONFIG.ENEMY_MISSILES.NUMBER] * CONFIG.ENNEMY_BATTERY.NUMBER),
                        #  # 'fr_bats': spaces.MultiDiscrete([[CONFIG.FRIENDLY_BATTERY.NUMBER] * CONFIG.ENEMY_MISSILES.NUMBER] * CONFIG.ENNEMY_BATTERY.NUMBER),
                        #  # 'fr_missiles': spaces.MultiDiscrete([[CONFIG.FRIENDLY_MISSILES.NUMBER] * CONFIG.ENEMY_MISSILES.NUMBER] * CONFIG.ENNEMY_BATTERY.NUMBER)
                        #  })
                }),
                # 'attackers': spaces.Dict(
                #     # 0 - None, 1 - Missile, 2 - Bats
                #     { 'launch': spaces.MultiDiscrete([3]*CONFIG.ENNEMY_BATTERY.NUMBER),
                #       'fr_bats': spaces.MultiDiscrete([CONFIG.FRIENDLY_BATTERY.NUMBER] * CONFIG.ENNEMY_BATTERY.NUMBER),
                #       'fr_missiles': spaces.MultiDiscrete([CONFIG.FRIENDLY_MISSILES.NUMBER] * CONFIG.ENNEMY_BATTERY.NUMBER)
                #      })
            }),
                # cities have similar values to the batteries
            'enemy_cities': spaces.Dict( {
                'pose': spaces.Box(pose_boxmin, pose_boxmax, shape=(CONFIG.ENNEMY_BATTERY.NUMBER, 4)),
                # 'pose': spaces.Box(0, CONFIG.WIDTH, shape=(CONFIG.ENNEMY_CITIES_BATTERY.NUMBER, 4)),
                # 'vel': spaces.Box(-1, 1, shape=(CONFIG.ENNEMY_CITIES_BATTERY.NUMBER, 2)),
                'health': spaces.Box(0, 1, shape=(CONFIG.ENNEMY_CITIES_BATTERY.NUMBER, 1)),
                'missiles': spaces.Dict({
                    'launch': spaces.MultiDiscrete([[2] * CONFIG.ENNEMY_CITIES.NUMBER] * CONFIG.ENNEMY_CITIES_BATTERY.NUMBER),
                    'enemy_tar': spaces.MultiDiscrete(
                        [[2] * CONFIG.ENNEMY_CITIES.NUMBER] * CONFIG.ENNEMY_CITIES_BATTERY.NUMBER),
                    'enemy_atc': spaces.MultiDiscrete(
                        [[2] * CONFIG.ENNEMY_CITIES.NUMBER] * CONFIG.ENNEMY_CITIES_BATTERY.NUMBER),
                    'pose': spaces.Box(pose_boxmin, pose_boxmax, shape=(CONFIG.ENNEMY_BATTERY.NUMBER, 4)),
                    'health': spaces.Box(0, 1, shape=(CONFIG.ENNEMY_CITIES_BATTERY.NUMBER, CONFIG.ENNEMY_CITIES.NUMBER)),

                }),
            }),
            # friends batteries have the same params as the enemy battery
            'friends_bat': spaces.Dict({
                'pose': spaces.Box(pose_boxmin, pose_boxmax, shape=(CONFIG.ENNEMY_BATTERY.NUMBER, 4)),
                'health': spaces.Box(0, 1, shape = (CONFIG.FRIENDLY_BATTERY.NUMBER, 1)),
                'missiles': spaces.Dict({
                    ############ 0 - No, 1 - Yes
                    'launch': spaces.MultiDiscrete([[2] * CONFIG.FRIENDLY_MISSILES.NUMBER] * CONFIG.FRIENDLY_BATTERY.NUMBER),
                    ############ 0 - bat, 1 - Missile
                    'enemy_tar': spaces.MultiDiscrete(
                        [[2] * CONFIG.FRIENDLY_MISSILES.NUMBER] * CONFIG.FRIENDLY_BATTERY.NUMBER),
                    'enemy_atc': spaces.MultiDiscrete(
                        [[2] * CONFIG.FRIENDLY_MISSILES.NUMBER] * CONFIG.FRIENDLY_BATTERY.NUMBER),
                    'pose': spaces.Box(pose_boxmin, pose_boxmax, shape=(CONFIG.ENNEMY_BATTERY.NUMBER, 4)),
                    'health': spaces.Box(0, 1, shape=(CONFIG.FRIENDLY_BATTERY.NUMBER, CONFIG.FRIENDLY_MISSILES.NUMBER)),   # 1
                    'targets': spaces.MultiDiscrete([[[CONFIG.ENNEMY_CITIES_BATTERY.NUMBER, CONFIG.ENNEMY_CITIES.NUMBER]] * CONFIG.FRIENDLY_MISSILES.NUMBER] * CONFIG.FRIENDLY_BATTERY.NUMBER)
                        # spaces.Dict(
                        # {
                        #  # 'bats': spaces.MultiDiscrete([[CONFIG.ENNEMY_BATTERY.NUMBER] * CONFIG.FRIENDLY_MISSILES.NUMBER] * CONFIG.FRIENDLY_BATTERY.NUMBER),
                        #  'missiles': spaces.MultiDiscrete([[[CONFIG.ENNEMY_CITIES_BATTERY.NUMBER, CONFIG.ENNEMY_CITIES.NUMBER]] * CONFIG.FRIENDLY_MISSILES.NUMBER] * CONFIG.FRIENDLY_BATTERY.NUMBER)
                        #  }),
                    # 'attackers': spaces.Dict(
                    #     # 0 - None, 1 - Missile, 2 - Bats
                    #     {'launch': spaces.MultiDiscrete([[3] * CONFIG.ENNEMY_CITIES.NUMBER] * CONFIG.ENNEMY_CITIES_BATTERY.NUMBER),
                    #      'en_bats': spaces.MultiDiscrete([[CONFIG.FRIENDLY_BATTERY.NUMBER] * CONFIG.ENNEMY_CITIES.NUMBER] * CONFIG.ENNEMY_CITIES_BATTERY.NUMBER),
                    #      'en_missiles': spaces.MultiDiscrete([[CONFIG.FRIENDLY_MISSILES.NUMBER] * CONFIG.ENNEMY_CITIES.NUMBER] * CONFIG.ENNEMY_CITIES_BATTERY.NUMBER)
                    #      })
                }),
                # 'attackers': spaces.Dict(
                #     # 0 - None, 1 - Missile, 2 - Bats
                #     {'launch': spaces.MultiDiscrete([3] * CONFIG.FRIENDLY_BATTERY.NUMBER),
                #      'en_bats': spaces.MultiDiscrete([CONFIG.ENNEMY_BATTERY.NUMBER] * CONFIG.FRIENDLY_BATTERY.NUMBER),
                #      'en_missiles': spaces.MultiDiscrete([CONFIG.ENEMY_MISSILES.NUMBER] * CONFIG.FRIENDLY_BATTERY.NUMBER)
                #      })
                # })
            }),
            'sensors': spaces.Dict({
                'vision': spaces.Box(0, 255, shape=(CONFIG.WIDTH, CONFIG.HEIGHT))
            })
        })

        self.observation = {
            'enemy_bat': {
                'pose': np.zeros((CONFIG.ENNEMY_BATTERY.NUMBER, 4)),
                'health': np.ones((CONFIG.ENNEMY_BATTERY.NUMBER)),
                'missiles': {
                    'launch': np.zeros((CONFIG.ENNEMY_BATTERY.NUMBER, CONFIG.ENEMY_MISSILES.NUMBER)),
                    # 0 - batt, 1 - Missile
                    'enemy_tar': np.zeros((CONFIG.ENNEMY_BATTERY.NUMBER, CONFIG.ENEMY_MISSILES.NUMBER)),
                    'enemy_atc': np.zeros((CONFIG.ENNEMY_BATTERY.NUMBER, CONFIG.ENEMY_MISSILES.NUMBER)),
                    'pose': np.zeros((CONFIG.ENNEMY_BATTERY.NUMBER, CONFIG.ENEMY_MISSILES.NUMBER, 4)),
                    'health': np.ones((CONFIG.ENNEMY_BATTERY.NUMBER, CONFIG.ENEMY_MISSILES.NUMBER)),
                    'targets': np.zeros((CONFIG.ENNEMY_BATTERY.NUMBER, CONFIG.ENEMY_MISSILES.NUMBER, 2), dtype=int)
                    #     {
                    #     'missiles': np.zeros((CONFIG.ENNEMY_BATTERY.NUMBER, CONFIG.ENEMY_MISSILES.NUMBER, 2), dtype=int)
                    #     # 'fr_bats': np.zeros((CONFIG.ENNEMY_BATTERY.NUMBER, CONFIG.ENEMY_MISSILES.NUMBER), dtype=int),
                    #     # 'fr_missiles': np.zeros((CONFIG.ENNEMY_BATTERY.NUMBER, CONFIG.ENEMY_MISSILES.NUMBER), dtype=int)
                    # },
                },
            },
            'enemy_cities': {
                'pose': np.zeros((CONFIG.ENNEMY_CITIES_BATTERY.NUMBER, 4)),
                'health': np.ones((CONFIG.ENNEMY_CITIES_BATTERY.NUMBER)),
                'missiles': {
                    'launch': np.zeros((CONFIG.ENNEMY_CITIES_BATTERY.NUMBER, CONFIG.ENNEMY_CITIES.NUMBER)),
                    'enemy_tar': np.zeros((CONFIG.ENNEMY_CITIES_BATTERY.NUMBER, CONFIG.ENNEMY_CITIES.NUMBER)),
                    'enemy_atc': np.zeros((CONFIG.ENNEMY_CITIES_BATTERY.NUMBER, CONFIG.ENNEMY_CITIES.NUMBER)),
                    'pose': np.zeros((CONFIG.ENNEMY_CITIES_BATTERY.NUMBER, CONFIG.ENNEMY_CITIES.NUMBER, 4)),
                    'health': np.ones((CONFIG.ENNEMY_CITIES_BATTERY.NUMBER, CONFIG.ENNEMY_CITIES.NUMBER)),
                    # 'attacker': np.zeros((CONFIG.ENNEMY_CITIES_BATTERY.NUMBER, CONFIG.ENNEMY_CITIES.NUMBER), dtype=int)
                    # 'attackers': {
                    #     'fr_bats': np.zeros((CONFIG.ENNEMY_CITIES_BATTERY.NUMBER, CONFIG.ENNEMY_CITIES.NUMBER), dtype=int),
                    #     'fr_missiles': np.zeros((CONFIG.ENNEMY_CITIES_BATTERY.NUMBER, CONFIG.ENNEMY_CITIES.NUMBER), dtype=int)
                    # },
                },
            },
            'friends_bat': {
                'pose': np.zeros((CONFIG.FRIENDLY_BATTERY.NUMBER, 4)),
                'health': np.ones((CONFIG.FRIENDLY_BATTERY.NUMBER)),
                'missiles': {
                    'launch': np.zeros((CONFIG.FRIENDLY_BATTERY.NUMBER, CONFIG.FRIENDLY_MISSILES.NUMBER)),
                    'enemy_tar': np.zeros((CONFIG.FRIENDLY_BATTERY.NUMBER, CONFIG.FRIENDLY_MISSILES.NUMBER)),
                    'enemy_atc': np.zeros((CONFIG.FRIENDLY_BATTERY.NUMBER, CONFIG.FRIENDLY_MISSILES.NUMBER)),
                    'pose': np.zeros((CONFIG.FRIENDLY_BATTERY.NUMBER, CONFIG.FRIENDLY_MISSILES.NUMBER, 4)),
                    'health': np.ones((CONFIG.FRIENDLY_BATTERY.NUMBER, CONFIG.FRIENDLY_MISSILES.NUMBER)),
                    'targets': np.zeros((CONFIG.FRIENDLY_BATTERY.NUMBER, CONFIG.FRIENDLY_MISSILES.NUMBER, 2), dtype=int)
                    #     {
                    #     # 'bats': np.zeros((CONFIG.FRIENDLY_BATTERY.NUMBER, CONFIG.FRIENDLY_MISSILES.NUMBER), dtype=int),
                    #     'missiles': np.zeros((CONFIG.FRIENDLY_BATTERY.NUMBER, CONFIG.FRIENDLY_MISSILES.NUMBER, 2), dtype=int)
                    #
                    # },

                },
            },
            'sensors': {
                'vision': np.zeros((CONFIG.WIDTH, CONFIG.HEIGHT, 3)),
            }
        }

        #############  Iinit Enemy and Friendly batteries and missiles #############################################
        #  Enemy batteries
        bat_pose_x = np.random.uniform(CONFIG.ENNEMY_BATTERY.INIT_POS_RANGE[0] * CONFIG.WIDTH,CONFIG.ENNEMY_BATTERY.INIT_POS_RANGE[1] * CONFIG.WIDTH,
                                       size=CONFIG.ENNEMY_BATTERY.NUMBER)
        bat_pose_y = np.random.uniform(CONFIG.ENNEMY_BATTERY.INIT_HEIGHT_RANGE[0] * CONFIG.WIDTH,
                                       CONFIG.ENNEMY_BATTERY.INIT_HEIGHT_RANGE[1] * CONFIG.WIDTH,
                                       size=CONFIG.ENNEMY_BATTERY.NUMBER)
        self.enemy_batteries = [EnemyBattery(pose=[bat_pose_x[ind], bat_pose_y[ind], CONFIG.ENNEMY_BATTERY.SPEED, CONFIG.ENNEMY_BATTERY.LAUNCH_THETA],
                         health=1.0, missiles=CONFIG.ENEMY_MISSILES.NUMBER) for ind in  range(CONFIG.ENNEMY_BATTERY.NUMBER)]

        # Friendly bateries
        Init_Height = np.array(CONFIG.FRIENDLY_BATTERY.INIT_HEIGHT_RANGE) * CONFIG.HEIGHT
        Init_Pose = np.array(CONFIG.FRIENDLY_BATTERY.INIT_POS_RANGE) * CONFIG.WIDTH

        bat_pose_x = np.random.uniform(Init_Pose[0], Init_Pose[1], CONFIG.FRIENDLY_BATTERY.NUMBER)
        bat_pose_y = np.random.uniform(Init_Height[0], Init_Height[1], CONFIG.FRIENDLY_BATTERY.NUMBER)

        self.friendly_batteries = [FriendlyBattery(pose=[bat_pose_x[ind], bat_pose_y[ind], CONFIG.FRIENDLY_BATTERY.SPEED, CONFIG.FRIENDLY_BATTERY.LAUNCH_THETA],
                                                   health=1.0, missiles=CONFIG.FRIENDLY_MISSILES.NUMBER) for ind in range(CONFIG.FRIENDLY_BATTERY.NUMBER)]
        self.target = Target()

        # Enemy cities
        Init_Height = np.array(CONFIG.ENNEMY_CITIES_BATTERY.INIT_HEIGHT_RANGE) * CONFIG.HEIGHT
        Init_Pose = np.array(CONFIG.ENNEMY_CITIES_BATTERY.INIT_POS_RANGE) * CONFIG.WIDTH

        bat_pose_x = np.random.uniform(Init_Pose[0], Init_Pose[1], CONFIG.ENNEMY_CITIES_BATTERY.NUMBER)
        bat_pose_y = np.random.uniform(Init_Height[0], Init_Height[1], CONFIG.ENNEMY_CITIES_BATTERY.NUMBER)

        self.enemy_cities = [CityBattery(pose=[bat_pose_x[ind], bat_pose_y[ind], CONFIG.ENNEMY_CITIES_BATTERY.SPEED, CONFIG.ENNEMY_CITIES_BATTERY.LAUNCH_THETA],
                                         health=1.0, missiles=CONFIG.ENNEMY_CITIES.NUMBER) for ind in range(CONFIG.ENNEMY_CITIES_BATTERY.NUMBER)]



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



    def _collisions_cities(self):
        """Check for cities collisions.

        Check cities destroyed by enemy missiles.
        """
        # Cities
        enemy_cities_pos = self.enemy_cities.pose
        enemy_cities_health = self.enemy_cities.health

        # Enemy missiles current position
        for en_bat in self.enemy_batteries:
            en_missiles_pose = en_bat.mi


        # Enemy missiles current positions
        enemy_m = self.enemy_missiles.enemy_missiles[:, [2, 3]]

        # Align cities and enemy missiles
        cities_dup = np.repeat(cities, enemy_m.shape[0], axis=0)
        enemy_m_dup = np.tile(enemy_m, reps=[cities.shape[0], 1])

        # Compute distances
        dx = enemy_m_dup[:, 0] - cities_dup[:, 0]
        dy = enemy_m_dup[:, 1] - cities_dup[:, 1]
        distances = np.sqrt(np.square(dx) + np.square(dy))

        # Get cities destroyed by enemy missiles
        exploded = distances <= (
            CONFIG.ENEMY_MISSILES.RADIUS + CONFIG.CITIES.RADIUS)
        exploded = exploded.astype(int)
        exploded = np.reshape(exploded, (cities.shape[0], enemy_m.shape[0]))

        # Get destroyed cities
        cities_out = np.argwhere(
            (np.sum(exploded, axis=1) >= 1) &
            (cities[:, 2] > 0.0)
        )

        # Update timestep reward
        self.reward_timestep += CONFIG.REWARD.DESTROYED_CITY * \
            cities_out.shape[0]

        # Destroy the cities
        self.cities.cities[cities_out, 2] -= 1

    def _collisions_missiles(self):
        """Check for missiles collisions."""

        ################# find submitted enemies and corresponding friends ... #################
        for en_bat in self.enemy_batteries:

            launched_enemy_missiles = np.where(en_bat.missiles.launch ==1)[0]
            target_fr_missiles = en_bat.missiles.targets['missiles'][launched_enemy_missiles]

            launched_pose = en_bat.missiles.pose[launched_enemy_missiles, :2]
            target_pose

            friendly_bats_indices = list(target_fr_missiles[0])

            for fr_bat in self.friendly_batteries:



                launch_en_pose = en_bat['missiles'][launched_enemy_missiles][0:2]
                target_fr_pose = en_bat['missiles'][target_fr_missiles][0:2]

        live_launched_fr, live_en_cities = live_launched_fr_missile[0].shape[0], live_enemy_cities[0].shape[0]
        if live_launched_fr > 0 and live_en_cities > 0:
            fr_missiles_pos = friends_missiless['pose'][live_launched_fr_missile][:, :2]
            fr_missiles_targets = friends_missiless['targets'][live_launched_fr_missile]
            live_en_cities_pos = observation['enemy_cities']['pose'][live_enemy_cities][:, :2]

        # Check enemy missiles destroyed by friendly exploding missiles.
        # """
        # # Friendly exploding missiles
        # friendly_exploding = self.friendly_missiles.missiles_explosion
        #
        # # Enemy missiles current positions
        # enemy_missiles = self.enemy_missiles.enemy_missiles[:, [2, 3]]
        #
        # # Align enemy missiles and friendly exploding ones
        # enemy_m_dup = np.repeat(enemy_missiles,
        #                         friendly_exploding.shape[0],
        #                         axis=0)
        # friendly_e_dup = np.tile(friendly_exploding,
        #                          reps=[enemy_missiles.shape[0], 1])
        #
        # # Compute distances
        # dx = friendly_e_dup[:, 0] - enemy_m_dup[:, 0]
        # dy = friendly_e_dup[:, 1] - enemy_m_dup[:, 1]
        # distances = np.sqrt(np.square(dx) + np.square(dy))
        #
        # # Get enemy missiles inside an explosion radius
        # inside_radius = distances <= (
        #     friendly_e_dup[:, 2] + CONFIG.ENEMY_MISSILES.RADIUS)
        # inside_radius = inside_radius.astype(int)
        # inside_radius = np.reshape(
        #     inside_radius,
        #     (enemy_missiles.shape[0], friendly_exploding.shape[0]),
        # )
        #
        # # Remove theses missiles
        # missiles_out = np.argwhere(np.sum(inside_radius, axis=1) >= 1)
        # self.enemy_missiles.enemy_missiles = np.delete(
        #     self.enemy_missiles.enemy_missiles,
        #     np.squeeze(missiles_out),
        #     axis=0,
        # )
        #
        # # Compute current reward
        # nb_missiles_destroyed = missiles_out.shape[0]
        # self.reward_timestep += CONFIG.REWARD.DESTROYED_ENEMEY_MISSILES * \
        #     nb_missiles_destroyed

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
                enemy_bat.render(im, self.observation['enemy_bat']['missiles']['health'][idx,:])
            for idx, friend_bat in enumerate(friendly_bats):
                friend_bat.render(im, self.observation['friends_bat']['missiles']['health'][idx,:])
            for idx, city_bat in enumerate(enemy_cities):
                city_bat.render(im, self.observation['enemy_cities']['missiles']['health'][idx,:])

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

        for fr_bat_ind in range(CONFIG.FRIENDLY_BATTERY.NUMBER):
            fr_battery = self.friendly_batteries[fr_bat_ind]
            self.observation['friends_bat']['pose'][fr_bat_ind] = fr_battery.pose #[0:2]
            # self.observation['friends_bat']['vel'][fr_bat_ind] = fr_battery.pose[2:4]
            self.observation['friends_bat']['health'][fr_bat_ind] = fr_battery.health
            self.observation['friends_bat']['missiles']['pose'][fr_bat_ind] = fr_battery.missiles.pose #[:, 0:2]
            # self.observation['friends_bat']['missiles']['vel'][fr_bat_ind] = fr_battery.missiles.pose[:, 2:4]
            self.observation['friends_bat']['missiles']['health'][fr_bat_ind] = fr_battery.missiles.health
            self.observation['friends_bat']['missiles']['launch'][fr_bat_ind] = fr_battery.missiles.launch
            self.observation['friends_bat']['missiles']['enemy_tar'][fr_bat_ind] = fr_battery.missiles.enemy_tar
            self.observation['friends_bat']['missiles']['enemy_atc'][fr_bat_ind] = fr_battery.missiles.enemy_atc

            # self.observation['friends_bat']['missiles']['targets']['bats'][fr_bat_ind] = fr_battery.missiles.targets['bats']
            self.observation['friends_bat']['missiles']['targets'][fr_bat_ind] = fr_battery.missiles.targets['missiles']
            # self.observation['friends_bat']['missiles']['targets']['city_bats'][fr_bat_ind]  = fr_battery.missiles.targets['city_bats']
            # self.observation['friends_bat']['missiles']['targets']['city_missiles'][fr_bat_ind] = fr_battery.missiles.targets['city_missiles']
        for en_bat_ind in range(CONFIG.ENNEMY_BATTERY.NUMBER):
            en_battery = self.enemy_batteries[en_bat_ind]
            self.observation['enemy_bat']['pose'][en_bat_ind] = en_battery.pose #[0:2]
            # self.observation['enemy_bat']['vel'][en_bat_ind] = en_battery.pose[2:4]
            self.observation['enemy_bat']['health'][en_bat_ind] = en_battery.health
            self.observation['enemy_bat']['missiles']['pose'][en_bat_ind] = en_battery.missiles.pose #[:, 0:2]
            # self.observation['enemy_bat']['missiles']['vel'][en_bat_ind] = en_battery.missiles.pose[:, 2:4]
            self.observation['enemy_bat']['missiles']['health'][en_bat_ind] = en_battery.missiles.health
            self.observation['enemy_bat']['missiles']['launch'][en_bat_ind] = en_battery.missiles.launch
            self.observation['enemy_bat']['missiles']['enemy_tar'][en_bat_ind] = en_battery.missiles.enemy_tar
            self.observation['enemy_bat']['missiles']['enemy_atc'][en_bat_ind] = en_battery.missiles.enemy_atc
            # self.observation['enemy_bat']['missiles']['targets'][en_bat_ind] = en_battery.missiles.targets

            # self.observation['enemy_bat']['missiles']['targets']['fr_bats'][en_bat_ind] = en_battery.missiles.targets['fr_bats']
            # self.observation['enemy_bat']['missiles']['targets']['fr_missiles'][en_bat_ind] = en_battery.missiles.targets['fr_missiles']
            self.observation['enemy_bat']['missiles']['targets'][en_bat_ind] = en_battery.missiles.targets['missiles']

        for city_bat_ind in range(CONFIG.ENNEMY_CITIES_BATTERY.NUMBER):
            en_cities_bat = self.enemy_cities[city_bat_ind]
            self.observation['enemy_cities']['pose'][city_bat_ind] = en_cities_bat.pose
            self.observation['enemy_cities']['health'][city_bat_ind] = en_cities_bat.health
            self.observation['enemy_cities']['missiles']['pose'][city_bat_ind] = en_cities_bat.missiles.pose
            self.observation['enemy_cities']['missiles']['health'][city_bat_ind] = en_cities_bat.missiles.health
            self.observation['enemy_cities']['missiles']['launch'][city_bat_ind] = en_cities_bat.missiles.launch
            self.observation['enemy_cities']['missiles']['enemy_tar'][city_bat_ind] = en_cities_bat.missiles.enemy_tar
            self.observation['enemy_cities']['missiles']['enemy_atc'][city_bat_ind] = en_cities_bat.missiles.enemy_atc
            # self.observation['enemy_cities']['missiles']['attacker'][city_bat_ind] = en_cities_bat.missiles.targets

            # self.observation['enemy_cities']['missiles']['attackers']['fr_bats'][city_bat_ind] = en_cities_bat.missiles.attackers['fr_bats']
            # self.observation['enemy_cities']['missiles']['attackers']['fr_missiles'][city_bat_ind] = en_cities_bat.missiles.attackers['fr_missiles']

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
        bat_pose_x = np.random.uniform(CONFIG.ENNEMY_BATTERY.INIT_POS_RANGE[0] * CONFIG.WIDTH,
                                       CONFIG.ENNEMY_BATTERY.INIT_POS_RANGE[1] * CONFIG.WIDTH,
                                       size=CONFIG.ENNEMY_BATTERY.NUMBER)
        bat_pose_y = np.random.uniform(CONFIG.ENNEMY_BATTERY.INIT_HEIGHT_RANGE[0] * CONFIG.WIDTH,
                                       CONFIG.ENNEMY_BATTERY.INIT_HEIGHT_RANGE[1] * CONFIG.WIDTH,
                                       size=CONFIG.ENNEMY_BATTERY.NUMBER)
        self.enemy_batteries = [ EnemyBattery(pose=[bat_pose_x[ind], bat_pose_y[ind], CONFIG.ENNEMY_BATTERY.SPEED, CONFIG.ENNEMY_BATTERY.LAUNCH_THETA],
                         health=1.0, missiles=CONFIG.ENEMY_MISSILES.NUMBER) for ind in range(CONFIG.ENNEMY_BATTERY.NUMBER)]

        # Friendly bateries and missiles
        Init_Height = np.array(CONFIG.FRIENDLY_BATTERY.INIT_HEIGHT_RANGE) * CONFIG.HEIGHT
        Init_Pose = np.array(CONFIG.FRIENDLY_BATTERY.INIT_POS_RANGE) * CONFIG.WIDTH
        bat_pose_x = np.random.uniform(Init_Pose[0], Init_Pose[1], CONFIG.FRIENDLY_BATTERY.NUMBER)
        bat_pose_y = np.random.uniform(Init_Height[0], Init_Height[1], CONFIG.FRIENDLY_BATTERY.NUMBER)

        self.friendly_batteries = [ FriendlyBattery(pose=[bat_pose_x[ind], bat_pose_y[ind], CONFIG.FRIENDLY_BATTERY.SPEED, CONFIG.FRIENDLY_BATTERY.LAUNCH_THETA], health=1.0,
                            missiles=CONFIG.FRIENDLY_MISSILES.NUMBER) for ind in range(CONFIG.FRIENDLY_BATTERY.NUMBER)]
        # self.target = Target()

        # Enemy cities
        Init_Height = np.array(CONFIG.ENNEMY_CITIES_BATTERY.INIT_HEIGHT_RANGE) * CONFIG.HEIGHT
        Init_Pose = np.array(CONFIG.ENNEMY_CITIES_BATTERY.INIT_POS_RANGE) * CONFIG.WIDTH
        if CONFIG.ENNEMY_CITIES_BATTERY.NUMBER == 1:
            bat_pose_x, bat_pose_y = [Init_Pose[0]], [Init_Height[0]]
        else:
            bat_pose_x = np.random.uniform(Init_Pose[0], Init_Pose[1], CONFIG.ENNEMY_CITIES_BATTERY.NUMBER)
            bat_pose_y = np.random.uniform(Init_Height[0], Init_Height[1], CONFIG.ENNEMY_CITIES_BATTERY.NUMBER)

        self.enemy_cities = [ CityBattery(pose=[bat_pose_x[ind], bat_pose_y[ind], CONFIG.ENNEMY_CITIES_BATTERY.SPEED, CONFIG.ENNEMY_CITIES_BATTERY.LAUNCH_THETA], health=1.0,
                        launch = 1, missiles=CONFIG.ENNEMY_CITIES.NUMBER) for ind in range(CONFIG.ENNEMY_CITIES_BATTERY.NUMBER)]
        #################################################################################################################

        return self._extract_observation()


    def step(self, action):
        """Go from current step to next one.

        Args:
            action (int): 0, 1, 2, 3, 4 or 9, the different actions, per friendly N missiles - total [N,10]

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

        for fr_bat_id in range(CONFIG.FRIENDLY_BATTERY.NUMBER):
            bat_act = 1
            if bat_act ==1:
                friend_observation, friendly_battery_reward, done_friendly_battery, info = \
                    self.friendly_batteries[fr_bat_id].step(action['friends'], fr_bat_id, self.observation['friends_bat'])

        # en_bat_active = action['enemies']['batteries']
        for en_bat_id in range(CONFIG.ENNEMY_BATTERY.NUMBER):
            bat_act = 1
            if bat_act == 1:
                enemy_observation, enemy_battery_reward, done_enemy_battery, info = \
                    self.enemy_batteries[en_bat_id].step(action['enemies'], en_bat_id, self.observation['enemy_bat'])

        for city_bat_id in range(CONFIG.ENNEMY_CITIES_BATTERY.NUMBER):
            bat_act = 1
            if bat_act == 1:
                enemy_cities_observation, enemy_cities_reward, done_enemy_cities, info  = \
                    self.enemy_cities[city_bat_id].step(action['cities'], city_bat_id, self.observation['enemy_cities'])

        # _, battery_reward, _, can_fire_dict = self.batteries.step(action)
        # _, _, done_cities, _ = self.cities.step(action)
        # _, _, done_enemy_missiles, _ = self.enemy_missiles.step(action)
        # _, _, _, _ = self.friendly_missiles.step(action)
        # _, _, _, _ = self.target.step(action)

        print('enemy',self.observation['enemy_bat']['missiles']['health'])
        print('friend', self.observation['friends_bat']['missiles']['health'])



        # Check for collisions
        # ------------------------------------------

        # self._collisions_missiles()
        # self._collisions_cities()

        # Check if episode is finished
        # ------------------------------------------

        done = False #done_enemy_cities or done_enemy_batteries

        # Render every objects
        # ------------------------------------------

        self.observation['sensors']['vision'] = \
            self._compute_sensor_observation('vision')

        # Return everything
        # ------------------------------------------

        self.timestep += 1
        self.reward_total += friendly_battery_reward + enemy_battery_reward + enemy_cities_reward + self.reward_timestep
        self.observation['sensors']['vision'] = self._process_observation()
        return self.observation, self.reward_timestep, done, {}

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
            obs = self.observation['sensors']['vision'] #np.array(Image.fromarray(self.observation['sensors']['vision']))
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
        pygame.display.flip() #update()

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