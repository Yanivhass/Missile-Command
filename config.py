"""Environment configuration."""

from dataclasses import dataclass

import numpy as np

import config


@dataclass
class CONFIG:
    """Configuration class.

    Attributes:
        DTYPE (numpy.dtype): numpy arrays type.
        FPS (int): limit FPS for rendering only.
        HEIGHT (int): environment height.
        WIDTH (int): environment width.
    """
    IMAGE_DTYPE: np.dtype = np.uint8
    FPS: int = 10  # 144
    HEIGHT: int = 2000
    WIDTH: int = 2000
    SCREEN_HEIGHT: int = 1000
    SCREEN_WIDTH: int = 1000
    SEED = 42
    SPEED_MODIFIER = 2
    BACK_COLOR: tuple = (0, 128, 0)  # Green

    @dataclass
    class DEFENDERS:
        """Anti-missiles battery configuration.

        Attributes:
            RADIUS (float): radius of the anti-missile battery object.
        """
        QUANTITY: int = 2  # number of entities
        RADIUS: int = 20
        RANGE: int = 300
        MAX_HEALTH: float = 1.0
        MISSILES_PER_UNIT = 4

        INIT_HEIGHT_RANGE = [0.4, 0.6]
        INIT_POS_RANGE = [0.5, 0.6]
        SPEED: float = 0.0 * 2
        # LAUNCH_VEL = [0.0, 1.0]
        LAUNCH_THETA: float = 90

        MAX_LAUNCH: int = 2
        DLaunch_Time: int = 4

        DETECTION_RANGE = [100, 500]
        KILL_RANGE = 800

        COLOR: tuple = (0, 0, 255)
        SIZE: int = 10
        WIDTH: int = 70
        HEIGHT: int = 70

        @dataclass
        class MISSILES:
            RADIUS: int = 10
            EXPLOSION_RADIUS: float = 30
            PROBA_IN: float = 0.005
            SPEED: float = 21.0 * 2
            MAX_HEALTH = 1.0

            NB_ACTIONS: int = 9
            LAUNCH_THETA: float = 90
            DTHETA = np.arange(-10, 25, 5)
            FUEL: int = 15
            GUIDED: bool = True

            COLOR: tuple = (255, 255, 255)
            SIZE: int = 10
            WIDTH: int = 20
            HEIGHT: int = 20

    @dataclass
    class ATTACKERS:
        """Atack-missiles battery configuration.

        Attributes:
            RADIUS (float): radius of the anti-missile battery object.
        """
        QUANTITY: int = 2
        RADIUS: int = 20
        RANGE: int = 250
        MAX_HEALTH: float = 1
        MISSILES_PER_UNIT: int = 2
        FUEL: int = 250

        INIT_HEIGHT_RANGE = [0.4, 0.6]  # [0.1, 0.9] #
        INIT_POS_RANGE = [0.1, 0.3]  # [0.1, 0.9] #
        SPEED: float = 20.0 * 2
        # LAUNCH_VEL = [0.0, 1.0]
        LAUNCH_THETA: float = 0
        MAX_LAUNCH: int = 4
        DLaunch_Time: int = 8

        DETECTION_RANGE = [200.0, 2000.0]
        KILL_RANGE = 1000.0

        COLOR: tuple = (255, 0, 0)
        SIZE: int = 10
        WIDTH: int = 100
        HEIGHT: int = 100

        @dataclass
        class MISSILES:
            RADIUS: int = 10
            EXPLOSION_RADIUS: float = 50
            PROBA_IN: float = 0.005
            MAX_HEALTH = 1
            SPEED: float = 20.0 * 2
            LAUNCH_THETA: float = 0
            DTHETA = np.arange(-10, 25, 5)
            FUEL: int = 200
            GUIDED: bool = False

            COLOR: tuple = (255, 255, 255)
            SIZE: int = 10
            WIDTH: int = 40
            HEIGHT: int = 40

    @dataclass
    class CITIES:
        """Cities configuration.

        Attributes:
            NUMBER (int): number of cities to defend (even integer >= 2).
            RADIUS (float): radius of a city object.
        """

        QUANTITY: int = 2
        RADIUS: int = 30
        # RANGE: float = 466.0
        MAX_HEALTH: float = 1

        INIT_HEIGHT_RANGE = [0.4, 0.6]
        INIT_POS_RANGE = [0.7, 0.8]
        SPEED: float = 0.0 * 3
        # LAUNCH_VEL = [0.0, 1.0]
        LAUNCH_THETA: float = 0

        MAX_LAUNCH: int = 4

        DLaunch_Time: int = 5

        COLOR: tuple = (0, 255, 0)
        SIZE: int = 10
        WIDTH: int = 100
        HEIGHT: int = 100

    #
    # class ENNEMY_CITIES():
    #     """Cities configuration.
    #
    #     Attributes:
    #         NUMBER (int): number of cities to defend (even integer >= 2).
    #         RADIUS (float): radius of a city object.
    #     """
    #     NUMBER: int = 10
    #     RADIUS: float = 5.0
    #     MAX_HEALTH = 1.0
    #
    #     INIT_HEIGHT_RANGE = [0.1, 0.2]
    #     INIT_POS_RANGE = [0.3, 1.0]
    #     SPEED: float = 0.0
    #     # LAUNCH_VEL = [0.0, 1.0]
    #     LAUNCH_THETA: float = 0
    #     MAX_LAUNCH: int = 4
    #     DLaunch_Time: int = 10

    # @dataclass
    # class BLUE_MISSILES():
    #     """Enemy missiles configuration.
    #
    #     Attributes:
    #         NUMBER (int): total number of enemy missiles for 1 episode.
    #         EXPLOSION_RADIUS (float): radius of target hit
    #         RADIUS (float): radius of an enemy missile object.
    #         SPEED (float): enemy missile speed.
    #     """
    #     NUMBER: int = 2
    #     RADIUS: float = 10.0
    #     EXPLOSION_RADIUS: float = 30
    #     PROBA_IN: float = 0.005
    #     SPEED: float = 40.0
    #     MAX_HEALTH = 1.0
    #
    #     NB_ACTIONS: int = 9
    #     LAUNCH_THETA: float = 90
    #     DTHETA = np.arange(-10, 25, 5)

    # @dataclass
    # class RED_MISSILES():
    #     """Friendly missiles configuration.
    #
    #     Attributes:
    #         NUMBER (int): total number of available friendly missiles.
    #         EXPLOSION_RADIUS (float): maximum explosion radius.
    #         EXPLOSION_SPEED (float): speed of the explosion.
    #         RADIUS (float): radius of a friendly missile object.
    #         SPEED (float): friendly missile speed.
    #     """
    #     NUMBER: int = 10
    #     RADIUS: float = 10.0
    #     EXPLOSION_RADIUS: float = 50
    #     PROBA_IN: float = 0.005
    #     MAX_HEALTH = 1
    #
    #     NB_ACTIONS: int = 9
    #     SPEED: float = 20.0
    #     LAUNCH_THETA: float = 0
    #     DTHETA = np.arange(-10, 25, 5)

    @dataclass
    class COLORS():
        """Colors configuration.

        Attributes:
            BACKGROUND (tuple): #000000.
            BATTERY (tuple): #ffffff.
            CITY (tuple): #0000ff.
            ENEMY_MISSILE (tuple): #ff0000.
            EXPLOSION (tuple): #ffff00.
            FRIENDLY_MISSILE (tuple): #00ff00.
            TARGET (tuple): #ffffff.
        """
        BACKGROUND: tuple = (0, 128, 0)
        BLUE_TEAM: tuple = (255, 255, 255)
        BLUE_TEAM_TARGETING_RANGE: tuple = (125, 255, 125)
        BLUE_TEAM_MISSILE: tuple = (255, 0, 0)
        BLUE_TEAM_BATTERY: tuple = (0, 0, 0)
        BLUE_TEAM_CITY: tuple = (0, 0, 255)
        RED_TEAM_ENTITY: tuple = (0, 255, 0)
        RED_TEAM_MISSILE: tuple = (0, 125, 125)
        EXPLOSION: tuple = (255, 255, 0)
        TARGET: tuple = (255, 255, 255)

    @dataclass
    class OBSERVATION():
        """Observation configuration.

        An agent takes as input the screen pixels. The resolution of the
        environment can be quite big: the computational cost to train an agent
        can then be high.

        For an agent to well perform on the Missile Command Atari game, a
        smaller resized version of the observation can be enough.

        So the environment returns a resized version of the environment
        observation.

        If you wish to not resize the observation, fix these variables to the
        same values as CONFIG.HEIGHT and CONFIG.WIDTH.

        Attributes:
            HEIGHT (float): observation height.
            RENDER_PROCESSED_HEIGHT (float): render window height of the
                processed observation.
            RENDER_PROCESSED_WIDTH (float): render window width of the
                processed observation.
            WIDTH (float): observation width.
        """
        HEIGHT: float = 1000
        WIDTH: float = 1000
        RENDER_PROCESSED_HEIGHT = 1000
        RENDER_PROCESSED_WIDTH = 1000

    @dataclass
    class REWARD():
        """Reward configuration.

        Attributes:
            DESTROYED_ENNEMY_CITY (float): reward for each destroyed city.
            DESTROYED_ENNEMY_BATTERY (float): reward for each destroyed enemy battery.
            DESTROYED_ENEMEY_MISSILES (float): reward for each destroyed ennemy missile.
            DESTROYED_FRIENDLY_BATTERY (float): reward for each destroyed friendly battery.
            DESTROYED_FRIENDLY_MISSILES (float): reward for each destroyed friendly missile.
            FRIENDLY_MISSILE_LAUNCHED (float); reward for each friendly missile launched.
            ENEMEY_MISSILE_LAUNCHED (float); reward for each ennemy missile launched.
            ENNEMY_TARGET_MISSED (float); reward for each friendly missile target miss.
        """
        DESTROYED_CITY: float = +100.0
        DESTROYED_AA_BATTERY: float = +10.0
        DESTROYED_BOMBER: float = 0.0
        # DESTROYED_MISSILES: float = + 15.0
        MISSILE_LAUNCHED: float = 1.0

    # @dataclass
    # class TARGET():
    #     """Target configuration.

    #
    #     Attributes:
    #         SIZE_(int): target size (only for render).
    #         VX (int): horizontal shifting of the target.
    #         VY (int): vertical shifting of the target.
    #     """
    #     SIZE: int = 12
    #     VX: int = 4
    #     VY: int = 4
