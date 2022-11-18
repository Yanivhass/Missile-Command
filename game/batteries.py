
"""Anti-missiles batteries."""

import cv2
import numpy as np
from PIL import Image

from config import CONFIG
from game.missiles import EnemyMissiles, FriendlyMissiles, EnemyCities
from utils import get_cv2_xy

############ dvel -15:15 deg in interval of 5 deg ##################################
# abs_vel = 5.0
# dvel_theta = np.arange(-20, 25, 5)
# dvel = abs_vel*np.array([np.cos(dvel_theta), np.sin(dvel_theta)]).transpose()
# fr_dvel = np.vstack((CONFIG.FRIENDLY_MISSILES.SPEED*np.cos(CONFIG.FRIENDLY_MISSILES.DTHETA),
#                     CONFIG.FRIENDLY_MISSILES.SPEED*np.sin(CONFIG.FRIENDLY_MISSILES.DTHETA))).transpose()
# en_dvel = np.vstack((CONFIG.ENEMY_MISSILES.SPEED*np.cos(CONFIG.ENEMY_MISSILES.DTHETA),
#                      CONFIG.ENEMY_MISSILES.SPEED*np.sin(CONFIG.ENEMY_MISSILES.DTHETA))).transpose()

class CityBattery(EnemyCities):
    """Anti-missiles batteries class.

    Attributes:
        NB_BATTERY (int): the number of batteries. It only supports 1 battery.
    """
    # NB_BATTERIES = 1
    MAX_HEALTH = 1.0

    def __init__(self, pose=(CONFIG.FRIENDLY_BATTERY.INIT_POS_RANGE[0], CONFIG.FRIENDLY_BATTERY.INIT_HEIGHT_RANGE[0],
                             CONFIG.FRIENDLY_BATTERY.SPEED, CONFIG.FRIENDLY_BATTERY.LAUNCH_THETA),
                 launch = 1, health=CONFIG.FRIENDLY_BATTERY.MAX_HEALTH, missiles=CONFIG.FRIENDLY_MISSILES.NUMBER):
        """Initialize EnemyBatteriy battery.

        Attributes:

        """

        self.pose = pose
        self.health = health
        self.missiles_number = missiles

        ################## init missiles ##############################
        missiles_pose = np.zeros((missiles, 4))
        missiles_pose[:] = pose

        Init_Height = np.array(CONFIG.ENNEMY_CITIES.INIT_HEIGHT_RANGE) * CONFIG.HEIGHT
        Init_Pose = np.array(CONFIG.ENNEMY_CITIES.INIT_POS_RANGE) * CONFIG.WIDTH
        if CONFIG.ENNEMY_CITIES.NUMBER == 1:
            miss_pose_x, miss_pose_y = [Init_Pose[0]], [Init_Height[0]]
        else:
            miss_pose_x = np.random.uniform(Init_Pose[0], Init_Pose[1], CONFIG.ENNEMY_CITIES.NUMBER)
            miss_pose_y = np.random.uniform(Init_Height[0], Init_Height[1], CONFIG.ENNEMY_CITIES.NUMBER)
        missiles_pose[:, 0] = miss_pose_x
        missiles_pose[:, 1] = miss_pose_y


        self.missiles = EnemyCities(number=missiles, pose=missiles_pose, health=self.MAX_HEALTH, launch = launch)

    def reset(self, pose=(CONFIG.FRIENDLY_BATTERY.INIT_POS_RANGE[0], CONFIG.FRIENDLY_BATTERY.INIT_HEIGHT_RANGE[0],
                          CONFIG.FRIENDLY_BATTERY.SPEED, CONFIG.FRIENDLY_BATTERY.LAUNCH_THETA),
              health=CONFIG.FRIENDLY_BATTERY.MAX_HEALTH):
        """Reset batteries.

        Total number of missiles is reset to default.

        Warning:
            To fully initialize a Batteries object, init function and reset
            function musts be called.
        """
        self.pose = pose
        self.health = health

        missiles_pose = np.zeros(self.missiles_number, 4)
        missiles_pose[:] = pose

        Init_Height = np.array(CONFIG.ENNEMY_CITIES.INIT_HEIGHT_RANGE) * CONFIG.HEIGHT
        Init_Pose = np.array(CONFIG.ENNEMY_CITIES.INIT_POS_RANGE) * CONFIG.WIDTH
        if CONFIG.ENNEMY_CITIES.NUMBER == 1:
            miss_pose_x, miss_pose_y = [Init_Pose[0]], [Init_Height[0]]
        else:
            miss_pose_x = np.random.uniform(Init_Pose[0], Init_Pose[1], CONFIG.ENNEMY_CITIES.NUMBER)
            miss_pose_y = np.random.uniform(Init_Height[0], Init_Height[1], CONFIG.ENNEMY_CITIES.NUMBER)
        missiles_pose[:, 0] = miss_pose_x
        missiles_pose[:, 1] = miss_pose_y

        self.missiles.reset(pose=missiles_pose)

    def step(self, self_action, bat_id, self_observation):

        total_reward, done, info = 0, False, None

        return self_observation, total_reward, done, info

    # self_observation, total_reward, done, info

    def render(self, observation, health):
        """Render anti-missiles batteries.

        Todo:
            Include the number of available missiles.

        Args:
            observation (numpy.array): the current environment observation
                representing the pixels. See the object description in the main
                environment class for information.
        """

        ########### render battery #####################################
        # cv2.circle(
        #     img=observation,
        #     center=(self.pose[0], self.pose[1]),
        #     radius=int(CONFIG.ENNEMY_CITIES_BATTERY.RADIUS),
        #     color=CONFIG.COLORS.ENNEMY_CITY_BATTERY,
        #     thickness=-1,
        # )

        ########### render missiles #####################################
        for ms_id in range(len(health)):
            if health[ms_id] == 1: #self.missiles
                cv2.circle(
                    img=observation,
                    center=(int(self.missiles.pose[ms_id, 0]), int(self.missiles.pose[ms_id, 1])),
                    radius=int(CONFIG.ENNEMY_CITIES.RADIUS),
                    color=CONFIG.COLORS.ENNEMY_CITY,
                    thickness=-1,
                )
            else:
                continue

        # for x, y, integrity in zip(self.batteries_pos[:, 0],
        #                            self.batteries_pos[:, 1],
        #                            self.batteries_pos[:, 2]):
        #     if integrity > 0:
        #
        #         cv2.circle(
        #             img=observation,
        #             center=(get_cv2_xy(self.batteries_pos[i,0], 0.0)),
        #             radius=int(CONFIG.BATTERY.RADIUS),
        #             color=CONFIG.COLORS.BATTERY,
        #             thickness=-1,
        #         )
        # color = CONFIG.COLORS.EXPLOSION,


class FriendlyBattery(FriendlyMissiles):
    """Anti-missiles batteries class.

    Attributes:
        NB_BATTERY (int): the number of batteries. It only supports 1 battery.
    """
    # NB_BATTERIES = 1
    MAX_HEALTH = 1.0

    def __init__(self, pose=(CONFIG.FRIENDLY_BATTERY.INIT_POS_RANGE[0], CONFIG.FRIENDLY_BATTERY.INIT_HEIGHT_RANGE[0],CONFIG.FRIENDLY_BATTERY.SPEED, CONFIG.FRIENDLY_BATTERY.LAUNCH_THETA),
                 health = CONFIG.FRIENDLY_BATTERY.MAX_HEALTH, missiles = CONFIG.FRIENDLY_MISSILES.NUMBER):
        """Initialize EnemyBatteriy battery.

        Attributes:

        """
        self.pose = pose
        self.health = health
        self.missiles_number = missiles

        missiles_pose =  np.tile(pose, (missiles, 1))

        self.missiles = FriendlyMissiles(number=missiles, pose=missiles_pose, health=self.MAX_HEALTH)


    def reset(self, pose=(CONFIG.FRIENDLY_BATTERY.INIT_POS_RANGE[0], CONFIG.FRIENDLY_BATTERY.INIT_HEIGHT_RANGE[0], CONFIG.FRIENDLY_BATTERY.SPEED, CONFIG.FRIENDLY_BATTERY.LAUNCH_THETA),
              health = CONFIG.FRIENDLY_BATTERY.MAX_HEALTH):
        """Reset batteries.

        Total number of missiles is reset to default.

        Warning:
            To fully initialize a Batteries object, init function and reset
            function musts be called.
        """
        self.pose = pose
        self.health = health

        missiles_pose =  np.tile(pose, (self.missiles_number, 1))

        self.missiles.reset(pose = missiles_pose, health = health)

    def step(self, action, bat_id, self_observation):
        """Go from current step to next one.

        The missile launched is done in the main environment class.

        Args:
            action (int): (0) do nothing, (1) fire missile,
            (2) target up, (3) target down, (4), target left, (5) target right,
            (6) target_left_up, (7) target_left_down, (8) target_right_up, (9) target_right_down

        Returns:
            observation: None.

            reward: None.

            done: None.

            info (dict): additional information of the current time step. It
                contains key "can_fire" with associated value "True" if the
                anti-missile battery can fire a missile and "False" otherwise.
        """
        ########## update battery pose and non launched missiles #########
        angle = self.pose[3] * np.pi / 180.0
        bat_dpos = self.pose[2] * np.array([np.cos(angle), np.sin(angle)])
        self.pose[0:2] += bat_dpos

        ################# update launched missiles ########################
        ####### find launch n lauch - per batery id #######################
        # launch_indices = np.where( np.logical_and((action['missiles']['launch'][bat_id] == 1), (self.missiles.launch == False)) == True)
        new_launch_indices = np.where(action['missiles']['launch'][bat_id] == 1)
        self.missiles.launch[new_launch_indices] = 1
        launch_indices = np.where(self.missiles.launch == 1)
        ######### update observation #######################
        self_observation['missiles']['launch'][bat_id, new_launch_indices] = 1

        ######### update targets ##########################
        self.missiles.targets['missiles'][launch_indices] = action['missiles']['targets'][bat_id][launch_indices]


        ######### update pose of non-launched missiles #######################
        non_launched_indices = np.where(self.missiles.launch == 0)
        self.missiles.pose[non_launched_indices, 0:2] += bat_dpos
        self_observation['missiles']['pose'][bat_id, non_launched_indices, 0:2] = self.missiles.pose[
                                                                                  non_launched_indices, 0:2]

        ####### update vel and new pose - of launched missiles  ###################
        # speed = self.missiles.pose[launch_indices, 2][0]
        # speed[:] = CONFIG.FRIENDLY_MISSILES.SPEED
        speed = self.missiles.pose[launch_indices, 2][0] = CONFIG.FRIENDLY_MISSILES.SPEED
        delta_angles = CONFIG.FRIENDLY_MISSILES.DTHETA[(action['missiles']['actions'][bat_id][launch_indices])]
        self.missiles.pose[launch_indices, 3] += delta_angles

        pose = self.missiles.pose[launch_indices, 3]
        if len(np.where(np.abs(pose > 180))[0]) > 0:
            t = 1
        # pose[pose > 180] -= 360
        # pose[pose< -180] += 360


        angles = self.missiles.pose[launch_indices, 3][0] * np.pi / 180.0
        miss_dpos = (speed * np.array([np.cos(angles), np.sin(angles)])).transpose()
        self.missiles.pose[launch_indices, 0:2] += miss_dpos
        kill_ind = (np.where(self.missiles.pose[launch_indices][:, 1] < 5))
        if len(kill_ind[0]) > 0:
            self.missiles.health[kill_ind] = 0

        ######### update observation #######################
        self_observation['missiles']['pose'][bat_id, launch_indices] = self.missiles.pose[launch_indices]
        # observation['friends_bat']['missiles']['vel'][bat_id, launch_indices, :] = self.missiles.pose[launch_indices, 2:4]
        if len(kill_ind[0]) > 0:
            self_observation['missiles']['health'][bat_id, kill_ind] = self.missiles.health[kill_ind]

        ####### compute rewards - per batery id  ###########################
        # enemy bat and enemy cities rewards !!!!!!!!!!
        # en_bats, en_cities = observation['enemy_bat'], observation['enemy_cities']
        total_reward = 0

        #######  check done task ###########################################
        done = False

        ####### update observation - per batery id ########

        # bat_state, bat_observation, bat_done = []
        # bat_reward = 0
        # for ind in range(CONFIG.FRIENDLY_MISSILES.NUMBER):
        #     action = [action[0][ind], action[1][ind]]
        #     state, observation, reward, done = self.missiles[ind].step(action)
        #     bat_state.append(state), bat_observation.append(observation), bat_done.append(done)
        #     bat_reward += reward

        info = None

        return self_observation, total_reward, done, info

    def render(self, observation, health):
        """Render anti-missiles batteries.

        Todo:
            Include the number of available missiles.

        Args:
            observation (numpy.array): the current environment observation
                representing the pixels. See the object description in the main
                environment class for information.
        """

        ########### render battery #####################################
        cv2.circle(
            img=observation,
            center=(int(self.pose[0]), int(self.pose[1])),
            radius=int(CONFIG.FRIENDLY_BATTERY.RADIUS),
            color=CONFIG.COLORS.FRIENDLY_BATTERY,
            thickness=-1,
        )
        cv2.circle(
            img=observation,
            center=(int(self.pose[0]), int(self.pose[1])),
            radius=int(CONFIG.FRIENDLY_BATTERY.DETECTION_RANGE[1]),
            color=CONFIG.COLORS.FRIENDLY_BATTERY,
            thickness=3,
        )

        ########### render missiles #####################################
        for ms_id in range(len(health)):
            if health[ms_id] == 1: #self.missiles
                cv2.circle(
                    img=observation,
                    center=(int(self.missiles.pose[ms_id, 0]), int(self.missiles.pose[ms_id, 1])),
                    radius=int(CONFIG.FRIENDLY_MISSILES.RADIUS),
                    color=CONFIG.COLORS.FRIENDLY_MISSILE,
                    thickness=-1,
                )
            else:
                continue

        # for x, y, integrity in zip(self.batteries_pos[:, 0],
        #                            self.batteries_pos[:, 1],
        #                            self.batteries_pos[:, 2]):
        #     if integrity > 0:
        #
        #         cv2.circle(
        #             img=observation,
        #             center=(get_cv2_xy(self.batteries_pos[i,0], 0.0)),
        #             radius=int(CONFIG.BATTERY.RADIUS),
        #             color=CONFIG.COLORS.BATTERY,
        #             thickness=-1,
        #         )
        # color = CONFIG.COLORS.EXPLOSION,


class EnemyBattery():
    """Anti-missiles batteries class.

    Attributes:
        NB_BATTERY (int): the number of batteries. It only supports 1 battery.
    """
    # NB_BATTERIES = 1
    MAX_HEALTH = 1.0

    def __init__(self, pose = (CONFIG.ENNEMY_BATTERY.INIT_POS_RANGE[0]*CONFIG.WIDTH, 0.0, CONFIG.ENNEMY_BATTERY.SPEED, CONFIG.ENNEMY_BATTERY.LAUNCH_THETA),
                 health = CONFIG.ENNEMY_BATTERY.MAX_HEALTH, missiles = CONFIG.ENEMY_MISSILES.NUMBER):
        """Initialize EnemyBatteriy battery.

        Attributes:

        """
        self.pose = pose
        self.health = health
        self.missiles_number = missiles

        missiles_pose = np.tile(pose, (missiles, 1))


        self.missiles = EnemyMissiles(pose=missiles_pose, health=self.health, number = missiles)

    def reset(self, pose = (CONFIG.ENNEMY_BATTERY.INIT_POS_RANGE[0]*CONFIG.WIDTH, 0.0, CONFIG.ENNEMY_BATTERY.SPEED, CONFIG.ENNEMY_BATTERY.LAUNCH_THETA),
              health=CONFIG.ENNEMY_BATTERY.MAX_HEALTH):

        """Reset batteries.

        Total number of missiles is reset to default.

        Warning:
            To fully initialize a Batteries object, init function and reset
            function musts be called.
        """
        self.pose = pose
        self.health = health

        missiles_pose = np.tile(pose, (self.missiles_number, 1))

        self.missiles.reset(pose=missiles_pose, health=health)


    def step(self, action, bat_id, self_observation):
        """Go from current step to next one.

        The missile launched is done in the main environment class.

        Args:
            action (int): (0) do nothing, (1) fire missile,
            (2) target up, (3) target down, (4), target left, (5) target right,
            (6) target_left_up, (7) target_left_down, (8) target_right_up, (9) target_right_down

        Returns:
            observation: None.

            reward: None.

            done: None.

            info (dict): additional information of the current time step. It
                contains key "can_fire" with associated value "True" if the
                anti-missile battery can fire a missile and "False" otherwise.
        """

        ########## THIS IS DONE EXTERNALLY _ SO THAT BATTERY REMAINS THE SAME #################
        ####### update action command !!!!!!!!! (as enemy works without control #######################
        ####### compute friendly targets ###############################
        ####### update launch missiles commands, and dvels #############

        ########## update battery pose and non launched missiles #########
        angle = self.pose[3] * np.pi / 180.0
        bat_dpos = self.pose[2] * np.array([np.cos(angle), np.sin(angle)])
        self.pose[0:2] += bat_dpos

        ################# update launched missiles ########################
        ####### find launch n lauch - per batery id #######################
        # launch_indices = np.where( np.logical_and((action['missiles']['launch'][bat_id] == 1), (self.missiles.launch == False)) == True)
        new_launch_indices = np.where(action['missiles']['launch'][bat_id] == 1)
        self.missiles.launch[new_launch_indices] = 1
        launch_indices = np.where(self.missiles.launch == 1)
        ######### update observation #######################
        self_observation['missiles']['launch'][bat_id, new_launch_indices] = 1

        ######### update targets ##########################
        self.missiles.targets['missiles'][launch_indices] = action['missiles']['targets'][bat_id][
            launch_indices]

        ######### update pose of non-launched missiles #######################
        non_launched_indices = np.where(self.missiles.launch == 0)
        self.missiles.pose[non_launched_indices, 0:2] += bat_dpos
        self_observation['missiles']['pose'][bat_id, non_launched_indices, 0:2] = self.missiles.pose[
                                                                                            non_launched_indices, 0:2]

        ####### update vel and new pose - of launched missiles  ###################
        # speed = self.missiles.pose[launch_indices, 2][0]
        # speed[:] = CONFIG.FRIENDLY_MISSILES.SPEED
        speed = self.missiles.pose[launch_indices, 2][0] = CONFIG.ENEMY_MISSILES.SPEED
        delta_angles = CONFIG.ENEMY_MISSILES.DTHETA[(action['missiles']['actions'][bat_id][launch_indices])]
        self.missiles.pose[launch_indices, 3] += delta_angles

        pose = self.missiles.pose[launch_indices, 3]
        if len(np.where(np.abs(pose>180))[0]) > 0:
            t = 1

        # pose[pose > 180] -= 360
        # pose[pose < -180] += 360

        angles = self.missiles.pose[launch_indices, 3][0] * np.pi / 180.0
        miss_dpos = (speed * np.array([np.cos(angles), np.sin(angles)])).transpose()
        self.missiles.pose[launch_indices, 0:2] += miss_dpos
        kill_ind = (np.where(self.missiles.pose[launch_indices][:, 1] < 5))
        if len(kill_ind[0]) > 0:
            self.missiles.health[kill_ind] = 0

        ######### update observation #######################
        self_observation['missiles']['pose'][bat_id, launch_indices] = self.missiles.pose[launch_indices]
        # observation['friends_bat']['missiles']['vel'][bat_id, launch_indices, :] = self.missiles.pose[launch_indices, 2:4]
        if len(kill_ind[0]) > 0:
            self_observation['missiles']['health'][bat_id, kill_ind] = self.missiles.health[kill_ind]

        ####### compute rewards - per batery id  ###########################
        # enemy bat and enemy cities rewards !!!!!!!!!!
        # en_bats, en_cities = observation['enemy_bat'], observation['enemy_cities']
        total_reward = 0

        #######  check done task ###########################################
        done = False

        ####### update observation - per batery id ########

        # bat_state, bat_observation, bat_done = []
        # bat_reward = 0
        # for ind in range(CONFIG.FRIENDLY_MISSILES.NUMBER):
        #     action = [action[0][ind], action[1][ind]]
        #     state, observation, reward, done = self.missiles[ind].step(action)
        #     bat_state.append(state), bat_observation.append(observation), bat_done.append(done)
        #     bat_reward += reward

        info = None

        return self_observation, total_reward, done, info




    def render(self, observation, health):
        """Render anti-missiles batteries.

        Todo:
            Include the number of available missiles.

        Args:
            observation (numpy.array): the current environment observation
                representing the pixels. See the object description in the main
                environment class for information.
        """

        ########### render battery #####################################
        cv2.circle(
            img=observation,
            center=(int(self.pose[0]), int(self.pose[1])),
            radius=int(CONFIG.ENNEMY_BATTERY.RADIUS),
            color=CONFIG.COLORS.ENNEMY_BATTERY,
            thickness=-1,
        )
        cv2.circle(
            img=observation,
            center=(int(self.pose[0]), int(self.pose[1])),
            radius=int(CONFIG.ENNEMY_BATTERY.DETECTION_RANGE[1]),
            color=CONFIG.COLORS.ENNEMY_BATTERY_RANGE,
            thickness=3,
        )


        ########### render missiles #####################################
        for ms_id in range(len(health)):
            if health[ms_id] == 1: #self.missiles
                cv2.circle(
                    img=observation,
                    center=(int(self.missiles.pose[ms_id, 0]), int(self.missiles.pose[ms_id, 1])),
                    radius=int(CONFIG.ENEMY_MISSILES.RADIUS),
                    color=CONFIG.COLORS.ENEMY_MISSILE,
                    thickness=-1,
                )
            else:
                continue


        # for x, y, integrity in zip(self.batteries_pos[:, 0],
        #                            self.batteries_pos[:, 1],
        #                            self.batteries_pos[:, 2]):
        #     if integrity > 0:
        #
        #         cv2.circle(
        #             img=observation,
        #             center=(get_cv2_xy(self.batteries_pos[i,0], 0.0)),
        #             radius=int(CONFIG.BATTERY.RADIUS),
        #             color=CONFIG.COLORS.BATTERY,
        #             thickness=-1,
        #         )
        # color = CONFIG.COLORS.EXPLOSION,
