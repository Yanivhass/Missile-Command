
import cv2
import numpy as np

from config import CONFIG
from utils import get_cv2_xy

class Missile:

    def __init__(self, pose, health):
        self.pose = pose  # [x,y,velocity,heading angle in degrees]
        self.health = health
        self.launched = False
        self.target = []
        self.target_xy = np.zeros(2, 1)
        self.guided = False

    def reset(self, pose, health):
        self.pose = pose
        self.health = health
        self.launched = False

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

    def launch(self, target, speed):
        if self.launched is False:
            self.launched = True
            self.pose[2] = speed
            self.target = target


class City:

    def __init__(self, id, pose, health):
        # self.pose = np.tile(pose, (number, 1))
        self.id = id
        self.pose = pose
        self.health = health

    def reset(self, pose, health):
        self.pose = pose
        self.health = health

    def step(self):
        None


class Unit:
    """
    Either a battery or an aircraft.

    """
    # NB_BATTERIES = 1
    MAX_HEALTH = 1.0

    def __init__(self, id, pose, health, missiles_count):
        """Initialize EnemyBatteriy battery.

        Attributes:

        """
        self.id = id
        self.pose = pose
        self.health = health
        self.missiles_number = missiles_count

        self.missiles = [Missile(
            pose=pose,  # All missiles are initialized in the unit's magazine
            health=1.0)
            for ind in missiles_count]

    def reset(self, pose, health, missiles_count):
        self.pose = pose
        self.health = health
        self.missiles_number = missiles_count

        self.missiles = [Missile(
            pose=pose,  # All missiles are initialized in the unit's magazine
            health=1.0)
            for ind in missiles_count]

    def step(self, action, unit_id, observation):
        """Go from current step to next one.

        The missile launched is done in the main environment class.

        Args:
            action:
            movement - [-1 = left, 0 = straight, 1 = right]
            launch - binary array of missiles to launch
            targets - array, target IDs for each missile

        """

        ########## THIS IS DONE EXTERNALLY _ SO THAT BATTERY REMAINS THE SAME #################
        ####### update action command !!!!!!!!! (as enemy works without control #######################
        ####### compute friendly targets ###############################
        ####### update launch missiles commands, and dvels #############

        ########## update battery pose and non launched missiles #########
        angle = self.pose[3] * np.pi / 180.0
        delta_pose = self.pose[2] * np.array([np.cos(angle), np.sin(angle)])
        self.pose[0:2] += delta_pose

        ################# update launched missiles ########################
        ####### find launch n lauch - per batery id #######################
        # launch_indices = np.where( np.logical_and((action['missiles']['launch'][bat_id] == 1), (self.missiles.launch == False)) == True)
        new_launch_indices = np.where(action['launch'] == 1)
        self.missiles[new_launch_indices].launch = 1
        launch_indices = np.where(self.missiles.launch == 1)
        ######### update observation #######################
        observation['missiles']['launch'][bat_id, new_launch_indices] = 1

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
        speed = self.missiles.pose[launch_indices, 2][0]  #= CONFIG.ENEMY_MISSILES.SPEED
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
