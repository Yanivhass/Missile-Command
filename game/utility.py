import cv2
import numpy as np

from config import CONFIG
from game.missiles import EnemyMissiles, FriendlyMissiles
from utils import get_cv2_xy

def compute_bat_missiles_ind(ind, div):
        bat = ind//div
        rem = ind % div
        return (bat, rem)


def action_reset(action):

    # action['cities']['missiles']['actions'][:] = 0
    # action['cities']['missiles']['launch'][:] = 1
    # action['cities']['missiles']['enemy_tar'][:] = False
    # action['cities']['missiles']['enemy_atc'][:] = False

    action['enemies']['missiles']['actions'][:] = 0
    action['enemies']['missiles']['launch'][:] = 0
    action['enemies']['missiles']['enemy_tar'][:] = False
    action['enemies']['missiles']['enemy_atc'][:] = False

    action['friends']['missiles']['actions'][:] = 0
    action['friends']['missiles']['launch'][:] = 0
    action['friends']['missiles']['enemy_tar'][:] = False
    action['friends']['missiles']['enemy_atc'][:] = False


# def _vector_distance(veca, vecb,  num_extract = None, expl_dist = None, veca_targets = None, switch = False):
#     """
#
#     compute distances between all pairs.
#     """
#
#     # Align vectors
#     veca_dup = np.repeat(veca, vecb.shape[0], axis=0)
#     vecb_dup = np.tile(vecb, reps=[veca.shape[0], 1])
#
#     # Compute distances
#     dx = vecb_dup[:, 0] - veca_dup[:, 0]
#     dy = vecb_dup[:, 1] - veca_dup[:, 1]
#     distances = np.sqrt(np.square(dx) + np.square(dy))
#
#     distances_ab = np.reshape(
#         distances,
#         (veca.shape[0], vecb.shape[0]),
#     )
#
#     if switch:
#
#         veca_inds = np.random.sample(np.arange(veca.shape[0]), veca.shape[0]//2)
#         veca_targ = veca_targets[veca_inds]
#         min_b = np.argmin(distances_ab, axis = 1)
#         t = 1
#
#         return
#
#
#         return np.array(min_ind_a), np.array(min_ind_b), _, _, _
#
#     # Get vectors indices inside an explosion radius
#     empty_arr = np.zeros((2,2))
#     if expl_dist != None:
#         inside_radius_indices = np.where(distances_ab <= expl_dist)
#     else:
#         inside_radius_indices = [[],[]]
#
#     ################################# filter minimal distance objects #################################
#     if num_extract != None:
#         num_extract = min(num_extract, distances.shape[0])
#         min_val_indices = np.argsort(distances) #[-num_extract:]
#         min_val_indices_b = min_val_indices//veca.shape[0]
#         min_val_indices_a = min_val_indices% veca.shape[0] #min_val_indices - min_val_indices_b*veca.shape[0]
#         ind, i = 1, 1
#         min_ind_a, min_ind_b = [min_val_indices_a[0]], [min_val_indices_b[0]]
#         while (ind < num_extract) and (i < min_val_indices_b.shape[0]) :
#             if (min_val_indices_a[i] not in min_ind_a) and (min_val_indices_b[i] not in min_ind_b):
#                 min_ind_a.append(min_val_indices_a[i])
#                 min_ind_b.append(min_val_indices_b[i])
#                 ind += 1
#             i += 1
#
#     else:
#         min_ind_b = min_ind_a = empty_arr[0]
#
#
#
#     # inside_radius = distances <= (
#     #         vecb_dup[:, 2] + CONFIG.ENEMY_MISSILES.RADIUS)
#     # inside_radius = inside_radius.astype(int)
#     # inside_radius = np.reshape(
#     #     inside_radius,
#     #     (veca.shape[0], vecb.shape[0]),
#     # )
#
#     # Remove theses missiles
#     # indices = np.argwhere(np.sum(inside_radius, axis=1) >= 1)
#
#     # self.enemy_missiles.enemy_missiles = np.delete(
#     #     self.enemy_missiles.enemy_missiles,
#     #     np.squeeze(missiles_out),
#     #     axis=0,
#     # )
#     #
#     # # Compute current reward
#     # nb_missiles_destroyed = missiles_out.shape[0]
#     # self.reward_timestep += CONFIG.REWARD.DESTROYED_ENEMEY_MISSILES * \
#     #                         nb_missiles_destroyed
#
#     return np.array(min_ind_a), np.array(min_ind_b), inside_radius_indices[0], inside_radius_indices[1], distances_ab







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
    """Check for missiles collisions.

    Check enemy missiles destroyed by friendly exploding missiles.
    """
    # Friendly exploding missiles
    friendly_exploding = self.friendly_missiles.missiles_explosion

    # Enemy missiles current positions
    enemy_missiles = self.enemy_missiles.enemy_missiles[:, [2, 3]]

    # Align enemy missiles and friendly exploding ones
    enemy_m_dup = np.repeat(enemy_missiles,
                            friendly_exploding.shape[0],
                            axis=0)
    friendly_e_dup = np.tile(friendly_exploding,
                             reps=[enemy_missiles.shape[0], 1])

    # Compute distances
    dx = friendly_e_dup[:, 0] - enemy_m_dup[:, 0]
    dy = friendly_e_dup[:, 1] - enemy_m_dup[:, 1]
    distances = np.sqrt(np.square(dx) + np.square(dy))

    # Get enemy missiles inside an explosion radius
    inside_radius = distances <= (
            friendly_e_dup[:, 2] + CONFIG.ENEMY_MISSILES.RADIUS)
    inside_radius = inside_radius.astype(int)
    inside_radius = np.reshape(
        inside_radius,
        (enemy_missiles.shape[0], friendly_exploding.shape[0]),
    )

    # Remove theses missiles
    missiles_out = np.argwhere(np.sum(inside_radius, axis=1) >= 1)
    self.enemy_missiles.enemy_missiles = np.delete(
        self.enemy_missiles.enemy_missiles,
        np.squeeze(missiles_out),
        axis=0,
    )

    # Compute current reward
    nb_missiles_destroyed = missiles_out.shape[0]
    self.reward_timestep += CONFIG.REWARD.DESTROYED_ENEMEY_MISSILES * \
                            nb_missiles_destroyed


N_switch = 20


def extract_friend_enemies_indices(friends_missiless, enenmy_cities):
    live_non_launched_fr_missile = np.where(
        np.logical_and(friends_missiless['health'] == 1, friends_missiless['launch'] == 0) == True )
    live_launched_fr_missile = np.where(
        np.logical_and(friends_missiless['health'] == 1, friends_missiless['launch'] == 1) == True)

    live_enemy_cities = np.where(enenmy_cities['health'] == 1)
    target_enemy_cities = np.where(
        np.logical_and(np.logical_and(enenmy_cities['health'] == 1, enenmy_cities['launch'] == 1), enenmy_cities['enemy_atc'] == True) == True)
    non_target_enemy_cities = np.where(
        np.logical_and(np.logical_and(enenmy_cities['health'] == 1, enenmy_cities['launch'] == 1), enenmy_cities['enemy_atc'] == False) == True)
    return  live_non_launched_fr_missile, live_launched_fr_missile, target_enemy_cities, non_target_enemy_cities


def kill_exp_friends_missiles(friends_missiless, enenmy_cities, live_launched_fr_missile, target_enemy_cities, third_bats = None):

    fr_missiles_pos = friends_missiless['pose'][live_launched_fr_missile][:, :2]
    en_cities_pos = enenmy_cities['pose'][target_enemy_cities][:, :2]
    launch_indicesr, target_indicesr, fr_exp_ind, tar_exp_ind, distances = _vector_distance(fr_missiles_pos, en_cities_pos,
                                                                                            num_extract=100,
                                                                                            expl_dist=CONFIG.FRIENDLY_MISSILES.EXPLOSION_RADIUS)
    fr_exp_ind, tar_exp_ind = list(set(fr_exp_ind)), list(set(tar_exp_ind))
    kill_launch_indices, kill_target_indices = tuple(np.array(live_launched_fr_missile)[:, fr_exp_ind]), \
                                               tuple(np.array(target_enemy_cities)[:, tar_exp_ind])

    if len(kill_launch_indices[0]) > 0:

        friends_missiless['launch'][kill_launch_indices] = 0
        friends_missiless['health'][kill_launch_indices] = 0

        ############ release targets of killed friends XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        released_cities = friends_missiless['targets'][kill_launch_indices]
        ind = tuple(released_cities.transpose())
        enenmy_cities['enemy_atc'][ind] = 0

        enenmy_cities['launch'][kill_target_indices] = 0
        enenmy_cities['health'][kill_target_indices] = 0

        ############ release targets of killed enemies XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        if third_bats != None:
            released_cities = enenmy_cities['targets'][kill_target_indices]
            ind = tuple(released_cities.transpose())
            third_bats['missiles']['enemy_atc'][ind] = 0

    return kill_launch_indices, kill_target_indices


def launch_action(src_action, friends_missiless, enenmy_cities, live_non_launched_fr_missile, non_target_enemy_cities):

    fr_missiles_pos = friends_missiless['pose'][live_non_launched_fr_missile][:, :2]
    en_cities_pos = enenmy_cities['pose'][non_target_enemy_cities][:, :2]
    launch_indicesr, target_indicesr, _, _, distances = _vector_distance(fr_missiles_pos, en_cities_pos,
                                                                         num_extract=CONFIG.FRIENDLY_BATTERY.MAX_LAUNCH)
    launch_indices, target_indices = tuple(np.array(live_non_launched_fr_missile)[:, launch_indicesr]), \
                                     tuple(np.array(non_target_enemy_cities)[:, target_indicesr].transpose())

    src_action['batteries'][launch_indices[0]] = 1
    src_action['missiles']['launch'][launch_indices] = 1
    src_action['missiles']['targets'][launch_indices] = target_indices

    return launch_indices, target_indices


def launch_update_observation(src_bats, dst_bats, launch_indices, target_indices):
        src_bats['missiles']['launch'][launch_indices] = 1
        src_bats['missiles']['targets'][launch_indices] = target_indices
        src_bats['missiles']['enemy_tar'][launch_indices] = True

        tt = np.array(target_indices).transpose()
        idx = tuple([tt[0], tt[1]])
        dst_bats['missiles']['enemy_atc'][idx] = True

        #### set direction/action command  ##########


def direction_action(src_action, friends_missiless, city_missiles):

    live_launched_fr_missile = np.where(
        np.logical_and(np.logical_and(friends_missiless['health'] == 1, friends_missiless['launch'] == 1), friends_missiless['enemy_tar'] == True))
    fr_missiles_pos = friends_missiless['pose'][live_launched_fr_missile][:, :2]

    target_indices = tuple(friends_missiless['targets'][live_launched_fr_missile].transpose())
    en_cities_pos = city_missiles['pose'][target_indices][:, :2]

    fr_missile_angle = friends_missiless['pose'][live_launched_fr_missile][:, 3]
    target_angle = np.arctan2(
        ( en_cities_pos[:, 1] - fr_missiles_pos[:, 1]), ( en_cities_pos[:, 0] - fr_missiles_pos[:, 0]))
    dtheta = 180 / np.pi * target_angle - fr_missile_angle
    print(dtheta)
    action_ind = np.clip(dtheta.astype(np.int), CONFIG.FRIENDLY_MISSILES.DTHETA[0],
                         CONFIG.FRIENDLY_MISSILES.DTHETA[-1]) // 5 - CONFIG.FRIENDLY_MISSILES.DTHETA[0] // 5
    src_action['missiles']['actions'][live_launched_fr_missile] = action_ind

    target_indices = src_action['missiles']['targets'][live_launched_fr_missile].transpose()

    return live_launched_fr_missile, target_indices


def state_process(src_bats, dst_bats, src_action, launch_step, launch_range, third_bats = None):

    friends_missiless = src_bats['missiles']
    enenmy_cities = dst_bats['missiles']

    ######### extract src target indices ##################################
    live_non_launched_fr_missile, live_launched_fr_missile, target_enemy_cities, non_target_enemy_cities \
        = extract_friend_enemies_indices(friends_missiless, enenmy_cities)

    #################### update launch and kill according to launch range ##########
    dst = np.linalg.norm(enenmy_cities['pose'][non_target_enemy_cities][:, :2] - dst_bats['pose'][non_target_enemy_cities[0]][:,:2], axis = 1)
    idx_launch = np.where(np.logical_and(dst > launch_range[0], dst < launch_range[1]))
    non_target_enemy_cities = tuple(np.array(non_target_enemy_cities).transpose()[idx_launch].transpose())

    idx_kill = np.where(dst > launch_range[1])
    print('dst = ', dst, ' idx = ', idx_kill)

    try:
        enenmy_cities['health'][idx_kill] = 0
    except:
        t = 1
    print(idx_kill)

    #################### kill src-targets if within range and update observation states #######################################
    live_launched_fr, target_en_cities = live_launched_fr_missile[0].shape[0], target_enemy_cities[0].shape[0]
    kill_launch_indices = [[]]
    if live_launched_fr > 0 and target_en_cities > 0:
        kill_launch_indices, kill_target_indices = kill_exp_friends_missiles(friends_missiless, enenmy_cities,
                                                        live_launched_fr_missile, target_enemy_cities, third_bats = third_bats)

    #### analyse targets - switch targets if required ##########################################################################

    return live_non_launched_fr_missile, non_target_enemy_cities



def sim_src_action_try(src_bats, dst_bats, src_action, launch_step, launch_range, third_bats = None):

    live_non_launched_fr_missile, non_target_enemy_cities = state_process(src_bats, dst_bats, src_action, launch_step, launch_range, third_bats = third_bats)
    # friends_missiless = src_bats['missiles']
    # enenmy_cities = dst_bats['missiles']
    #
    # ######### extract src target indices ##################################
    # live_non_launched_fr_missile, live_launched_fr_missile, target_enemy_cities, non_target_enemy_cities \
    #     = extract_friend_enemies_indices(friends_missiless, enenmy_cities)
    #
    # #################### update launch and kill according to launch range ##########
    # dst = np.linalg.norm(enenmy_cities['pose'][non_target_enemy_cities][:, :2] - dst_bats['pose'][non_target_enemy_cities[0]][:,:2], axis = 1)
    # idx_launch = np.where(np.logical_and(dst > launch_range[0], dst < launch_range[1]))
    # non_target_enemy_cities = tuple(np.array(non_target_enemy_cities).transpose()[idx_launch].transpose())
    #
    # idx_kill = np.where(dst > launch_range[1])
    # print('dst = ', dst, ' idx = ', idx_kill)
    #
    # try:
    #     enenmy_cities['health'][idx_kill] = 0
    # except:
    #     t = 1
    # print(idx_kill)
    #
    # #################### kill src-targets if within range and update observation states #######################################
    # live_launched_fr, target_en_cities = live_launched_fr_missile[0].shape[0], target_enemy_cities[0].shape[0]
    # kill_launch_indices = [[]]
    # if live_launched_fr > 0 and target_en_cities > 0:
    #     kill_launch_indices, kill_target_indices = kill_exp_friends_missiles(friends_missiless, enenmy_cities,
    #                                                     live_launched_fr_missile, target_enemy_cities, third_bats = third_bats)
    #
    # #### analyse targets - switch targets if required ##########################################################################
    #

    #### launch new missiles if available - to minimal distance targets ########################################################
    live_non_launched_fr, target_en_cities = live_non_launched_fr_missile[0].shape[0], non_target_enemy_cities[0].shape[0]
    if live_non_launched_fr > 0 and target_en_cities > 0 and launch_step == 0:

        ########## launch action to new targets #####################################################
        launch_indices, target_indices = launch_action(src_action, src_bats['missiles'], dst_bats['missiles'], live_non_launched_fr_missile, non_target_enemy_cities)
        launch_update_observation(src_bats, dst_bats, launch_indices, target_indices)

    #### set direction/action command  ##########
    live_launched_fr_missile, target_indices = direction_action(src_action, src_bats['missiles'], dst_bats['missiles'])

    done = False
    info = False
    return (live_launched_fr_missile, target_indices), done, info




def sim_src_action(src_bats, dst_bats, src_action, launch_step, launch_range, third_bats = None):

    friends_missiless = src_bats['missiles']
    enenmy_cities = dst_bats['missiles']

    ######### extract src target indices ##################################
    live_non_launched_fr_missile, live_launched_fr_missile, target_enemy_cities, non_target_enemy_cities \
        = extract_friend_enemies_indices(friends_missiless, enenmy_cities)

    #################### update launch and kill according to launch range ##########
    dst = np.linalg.norm(enenmy_cities['pose'][non_target_enemy_cities][:, :2] - dst_bats['pose'][non_target_enemy_cities[0]][:,:2], axis = 1)
    idx_launch = np.where(np.logical_and(dst > launch_range[0], dst < launch_range[1]))
    non_target_enemy_cities = tuple(np.array(non_target_enemy_cities).transpose()[idx_launch].transpose())

    idx_kill = np.where(dst > launch_range[1])
    print('dst = ', dst, ' idx = ', idx_kill)

    try:
        enenmy_cities['health'][idx_kill] = 0
    except:
        t = 1
    print(idx_kill)

    #################### kill src-targets if within range and update observation states #######################################
    live_launched_fr, target_en_cities = live_launched_fr_missile[0].shape[0], target_enemy_cities[0].shape[0]
    kill_launch_indices = [[]]
    if live_launched_fr > 0 and target_en_cities > 0:
        kill_launch_indices, kill_target_indices = kill_exp_friends_missiles(friends_missiless, enenmy_cities,
                                                        live_launched_fr_missile, target_enemy_cities, third_bats = third_bats)

    #### analyse targets - switch targets if required ##########################################################################


    #### launch new missiles if available - to minimal distance targets ########################################################
    live_non_launched_fr, target_en_cities = live_non_launched_fr_missile[0].shape[0], non_target_enemy_cities[0].shape[0]
    if live_non_launched_fr > 0 and target_en_cities > 0 and launch_step == 0:

        ########## launch action to new targets #####################################################
        launch_indices, target_indices = launch_action(src_action, friends_missiless, enenmy_cities, live_non_launched_fr_missile, non_target_enemy_cities)
        launch_update_observation(src_bats, dst_bats, launch_indices, target_indices)

    #### set direction/action command  ##########
    live_launched_fr_missile, target_indices = direction_action(src_action, friends_missiless, enenmy_cities)

    done = False
    info = False
    return (live_launched_fr_missile, target_indices), done, info


def sim_src_action_orig(src_bats, dst_bats, src_action, launch_step, launch_range, third_bats = None):

    friends_missiless = src_bats['missiles']
    enenmy_cities = dst_bats['missiles']

    ######### extract src target indices ##################################
    live_non_launched_fr_missile, live_launched_fr_missile, target_enemy_cities, non_target_enemy_cities \
        = extract_friend_enemies_indices(friends_missiless, enenmy_cities)

    #################### update launch and kill according to launch range ##########
    dst = np.linalg.norm(enenmy_cities['pose'][non_target_enemy_cities][:, :2] - dst_bats['pose'][non_target_enemy_cities[0]][:,:2], axis = 1)
    idx_launch = np.where(np.logical_and(dst > launch_range[0], dst < launch_range[1]))
    non_target_enemy_cities = tuple(np.array(non_target_enemy_cities).transpose()[idx_launch].transpose())
    # idx_kill = np.where(dst > launch_range[1])
    # enenmy_cities['health'][idx_kill] = 0
    # print(idx_kill)

    #################### kill src-targets if within range and update observation states #######################################
    live_launched_fr, target_en_cities = live_launched_fr_missile[0].shape[0], target_enemy_cities[0].shape[0]
    kill_launch_indices = [[]]
    if live_launched_fr > 0 and target_en_cities > 0:
        kill_launch_indices, kill_target_indices = kill_exp_friends_missiles(friends_missiless, enenmy_cities,
                                                        live_launched_fr_missile, target_enemy_cities, third_bats = third_bats)

    #### analyse targets - switch targets if required ##########################################################################


    #### launch new missiles if available - to minimal distance targets ########################################################
    live_non_launched_fr, target_en_cities = live_non_launched_fr_missile[0].shape[0], non_target_enemy_cities[0].shape[0]
    if live_non_launched_fr > 0 and target_en_cities > 0 and launch_step == 0:

        ########## launch action to new targets #####################################################
        launch_indices, target_indices = launch_action(src_action, friends_missiless, enenmy_cities, live_non_launched_fr_missile, non_target_enemy_cities)
        launch_update_observation(src_bats, dst_bats, launch_indices, target_indices)

    #### set direction/action command  ##########
    live_launched_fr_missile, target_indices = direction_action(src_action, friends_missiless, enenmy_cities)

    done = False
    info = False
    return (live_launched_fr_missile, target_indices), done, info




