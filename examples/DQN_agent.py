"""Environment usage for a machine."""

import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

if __name__ == "__main__":

    # Create the environment
    env = gym.make("gym_missile_command:missile-command-v0")

    # Reset it
    observation = env.reset()

    recorded_frames = []

    action = env.action_space.sample()
    # action_reset(action)

    # While the episode is not finished
    done = False
    step = 0

    while step < 500 and not done:
        action = env.action_space.sample()
        # Select an action (here, a random one)
        # random_action = True
        # if random_action:
        #     action = env.action_space.sample()
        # else:
        #     ######## manual compute of friends and enemy actions ############
        #     # done, info = sim_friends_action(observation, action, step)
        #     # done, info = sim_enemies_action(observation, action, step)
        #     # fr_to_en_cities_indices, done, info =  sim_src_action(observation['friends_bat'], observation['enemy_cities'], action['friends'],
            #                                                       step%CONFIG.FRIENDLY_BATTERY.DLaunch_Time, np.array(CONFIG.FRIENDLY_BATTERY.DETECTION_RANGE))
            # en_to_fr_missiles_indices, done, info = sim_src_action(observation['enemy_bat'], observation['friends_bat'], action['enemies'],
            #                                                        step%CONFIG.ENNEMY_BATTERY.DLaunch_Time,
            #                                                        np.array(CONFIG.ENNEMY_BATTERY.DETECTION_RANGE), third_bats = observation['enemy_cities'])
            #################################################################

        # One step forward
        observation, reward, done, _ = env.step(action)

        # Render (or not) the environment
        env.render(mode="rgb_array")  # "processed_observation"/"rgb_array"

        # frame = np.array(Image.fromarray(observation['sensors']['vision'].astype('uint8')).rotate(180))
        # recorded_frames.append(frame)

        step += 1

    # clip = ImageSequenceClip(recorded_frames, fps=10)
    # clip.write_videofile('capt.mp4')

        # # Select an action (here, a random one)
        # en_action = []
        # for ind in range(CONFIG.ENNEMY_BATTERY.NUMBER):
        #     en_action.append(env.action_space['enemies'][ind].sample())
        # fr_action = []
        # for ind in range(CONFIG.FRIENDLY_BATTERY.NUMBER):
        #     fr_action.append(env.action_space['friends'][ind].sample())
        # action =  {'friends': fr_action, 'enemies': en_action}
        #
        # # One step forward
        # observation, reward, done, _ = env.step(action)
        #
        # # Render (or not) the environment
        # env.render(mode="rgb_array") # "processed_observation"/"rgb_array"
