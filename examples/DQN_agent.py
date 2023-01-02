"""Environment usage for a machine."""

import gym
from gym.spaces.utils import flatten_space, flatten
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import base64, io

import numpy as np
from collections import deque, namedtuple

# For visualization
from gym.wrappers.monitoring import video_recorder
# from IPython.display import HTML
# from IPython import display
# import glob



from QNetwork import Agent, dqn

# def show_video(env_name):
#     mp4list = glob.glob('video/*.mp4')
#     if len(mp4list) > 0:
#         mp4 = 'video/{}.mp4'.format(env_name)
#         video = io.open(mp4, 'r+b').read()
#         encoded = base64.b64encode(video)
#         display.display(HTML(data=''''''.format(encoded.decode('ascii'))))
#     else:
#         print("Could not find video")


def show_video_of_model(agent, env_name):
    env = gym.make(env_name)
    # vid = video_recorder.VideoRecorder(env, path="video/{}.mp4".format(env_name))
    recorded_frames = []
    agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))
    state = env.reset()
    done = False
    while not done:
        frame = env.render(mode='rgb_array')
        # vid.capture_frame()
        # frame = np.array(Image.fromarray(observation['sensors']['vision'].astype('uint8')).rotate(180))
        recorded_frames.append(frame)

        action = agent.act(state)

        state, reward, done, _ = env.step(action)


    env.close()


if __name__ == "__main__":
    print(torch.cuda.is_available())
    # print(torch.device())
    # Create the environment
    env = gym.make("gym_missile_command:missile-command-v0")
    env.seed(0)
    print('State shape: ', flatten(env.observation_space).shape)
    print('Number of actions: ', flatten(env.action_space).shape)
    # Reset it
    observation = env.reset()

    recorded_frames = []

    BUFFER_SIZE = int(1e5)  # replay buffer size
    BATCH_SIZE = 64  # minibatch size
    GAMMA = 0.99  # discount factor
    TAU = 1e-3  # for soft update of target parameters
    LR = 5e-4  # learning rate
    UPDATE_EVERY = 4  # how often to update the network
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    state_size = flatten_space(env.observation_space).shape[0]
    action_size = flatten_space(env.action_dictionary["attackers"]).shape[0]

    # While the episode is not finished
    done = False
    step = 0

    # agent = Agent(state_size=8, action_size=4, seed=0)
    scores = dqn(env=env,state_size=state_size, action_size=action_size)

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

    agent = Agent(state_size=state_size, action_size=action_size, seed=0)
    show_video_of_model(agent, 'LunarLander-v2')
    # show_video('LunarLander-v2')

