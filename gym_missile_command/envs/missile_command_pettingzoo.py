import functools

import gymnasium
import numpy as np
from gymnasium.spaces import Discrete, Box

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers

# Global variables

LEFT = 0
STRAIGHT = 1
RIGHT = 2
TARGET = 3
FIRE = 4
ACTIONS = ["LEFT",
           "STRAIGHT",
           "RIGHT",
           "TARGET",
           "FIRE"]
NUM_ACTIONS = len(ACTIONS)
X = 0
Y = 1
VX = 2
VY = 3
TARGET_X = 4
TARGET_Y = 5
TARGET_VX = 6
TARGET_VY = 7
OBS = ["X", "Y", "VX", "VY", "TARGET_X", "TARGET_Y", "TARGET_VX", "TARGET_VY", "HEALTH"]
DIM_OBS = len(OBS)
INIT_STATE = 0


def env(num_attackers, num_defenders, render_mode='human'):
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = missile_command(num_attackers, num_defenders, render_mode=internal_render_mode)
    # This wrapper is only for environments which print results to the terminal
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

class missile_command(AECEnv):
    metadata = {'render.modes': ['human', 'rgb_array'], 'name': 'missile_command'}

    def __init__(self, num_attackers, num_defenders, num_missiles, render_mode='human'):
        super().__init__()
        self.render_mode = render_mode
        self.possible_attackers = ["attacker_" + str(r) for r in range(num_attackers)]
        self.possible_defenders = ["defender_" + str(r) for r in range(num_defenders)]
        self.possible_agents = self.possible_attackers + self.possible_defenders
        self.possible_atk_missiles = ["attacker_missile_" + str(r) for r in range(num_attackers*num_missiles)]
        self.possible_def_missiles = ["defender_missile_" + str(r) for r in range(num_defenders*num_missiles)]
        self.possible_missiles = self.possible_atk_missiles + self.possible_def_missiles
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self.missile_name_mapping = dict(
            zip(self.possible_missiles, list(range(len(self.possible_missiles))))
        )
        state_dict = gymnasium.spaces.Dict({
            "X": gymnasium.spaces.Box(-1, 1, shape=(1,)),
            "Y": gymnasium.spaces.Box(-1, 1, shape=(1,)),
            "VX": gymnasium.spaces.Box(-1, 1, shape=(1,)),
            "VY": gymnasium.spaces.Box(-1, 1, shape=(1,)),
            "TARGET_X": gymnasium.spaces.Box(-1, 1, shape=(1,)),
            "TARGET_Y": gymnasium.spaces.Box(-1, 1, shape=(1,)),
            "TARGET_VX": gymnasium.spaces.Box(-1, 1, shape=(1,)),
            "TARGET_VY": gymnasium.spaces.Box(-1, 1, shape=(1,)),
            "HEALTH": gymnasium.spaces.Box(0, 1, shape=(1,))
        })
        self.possible_all = self.possible_agents + self.possible_missiles
        self._action_spaces = {agent: Discrete(NUM_ACTIONS) for agent in self.possible_agents}
        self._observation_spaces = {agent: state_dict for agent in self.possible_all}
        self.render_mode = render_mode

    def observation_space(self, agent):
        return Box(-1, 1, shape=(DIM_OBS,))

    def action_space(self, agent):
        return Discrete(NUM_ACTIONS)

    def render(self):
        # NEED TO IMPLEMENT
        pass

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.missiles = self.possible_missiles[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: INIT_STATE for agent in self.agents}
        self.observations = {agent: INIT_STATE for agent in self.agents}
        self.num_moves = 0
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def observe(self, agent):
        return np.array(self.observations[agent])

    def step(self, action):
        # NEED TO IMPLEMENT
        pass


