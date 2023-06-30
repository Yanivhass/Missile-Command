# import pandas as pd
import json
import os
import shutil
import sys
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import gym
from ray.tune.registry import register_env
import random
from ray import air, tune
from ray.tune.schedulers import PopulationBasedTraining
import ray
# from ray.rllib.algorithms import ppo
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.algorithms.alpha_zero import AlphaZeroConfig
from ray.tune.logger import pretty_print
import torch
from gym_missile_command import MissileCommandEnv
from ray.rllib.algorithms.impala import ImpalaConfig
from rllib_example import CartPoleSparseRewards
import optuna
from ray import air
from ray import tune
N_ITER = 10
def objective(trial):
    # sgd_minibatch_size = trial.suggest_int("sgd_minibatch_size", 128, 16384)
    # gamma = trial.suggest_float("gamma", 0.9, 1.0)
    # training_intensity = trial.suggest_int("training_intensity", 100, 10000)
    train_batch_size = trial.suggest_int("train_batch_size", 2000, 160000)
    config = SACConfig()
    config = config.training(train_batch_size=train_batch_size)
    config = config.resources(num_gpus=1)
    config = config.rollouts(num_rollout_workers=10)
    config = config.framework(framework="torch")
    agent = config.build(env=MissileCommandEnv)
    for n in range(N_ITER):
        result = agent.train()
    print(result["episode_reward_mean"])
    return result["episode_reward_mean"]


if __name__ == "__main__":
    ray.shutdown()
    load_checkpoint = False
    path_to_checkpoint = "checkpoints/"
    algo = "SAC"  # "PPO"\"AlphaZero"
    # N_ITER = 40000
    register_env("MissileCommandEnv", lambda config: MissileCommandEnv(config))
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"

    info = ray.init(ignore_reinit_error=True, num_gpus=1)

    SELECT_ENV = MissileCommandEnv  # MissileCommandEnv("")  # "missile-command-v0"   # "Taxi-v3" "CartPole-v1"
    # SELECT_ENV = CartPoleSparseRewards
    # print(ray.rllib.utils.check_env(SELECT_ENV))
    config = ImpalaConfig()
    config = config.training(
        lr=tune.grid_search(np.linspace(0.0001, 0.0005)), grad_clip=20.0
    )
    config = config.environment(env="MissileCommandEnv")
    tune.Tuner(
        "IMPALA",
        run_config=air.RunConfig(stop={"episode_reward_mean": 100, "training_iteration": 100}),
        param_space=config.to_dict(),
        tune_config=tune.TuneConfig(metric="episode_reward_mean", mode="max", num_samples=10),
    ).fit()
    ray.shutdown()
    # import pprint
    # plot the scores
