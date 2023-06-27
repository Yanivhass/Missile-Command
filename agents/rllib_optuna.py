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
from ray.rllib.algorithms.alpha_zero import AlphaZeroConfig
from ray.tune.logger import pretty_print
import torch
from gym_missile_command import MissileCommandEnv
from rllib_example import CartPoleSparseRewards
import optuna
N_ITER = 100
def objective(trial):
    sgd_minibatch_size = trial.suggest_int("sgd_minibatch_size", 128, 16384)
    gamma = trial.suggest_float("gamma", 0.9, 1.0)
    train_batch_size = trial.suggest_int("train_batch_size", 2000, 160000)
    config = PPOConfig()
    config = config.training(sgd_minibatch_size=sgd_minibatch_size,
                             entropy_coeff=0.001,
                             gamma=gamma,
                             train_batch_size=train_batch_size)
    config = config.resources(num_gpus=1)
    config = config.rollouts(num_rollout_workers=10)
    config = config.framework(framework="torch")
    agent = config.build(env=MissileCommandEnv)
    for n in range(N_ITER):
        result = agent.train()
    return result["episode_reward_mean"]


if __name__ == "__main__":
    load_checkpoint = False
    path_to_checkpoint = "checkpoints/"
    algo = "PPO"  # "PPO"\"AlphaZero"
    N_ITER = 40000
    register_env("MissileCommandEnv", lambda config: MissileCommandEnv(config))
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"

    info = ray.init(ignore_reinit_error=True)
    print("Dashboard URL: http://{}".format(info["webui_url"]))

    checkpoint_root = "tmp/ppo/"
    shutil.rmtree(checkpoint_root, ignore_errors=True, onerror=None)  # clean up old runs
    SELECT_ENV = MissileCommandEnv  # MissileCommandEnv("")  # "missile-command-v0"   # "Taxi-v3" "CartPole-v1"
    # SELECT_ENV = CartPoleSparseRewards
    # print(ray.rllib.utils.check_env(SELECT_ENV))

    study = optuna.create_study(
        storage="sqlite:///db.sqlite3",  # Specify the storage URL here.
        study_name="ppo_optuna",
        direction="maximize",)
    study.optimize(objective, n_trials=100)
    print(study.best_trial)

    # import pprint
    # plot the scores
