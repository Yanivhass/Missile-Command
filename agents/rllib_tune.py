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


def explore(config):
    # ensure we collect enough timesteps to do sgd
    if config["train_batch_size"] < config["sgd_minibatch_size"] * 2:
        config["train_batch_size"] = config["sgd_minibatch_size"] * 2
    # ensure we run at least one sgd iter
    if config["num_sgd_iter"] < 1:
        config["num_sgd_iter"] = 1
    return config



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

    if algo == "PPO":
        hyperparam_mutations = {
            "lambda": lambda: random.uniform(0.9, 1.0),
            "clip_param": lambda: random.uniform(0.01, 0.5),
            "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
            "num_sgd_iter": lambda: random.randint(1, 30),
            "sgd_minibatch_size": lambda: random.randint(128, 16384),
            "train_batch_size": lambda: random.randint(2000, 160000),
        }

        pbt = PopulationBasedTraining(
            time_attr="time_total_s",
            perturbation_interval=120,
            resample_probability=0.25,
            # Specifies the mutations of these hyperparams
            hyperparam_mutations=hyperparam_mutations,
            custom_explore_fn=explore,
        )
        stopping_criteria = {"training_iteration": 10000, "episode_reward_mean": 200}
        tuner = tune.Tuner(
            "PPO",
            tune_config=tune.TuneConfig(
                metric="episode_reward_mean",
                mode="max",
                scheduler=pbt,
                num_samples=2,
            ),
            param_space={
                "env": "MissileCommandEnv",
                "kl_coeff": 1.0,
                "num_workers": 4,
                "num_gpus": 1,  # number of GPUs to use per trial
                "model": {"free_log_std": True},
                # These params are tuned from a fixed starting value.
                "lambda": 0.95,
                "clip_param": 0.2,
                "lr": 1e-4,
                # These params start off randomly drawn from a set.
                "num_sgd_iter": tune.choice([10, 20, 30]),
                "sgd_minibatch_size": tune.choice([128, 512, 2048]),
                "train_batch_size": tune.choice([10000, 20000, 40000]),
            },
            run_config=air.RunConfig(stop=stopping_criteria,name="ppo_hyperparmas"),
        )
        results = tuner.fit()
    if algo == "SAC":
        hyperparam_mutations = {
            "lambda": lambda: random.uniform(0.9, 1.0),
            "clip_param": lambda: random.uniform(0.01, 0.5),
            "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
            "num_sgd_iter": lambda: random.randint(1, 30),
            "sgd_minibatch_size": lambda: random.randint(128, 16384),
            "train_batch_size": lambda: random.randint(2000, 160000),
        }

        pbt = PopulationBasedTraining(
            time_attr="time_total_s",
            perturbation_interval=120,
            resample_probability=0.25,
            # Specifies the mutations of these hyperparams
            hyperparam_mutations=hyperparam_mutations,
            custom_explore_fn=explore,
        )
        stopping_criteria = {"training_iteration": 10000, "episode_reward_mean": 150}
        tuner = tune.Tuner(
            "SAC",
            tune_config=tune.TuneConfig(
                metric="episode_reward_mean",
                mode="max",
                scheduler=pbt,
                num_samples=2,
            ),
            param_space={
                "env": "MissileCommandEnv",
                "kl_coeff": 1.0,
                "num_workers": 4,
                "num_gpus": 1,  # number of GPUs to use per trial
                "model": {"free_log_std": True},
                # These params are tuned from a fixed starting value.
                "lambda": 0.95,
                "clip_param": 0.2,
                "lr": 1e-4,
                # These params start off randomly drawn from a set.
                "num_sgd_iter": tune.choice([10, 20, 30]),
                "sgd_minibatch_size": tune.choice([128, 512, 2048]),
                "train_batch_size": tune.choice([10000, 20000, 40000]),
            },
            run_config=air.RunConfig(stop=stopping_criteria, name="sac_hyperparmas"),
        )
        results = tuner.fit()

    if algo == "AlphaZero":
        config = AlphaZeroConfig()
        config = config.framework(framework="torch")
        config = config.training(sgd_minibatch_size=256)
        config = config.resources(num_gpus=1)
        config = config.rollouts(num_rollout_workers=6)
        print(config.to_dict())
        # Build a Algorithm object from the config and run 1 training iteration.
        agent = config.build(env=SELECT_ENV)

    results = []
    episode_data = []
    episode_json = []


    # import pprint
    # plot the scores
