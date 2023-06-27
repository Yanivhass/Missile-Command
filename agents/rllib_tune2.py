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
import numpy as np


def loguniform(low=0, high=1):
    return np.exp(np.random.uniform(low, high))


def sample_ppo_params():
    return {
        'lr': tune.loguniform(5e-5, 0.0001),
        'sgd_minibatch_size': tune.choice([64, 128, 256]),
        'entropy_coeff': tune.loguniform(0.00000001, 0.1),
        'clip_param': tune.choice([0.1, 0.2, 0.3, 0.4]),
        "vf_loss_coeff": tune.uniform(0, 1),
        'lambda': tune.choice([0.9, 0.95, 0.98, 0.99, 0.995, 0.999]),
        'kl_target': tune.choice([0.001, 0.01, 0.1]),
    }


def mutate_ppo_hyperparams():
    return {
        "entropy_coeff": lambda: loguniform(0.00000001, 0.1),
        "lr": lambda: loguniform(5e-5, 0.0001),
        "sgd_minibatch_size": [32, 64, 128, 256, 512],
        "lambda": [0.9, 0.95, 0.98, 0.99, 0.995, 0.999],
        'clip_param': [0.1, 0.2, 0.3, 0.4],
        "vf_loss_coeff": lambda: np.random.uniform(0, 1),
        'kl_target': [0.001, 0.01, 0.1]
    }

MODEL_TRAINER = {'a2c':A2CTrainer,'ppo':PPOTrainer,'ddpg':DDPGTrainer}
model_name = 'ppo'
pbt_scheduler = PopulationBasedTraining(
    time_attr = "training_iteration",
    perturbation_interval = training_iterations/5,
    burn_in_period = 0.0,
    quantile_fraction = 0.25,
    hyperparam_mutations = mutate_hyperparameters
)

def run_PBT(log_dir):

  analysis = tune.run(
      MODEL_TRAINER[model_name],
      scheduler=pbt_scheduler, #To prune bad trials
      metric='episode_reward_mean',
      mode='max',
      config = {**sample_hyperparameters,
                'env':'StockTrading_train_env','num_workers':1,
                'num_gpus':1,'framework':'torch','log_level':'DEBUG'},
      num_samples = num_samples, #Number of hyperparameters to test out
      stop = {'training_iteration':training_iterations},#Time attribute to validate the results
      verbose=1,local_dir="./"+log_dir,#Saving tensorboard plots
      resources_per_trial={'gpu':1,'cpu':1},
      max_failures = 1,#Extra Trying for the failed trials
      raise_on_failed_trial=False,#Don't return error even if you have errored trials
      # keep_checkpoints_num = 2,
      checkpoint_score_attr ='episode_reward_mean',#Only store keep_checkpoints_num trials based on this score
      checkpoint_freq=training_iterations,#Checpointing all the trials,
  )
  return analysis


def ppo_param_bounds():
  return {
    'lr':[5e-5, 0.0001],
    'sgd_minibatch_size': [64,256],
    'entropy_coeff': [0.00000001, 0.1],
      'clip_param': [0.1,0.4],
      "vf_loss_coeff": [0,1],
      'lambda': [0.9,0.999],
      'kl_target': [0.001,0.1],
  }

pb2_scheduler = PB2(
                    time_attr='training_iteration',
                    perturbation_interval=training_iterations/5,
                    quantile_fraction= 0.25,
                    hyperparam_bounds={
                       **params_bounds
                    })

def run_PB2(log_dir):

  analysis = tune.run(
      MODEL_TRAINER[model_name],
      scheduler=pb2_scheduler, #To prune bad trials
      metric='episode_reward_mean',
      mode='max',
      config = {**sample_ppo_params, #It is the sample_ppo_params function
                'env':'StockTrading_train_env','num_workers':1,
                'num_gpus':1,'framework':'torch','log_level':'DEBUG'},
      num_samples = num_samples, #Number of hyperparameters to test out
      stop = {'training_iteration':training_iterations},#Time attribute to validate the results
      verbose=1,local_dir="./"+log_dir,#Saving tensorboard plots
      resources_per_trial={'gpu':1,'cpu':1},
      max_failures = 1,#Extra Trying for the failed trials
      raise_on_failed_trial=False,#Don't return error even if you have errored trials
      # keep_checkpoints_num = 2,
      checkpoint_score_attr ='episode_reward_mean',#Only store keep_checkpoints_num trials based on this score
      checkpoint_freq=training_iterations,#Checpointing all the trials
  # print("Best hyperparameter: ", analysis.best_config)
  return analysis
