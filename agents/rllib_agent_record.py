# import pandas as pd
import json
import os
import shutil
import sys
import numpy as np
from PIL import Image
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import gym
import ray.rllib.utils.exploration.thompson_sampling
import ray
# from ray.rllib.algorithms import ppo
from ray.rllib.algorithms import alpha_zero
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.algorithms.impala import ImpalaConfig

from gym_missile_command import MissileCommandEnv

if __name__ == "__main__":

    path_to_checkpoint = "C:/Users/Yaniv/ray_results/" \
                         "PPO_MissileCommandEnv_2023-04-22_11-02-06ixkp89c_\checkpoint_000701"
    path_to_checkpoint = "tmp/sac2/checkpoint_009901"
    algo = "IMPALA"
    info = ray.init(ignore_reinit_error=True)
    print("Dashboard URL: http://{}".format(info["webui_url"]))
    # checkpoint_root = "tmp/ppo/"
    # shutil.rmtree(checkpoint_root, ignore_errors=True, onerror=None)  # clean up old runs

    SELECT_ENV = MissileCommandEnv("")  # "missile-command-v0"  # MissileCommandEnv  # "Taxi-v3" "CartPole-v1"
    register_env('MissileCommand', lambda config: MissileCommandEnv(config))
    N_ITER = 10


    print(ray.rllib.utils.check_env(SELECT_ENV))
    if algo == 'PPO':
        agent = (
            PPOConfig()
            .framework(framework="torch")
            .rollouts(num_rollout_workers=1)
            .resources(num_gpus=0)
            .environment(env=MissileCommandEnv, env_config={})
            .build()
        )
    if algo == 'IMPALA':
        eval_config = {

        }
        agent = (
            ImpalaConfig()
            .framework(framework="torch")
            .rollouts(num_rollout_workers=1)
            .resources(num_gpus=0)
            .environment(env=MissileCommandEnv, env_config={})
            .build()
        )
    # agent.restore(path_to_checkpoint)

    env = MissileCommandEnv("")
    for i in range(30):
        recorded_frames = []
        try:
            episode_reward = 0
            done = False
            obs, info = env.reset()
            while not done:
                frame = env.render()
                if frame is not None:
                    frame = np.array(Image.fromarray(frame.astype('uint8')))  #.rotate(180)
                    recorded_frames.append(frame)

                action = agent.compute_single_action(obs)
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
            print(episode_reward)
            clip = ImageSequenceClip(recorded_frames, fps=10)
            clip.write_videofile(f'Results'+algo+'/captRllib'+str(i)+'.mp4')

        except Exception as e:
            print(e)
    ray.shutdown()  # "Undo ray.init()".
