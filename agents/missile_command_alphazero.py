import json
import os
import shutil
import yaml
import sys
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import gym
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms import alpha_zero
from ray.tune.logger import pretty_print
import torch
from gym_missile_command.envs.missile_command_env import MissileCommandEnv
from rllib_example import CartPoleSparseRewards
from ray.rllib.algorithms.alpha_zero.models.custom_torch_models import ConvNetModel, DenseModel, ActorCriticModel
from ray.tune.registry import register_env
from ray.rllib.models.catalog import ModelCatalog

if __name__ == "__main__":
    algo = "AlphaZero"  # "PPO"\"AlphaZero"
    N_ITER = 40000
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    info = ray.init(ignore_reinit_error=True)
    """print(info)
    print("Dashboard URL: http://{}".format(info.address_info["webui_url"]))"""
    checkpoint_root = "/tmp/alphazero/"
    shutil.rmtree(checkpoint_root, ignore_errors=True, onerror=None)  # clean up old runs
    register_env('MissileCommand', lambda config: MissileCommandEnv(config))
    ModelCatalog.register_custom_model("dense_model", DenseModel)
    ModelCatalog.register_custom_model("convnet_model", ConvNetModel)
    ModelCatalog.register_custom_model("actor_critic_model", ActorCriticModel)
    config = alpha_zero.AlphaZeroConfig()
    config = config.framework(framework="torch")
    config = config.resources(num_gpus=0)
    config = config.training(
        lr=0.0001,
        ranked_rewards={
            "enable": True,
        },
        mcts_config={
            'puct_coefficient': 1.5,
            'num_simulations': 100,
            'temperature': 1.0,
            'dirichlet_epsilon': 0.20,
            'dirichlet_noise': 0.03,
            'argmax_tree_policy': False,
            'add_dirichlet_noise': True,
        },
        model={
            "custom_model": "convnet_model",
            "custom_model_config": {
                "in_channels": 3,
                "feature_dim": 256,
            }
        },
    )
    print(config.to_dict())
    agent = config.build(env='MissileCommand')

    results = []
    episode_data = []
    episode_json = []
    for n in range(N_ITER):
        result = agent.train()
        results.append(result)

        episode = {
            "n": n,
            "episode_reward_min": result["episode_reward_min"],
            "episode_reward_mean": result["episode_reward_mean"],
            "episode_reward_max": result["episode_reward_max"],
            "episode_len_mean": result["episode_len_mean"],
        }
        episode_data.append(episode)
        episode_json.append(json.dumps(episode))
        file_name = agent.save(checkpoint_root)

        print(
            f'{n + 1:3d}: Min/Mean/Max reward: '
            f'{result["episode_reward_min"]:8.4f}/{result["episode_reward_mean"]:8.4f}/'
            f'{result["episode_reward_max"]:8.4f}, '
            f'len mean: {result["episode_len_mean"]:8.4f}. '
            f'Checkpoint saved to {file_name}'
        )
        if n % 50 == 0:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(np.arange(len(episode_data)), [d["episode_reward_min"] for d in episode_data])
            ax.plot(np.arange(len(episode_data)), [d["episode_reward_mean"] for d in episode_data])
            ax.plot(np.arange(len(episode_data)), [d["episode_reward_max"] for d in episode_data])
            ax.legend(['Min reward', 'Average reward', 'Max reward'])
            plt.ylabel('Score')
            plt.xlabel('Episode #')
            # plt.savefig(f"../Results/training scores{n}.png")
            plt.show()
            agent.save()


    # import pprint
    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(len(episode_data)), [d["episode_reward_min"] for d in episode_data])
    ax.plot(np.arange(len(episode_data)), [d["episode_reward_mean"] for d in episode_data])
    ax.plot(np.arange(len(episode_data)), [d["episode_reward_max"] for d in episode_data])
    ax.legend(['Min reward', 'Average reward', 'Max reward'])
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    # plt.savefig("../Results/training scores.png")
    plt.show()

    policy = agent.get_policy()
    model = policy.model
    path_to_checkpoint = agent.save()
    print(
        "An Algorithm checkpoint has been created inside directory: "
        f"'{path_to_checkpoint}'."
    )
    recorded_frames = []

    env = MissileCommandEnv("")
    for i in range(10):
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

            clip = ImageSequenceClip(recorded_frames, fps=10)
            # clip.write_videofile(f'../Results/captRllib_MA{i}.mp4')
        except Exception as e:
            print(e)
    ray.shutdown()  # "Undo ray.init()".