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
import ray
# from ray.rllib.algorithms import ppo
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.alpha_zero import AlphaZeroConfig
from ray.tune.logger import pretty_print
import torch
from gym_missile_command import MissileCommandEnv
from rllib_example import CartPoleSparseRewards

if __name__ == "__main__":
    load_checkpoint = True
    path_to_checkpoint = "C:/Users/Yaniv/ray_results/" \
                         "PPO_MissileCommandEnv_2023-04-25_16-02-25f22dj65w\checkpoint_002209"
    algo = "PPO"  # "PPO"\"AlphaZero"
    N_ITER = 40000
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"

    info = ray.init(ignore_reinit_error=True)
    print("Dashboard URL: http://{}".format(info["webui_url"]))

    checkpoint_root = "tmp/ppo/"
    shutil.rmtree(checkpoint_root, ignore_errors=True, onerror=None)  # clean up old runs
    SELECT_ENV = MissileCommandEnv  # MissileCommandEnv("")  # "missile-command-v0"   # "Taxi-v3" "CartPole-v1"
    # SELECT_ENV = CartPoleSparseRewards
    # print(ray.rllib.utils.check_env(SELECT_ENV))

    if algo == "PPO":
        config = PPOConfig()
        config = config.training(sgd_minibatch_size=256,
                                 entropy_coeff=0.001)
        config = config.resources(num_gpus=1)
        config = config.rollouts(num_rollout_workers=10)
        config = config.framework(framework="torch")
        '''explore_config = config.exploration_config.update(
            {"type": "EpsilonGreedy",
             "initial_epsilon": 0.96,
             "final_epsilon": 0.01,
             "epsilon_timesteps": 5000}
            )
        config = config.exploration(exploration_config=explore_config)'''
        print(config.to_dict())
        # Build a Algorithm object from the config and run 1 training iteration.
        agent = config.build(env=SELECT_ENV)
        '''agent = (
            PPOConfig()
                .framework(framework="torch")
                .rollouts(num_rollout_workers=10,num_envs_per_worker=4)
                .resources(num_gpus=1)
                .environment(env=SELECT_ENV, env_config={})
                .build()
        )'''
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
        if n % 100 == 0:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(np.arange(len(episode_data)), [d["episode_reward_min"] for d in episode_data])
            ax.plot(np.arange(len(episode_data)), [d["episode_reward_mean"] for d in episode_data])
            ax.plot(np.arange(len(episode_data)), [d["episode_reward_max"] for d in episode_data])
            ax.legend(['Min reward', 'Average reward', 'Max reward'])
            plt.ylabel('Score')
            plt.xlabel('Episode #')
            plt.savefig(f"../Results/training scores{n}.png")
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
    plt.savefig("../Results/training scores.png")
    plt.show()

    policy = agent.get_policy()
    model = policy.model
    path_to_checkpoint = agent.save()
    print(
        "An Algorithm checkpoint has been created inside directory: "
        f"'{path_to_checkpoint}'."
    )

    # pprint.pprint(model.variables())
    # pprint.pprint(model.value_function())
    # agent.evaluate()
    # print(model.base_model.summary())

    # !rllib
    # rollout \
    #         tmp / ppo / taxi / checkpoint_10 / checkpoint - 10 \
    #         - -config
    # "{\"env\": \"Taxi-v3\"}" \
    # - -run
    # PPO \
    # - -steps
    # 2000
    # Run manual inference loop for n episodes.
    # env = StatelessCartPole()
    # env = gym.make("gym_missile_command:missile-command-v0")
    # for _ in range(10):
    #     episode_reward = 0.0
    #     reward = 0.0
    #     action = 0
    #     terminated = truncated = False
    #     obs, info = env.reset()
    #     while not terminated and not truncated:
    #         # Create a dummy action using the same observation n times,
    #         # as well as dummy prev-n-actions and prev-n-rewards.
    #         action, state, logits = algo.compute_single_action(
    #             input_dict={
    #                 "obs": obs,
    #                 "prev_n_obs": np.stack([obs for _ in range(num_frames)]),
    #                 "prev_n_actions": np.stack([0 for _ in range(num_frames)]),
    #                 "prev_n_rewards": np.stack([1.0 for _ in range(num_frames)]),
    #             },
    #             full_fetch=True,
    #         )
    #         obs, reward, terminated, truncated, info = env.step(action)
    #         episode_reward += reward
    #
    #     print(f"Episode reward={episode_reward}")
    #
    # algo.stop()

    # env_name = "CartPole-v1"
    # algo = PPOConfig().environment(env=env_name).build()
    # env = gym.make(env_name)
    # env = gym.make("missile-command-v0")
    # env = gym.make("missile-command-v0", env_config={})
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
            clip.write_videofile(f'../Results/captRllib_MA{i}.mp4')
        except Exception as e:
            print(e)
    ray.shutdown()  # "Undo ray.init()".
