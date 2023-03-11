# import pandas as pd
import json
import os
import shutil
import sys

import gym
import ray
# from ray.rllib.algorithms import ppo
import ray.rllib.agents.ppo as ppo
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print

from gym_missile_command import MissileCommandEnv

if __name__ == "__main__":

    # algo = (
    #     PPOConfig()
    #         .framework(framework="torch")
    #         .rollouts(num_rollout_workers=1)
    #         .resources(num_gpus=0)
    #         .environment(env="CartPole-v1", render_env=True)
    #         .build()
    # )
    #
    # for i in range(10):
    #     result = algo.train()
    #     print(pretty_print(result))
    #
    #     if i % 5 == 0:
    #         checkpoint_dir = algo.save()
    #         print(f"Checkpoint saved in directory {checkpoint_dir}")
    info = ray.init(ignore_reinit_error=True)
    print("Dashboard URL: http://{}".format(info["webui_url"]))

    # algo = ppo.PPO(env=MissileCommandEnv, config={
    #     "env_config": {},  # config to pass to env class
    # }).framework(framework="torch")
    # print(algo.train())
    checkpoint_root = "tmp/ppo/"
    shutil.rmtree(checkpoint_root, ignore_errors=True, onerror=None)  # clean up old runs

    SELECT_ENV = "missile-command-v0"  # MissileCommandEnv  # "Taxi-v3" "CartPole-v1"
    N_ITER = 10

    config = ppo.DEFAULT_CONFIG.copy()
    config["log_level"] = "WARN"
    config["framework"] = "torch"

    # agent = ppo.PPOTrainer(config, env=SELECT_ENV)
    print(ray.rllib.utils.check_env([MissileCommandEnv]))
    # print(ray.rllib.utils.check_gym_environments([MissileCommandEnv]))
    agent = (
        PPOConfig()
            .framework(framework="torch")
            .rollouts(num_rollout_workers=1)
            .resources(num_gpus=0)
            .environment(env=MissileCommandEnv, env_config={})
            .build()
    )

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

    import pprint

    policy = agent.get_policy()
    model = policy.model
    path_to_checkpoint = agent.save()
    print(
        "An Algorithm checkpoint has been created inside directory: "
        f"'{path_to_checkpoint}'."
    )

    pprint.pprint(model.variables())
    pprint.pprint(model.value_function())
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
    env = gym.make("missile-command-v0")

    episode_reward = 0
    done = False
    obs = env.reset()
    while not done:
        env.render()
        action = agent.compute_single_action(obs)
        obs, reward, done, info = env.step(action)
        episode_reward += reward

    ray.shutdown()  # "Undo ray.init()".
