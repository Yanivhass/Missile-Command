# import pandas as pd
import json
import os
import shutil
import sys
import ray
import ray.rllib.agents.ppo as ppo

from gym_missile_command import MissileCommandEnv

if __name__ == "__main__":
    info = ray.init(ignore_reinit_error=True)
    print("Dashboard URL: http://{}".format(info["webui_url"]))

    checkpoint_root = "tmp/ppo/"
    shutil.rmtree(checkpoint_root, ignore_errors=True, onerror=None)  # clean up old runs

    SELECT_ENV = MissileCommandEnv  # "Taxi-v3"
    N_ITER = 10

    config = ppo.DEFAULT_CONFIG.copy()
    config["log_level"] = "WARN"

    agent = ppo.PPOTrainer(config, env=SELECT_ENV)

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

        pprint.pprint(model.variables())
        pprint.pprint(model.value_function())

        print(model.base_model.summary())

        # !rllib
        # rollout \
        #         tmp / ppo / taxi / checkpoint_10 / checkpoint - 10 \
        #         - -config
        # "{\"env\": \"Taxi-v3\"}" \
        # - -run
        # PPO \
        # - -steps
        # 2000
        ray.shutdown()  # "Undo ray.init()".
