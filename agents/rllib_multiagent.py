# import pandas as pd
import json
import os
import shutil
import sys
import numpy as np
from PIL import Image
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import gym
import ray
# from ray.rllib.algorithms import ppo
# import ray.rllib.agents.ppo as ppo
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.alpha_zero import AlphaZeroConfig
from ray.tune.logger import pretty_print
from ray.rllib.env.multi_agent_env import make_multi_agent
from ray.tune.registry import get_trainable_cls
from ray.rllib.algorithms.qmix import QMixConfig
from ray.rllib.env.env_context import EnvContext

from gym_missile_command import MissileCommandEnv
from gym_missile_command import MissileCommandEnv_MA,MissileCommandEnv_MAGroupedAgents

if __name__ == "__main__":
    algo = "QMIX"
    N_ITER = 1

    info = ray.init(ignore_reinit_error=True)
    print("Dashboard URL: http://{}".format(info["webui_url"]))

    # algo = ppo.PPO(env=MissileCommandEnv, config={
    #     "env_config": {},  # config to pass to env class
    # }).framework(framework="torch")
    # print(algo.train())
    checkpoint_root = f"tmp/{algo}/"
    shutil.rmtree(checkpoint_root, ignore_errors=True, onerror=None)  # clean up old runs

    # MA_MissileCommandEnv = make_multi_agent(lambda config: MissileCommandEnv(""))
    SELECTED_ENV = MissileCommandEnv_MAGroupedAgents#(env_config=EnvContext())

    # config = ppo.DEFAULT_CONFIG.copy()
    # config["log_level"] = "WARN"
    # config["framework"] = "torch"

    # agent = ppo.PPOTrainer(config, env=SELECT_ENV)
    # config.environment(disable_env_checking=True)
    # print(ray.rllib.utils.check_env(SELECTED_ENV))
    # print(ray.rllib.utils.check_gym_environments([MissileCommandEnv]))
    if algo is "PPO":
        agent = (
            PPOConfig()
                .framework(framework="torch")
                .rollouts(num_rollout_workers=1)
                .resources(num_gpus=0)
                .environment(env=SELECTED_ENV, env_config={"num_agents": 2})
                .build()
        )
    if algo is "QMIX":
        # config = (
        #     get_trainable_cls(algo)
        #         .get_default_config()
        #         .environment(MA_MissileCommandEnv)
        #         .framework("torch")
        #         # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        #         .resources(num_gpus=1)
        # )
        #
        # config.framework("torch")\
        #     .training(mixer="qmix", train_batch_size=32)\
        #     .rollouts(num_rollout_workers=0, rollout_fragment_length=4)\
        #     .exploration(
        #     exploration_config={
        #         "final_epsilon": 0.0,
        #     })\
        #     .environment(
        #         env=MA_MissileCommandEnv
        #     )
        config = QMixConfig()
        # config = config.training(gamma=0.9, lr=0.01, kl_coeff=0.3)
        config = config.resources(num_gpus=1)
        config = config.rollouts(num_rollout_workers=0)
        print(config.to_dict())
        # Build an Algorithm object from the config and run 1 training iteration.
        # algo = config.build(env=MissileCommandEnv_MA)
        agent = config.build(env=SELECTED_ENV)

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

    env = SELECTED_ENV #MissileCommandEnv("")
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
