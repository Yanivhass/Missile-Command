import ray
from ray import rllib
from ray.rllib.examples.env.two_step_game import TwoStepGameWithGroupedAgents
from gym_missile_command.envs.MA_missile_command_env import MissileCommandEnv_MAGroupedAgents
from ray.rllib.algorithms.qmix import QMixConfig
from ray import air
from ray import tune

if __name__ == "__main__":
    '''
    config = {
        "env": "custom_env",
        "framework": "torch",
        "num_workers": 1,
        "num_envs_per_worker": 1,
        "rollout_fragment_length": 100,
        "train_batch_size": 1000,
        "exploration_config": {
            "epsilon_timesteps": 1000
        },
        "Q_model": {
            "atom_dim": 50,
            "num_atoms": 51,
            "dueling": False,
            "hidden_activation": "relu",
            "hidden_layers": [64, 64],
            "outbound_layers": [64, 64],
            "lstm_use_prev_action_reward": False,
            "lstm_cell_size": 64,
            "lstm_use_prev_action": False,
            "lstm_use_prev_reward": False,
            "framestack": True
        },
        "num_atoms": 51,
        "learning_starts": 1000,
        "target_network_update_freq": 500,
        "n_step": 1,
        "gamma": 0.99,
        "prioritized_replay": False,
        "buffer_size": 50000,
        "optimizer": {
            "lr": 0.0005
        },
        "model": {
            "free_log_std": False
        },
        "min_iter_time_s": 10,
        "evaluation_interval": 10,
        "evaluation_num_episodes": 5
    }
    '''
    config = QMixConfig()

    config = config.training(gamma=0.9, lr=0.01)
    config = config.framework(framework="torch")

    config = config.resources(num_gpus=0)

    config = config.rollouts(num_rollout_workers=4)

    print(config.to_dict())

    # Build an Algorithm object from the config and run 1 training iteration.

    algo = config.build(env=MissileCommandEnv_MAGroupedAgents)

    algo.train()