"""Script to train RL agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from RL_Algorithm.Algorithm.SARSA import SARSA

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
from datetime import datetime
import random

import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
# from omni.isaac.lab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import CartPole.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

steps_done = 0

@hydra_task_config(args_cli.task, "sb3_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with stable-baselines agent."""
    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # ==================================================================== #
    # ========================= Can be modified ========================== #

    # hyperparameters
    num_of_action = 9
    action_range = [-25, 25]  # [min, max]
    discretize_state_weight = [2, 7, 1, 1]  # [pose_cart:int, pose_pole:int, vel_cart:int, vel_pole:int]
    learning_rate = 2.0
    n_episodes = 5000
    start_epsilon = 1.0
    epsilon_decay = 0.998  # reduce the exploration over time
    final_epsilon = 0.01
    discount = 0.99




    # set up matplotlib
    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        from IPython import display

    plt.ion()

    # if GPU is to be used
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )

    print("device: ", device)

    task_name = str(args_cli.task).split('-')[0]  # Stabilize, SwingUp
    Algorithm_name = "SARSA"

    agent = SARSA(
        num_of_action=num_of_action,
        action_range=action_range,
        discretize_state_weight=discretize_state_weight,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
        discount_factor=discount,
    )

    tb_log_dir = os.path.join("runs", task_name, Algorithm_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    writer = SummaryWriter(log_dir=tb_log_dir)

    # reset environment
    obs, _ = env.reset()
    sum_reward = 0
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        with torch.inference_mode():
            for episode in tqdm(range(n_episodes)):
                obs, _ = env.reset()
                action, action_idx = agent.get_action(obs)
                done = False
                cumulative_reward = 0
                count = 0

                while not done:
                    next_obs, reward, terminated, truncated, _ = env.step(action)

                    reward_value = reward.item()
                    terminated_value = terminated.item()
                    truncated_value = truncated.item()
                    done = terminated_value or truncated_value
                    cumulative_reward += reward_value

                    if done:
                        next_action_idx = None
                    else:
                        next_action, next_action_idx = agent.get_action(next_obs)

                    agent.update(
                        obs,
                        action_idx,
                        reward_value,
                        done,
                        next_obs,
                        next_action_idx,
                    )

                    if not done:
                        obs = next_obs
                        action = next_action
                        action_idx = next_action_idx

                    count += 1

                writer.add_scalar("Reward/Episode", cumulative_reward, episode)
                writer.add_scalar("Policy/Epsilon", agent.epsilon, episode)
                writer.add_scalar("Episode/Length", count, episode)

                sum_reward += cumulative_reward
                if (episode + 1) % 100 == 0:
                    avg_score = sum_reward / 100.0
                    print("avg_score:", avg_score)
                    print(agent.epsilon)
                    sum_reward = 0

                    q_value_file = f"{Algorithm_name}_{episode}_{num_of_action}_{action_range[1]}_{discretize_state_weight[0]}_{discretize_state_weight[1]}.json"
                    full_path = os.path.join(f"q_value/{task_name}", Algorithm_name)
                    os.makedirs(full_path, exist_ok=True)
                    agent.save_q_value(full_path, q_value_file)

                agent.decay_epsilon()

        print("!!! Training is complete !!!")
        break
    # ==================================================================== #

    # close the simulator
    writer.close()
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()