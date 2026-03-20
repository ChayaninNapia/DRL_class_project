"""Script to play RL agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from RL_Algorithm.Algorithm.SARSA import SARSA

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a trained tabular SARSA agent.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (unused in play).")
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

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab_tasks.utils import parse_env_cfg

# Import extensions to set up environment tasks
import CartPole.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def main():
    """Play with a trained SARSA agent."""
    if args_cli.num_envs != 1:
        print(f"[INFO] SARSA play only supports a single environment. Overriding --num_envs {args_cli.num_envs} -> 1.")

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
    )

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # ==================================================================== #
    # ========================= Can be modified ========================== #

    num_of_action = 7
    action_range = [-20, 20]  # [min, max]
    discretize_state_weight = [2, 5, 1, 1]  # [pose_cart:int, pose_pole:int, vel_cart:int, vel_pole:int]
    learning_rate = 0.1
    n_episodes = 10
    initial_epsilon = 1.0
    epsilon_decay = 0.995
    final_epsilon = 0.01
    discount = 1.0

    agent = SARSA(
        num_of_action=num_of_action,
        action_range=action_range,
        discretize_state_weight=discretize_state_weight,
        learning_rate=learning_rate,
        initial_epsilon=initial_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
        discount_factor=discount,
    )

    task_name = str(args_cli.task).split("-")[0]  # Stabilize, SwingUp
    checkpoint_dir = os.path.join("q_value", task_name, "SARSA")
    q_value_file = "SARSA_899_7_20_2_5.json"

    checkpoint_path = os.path.join(checkpoint_dir, q_value_file)
    if not os.path.isfile(checkpoint_path):
        print(
            "[ERROR] SARSA checkpoint file not found.\n"
            f"Expected file: {checkpoint_path}"
        )
        env.close()
        return

    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    agent.load_q_value(checkpoint_dir, q_value_file)
    agent.epsilon = 0.0

    # reset environment
    obs, _ = env.reset()
    timestep = 0

    # simulate environment
    while simulation_app.is_running():
        with torch.inference_mode():
            for episode in range(n_episodes):
                obs, _ = env.reset()
                done = False

                while not done:
                    # agent stepping
                    action, action_idx = agent.get_action(obs)

                    # env stepping
                    next_obs, reward, terminated, truncated, _ = env.step(action)

                    done = terminated.item() or truncated.item()
                    obs = next_obs

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        break
    # ==================================================================== #

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
