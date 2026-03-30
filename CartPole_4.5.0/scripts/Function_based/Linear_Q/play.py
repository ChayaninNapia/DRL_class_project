"""Play a trained Linear Q-Learning agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import random
import sys

from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a trained Linear Q-Learning agent.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during playing.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=1, help="Linear_Q uses a single environment; other values are ignored.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="Unused placeholder for CLI compatibility.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import torch

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

from RL_Algorithm.Function_based.Linear_Q import Linear_QN as Algorithm

LOCAL_CARTPOLE_USD = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../../../assets/IsaacLab/Robots/Classic/Cartpole/cartpole.usd",
    )
)
if os.path.isfile(LOCAL_CARTPOLE_USD):
    CARTPOLE_CFG.spawn.usd_path = LOCAL_CARTPOLE_USD

import CartPole.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "sb3_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with a trained Linear Q-Learning agent."""

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    if args_cli.num_envs not in (None, 1):
        print(f"[Linear_Q] overriding num_envs={args_cli.num_envs} to 1 for single-env play.")
    env_cfg.scene.num_envs = 1
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # ==================================================================== #
    # ========================= Can be modified ========================== #

    # ------------------------------------------------------------------ #
    # Device & naming
    # ------------------------------------------------------------------ #
    device = torch.device(args_cli.device)
    print("device:", device)

    task_name = str(args_cli.task).split("-")[0]
    Algorithm_name = "Linear_Q"
    n_episodes = 10

    num_of_action = 9
    action_range = [-25.0, 25.0]
    learning_rate = 0.01
    discount_factor = 0.99
    initial_epsilon = 0.0
    epsilon_decay = 1.0
    final_epsilon = 0.0

    agent = Algorithm(
        num_of_action=num_of_action,
        action_range=action_range,
        learning_rate=learning_rate,
        initial_epsilon=initial_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
        discount_factor=discount_factor,
    )

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    model_dir = os.path.join(project_root, "model", "Function_based", Algorithm_name, task_name)
    model_filename = f"{Algorithm_name}_final.npy"
    agent.load_model(model_dir, model_filename)
    print(f"Loaded: {os.path.join(model_dir, model_filename)}")

    while simulation_app.is_running():
        with torch.inference_mode():
            for episode in range(n_episodes):

                # ========= put your code here ========= #
                obs, _ = env.reset()
                episode_return = 0.0

                step = 0
                while True:
                    step += 1
                    if isinstance(obs, dict):
                        obs = obs.get("policy", next(iter(obs.values())))

                    if isinstance(obs, torch.Tensor):
                        obs_np = obs.detach().cpu().numpy()
                    else:
                        obs_np = np.asarray(obs, dtype=np.float32)

                    obs_np = np.asarray(obs_np, dtype=np.float32).reshape(-1)
                    q_values = np.asarray(agent.q(obs_np), dtype=np.float32)
                    action_idx = int(np.argmax(q_values))

                    action_min, action_max = agent.action_range
                    if agent.num_of_action == 1:
                        scaled_action = np.array([[action_min]], dtype=np.float32)
                    else:
                        scaled_action = action_min + (
                            np.array([[action_idx]], dtype=np.float32) / (agent.num_of_action - 1)
                        ) * (action_max - action_min)

                    env_action = torch.tensor(scaled_action, dtype=torch.float32)

                    obs, reward, terminated, truncated, _ = env.step(env_action)

                    if isinstance(obs, dict):
                        obs = obs.get("policy", next(iter(obs.values())))

                    reward_np = reward.detach().cpu().numpy() if isinstance(reward, torch.Tensor) else np.asarray(reward)
                    terminated_np = (
                        terminated.detach().cpu().numpy() if isinstance(terminated, torch.Tensor) else np.asarray(terminated)
                    )
                    truncated_np = (
                        truncated.detach().cpu().numpy() if isinstance(truncated, torch.Tensor) else np.asarray(truncated)
                    )

                    episode_return += float(np.asarray(reward_np, dtype=np.float32))
                    done = bool(np.logical_or(terminated_np, truncated_np))
                    if done:
                        print(f"[{Algorithm_name}] episode {episode} return={episode_return:.2f} steps={step}")
                        break
                # ====================================== #

        break
    # ==================================================================== #

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
