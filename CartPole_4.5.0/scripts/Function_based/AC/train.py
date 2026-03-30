"""Train an AC agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
from datetime import datetime
import os
import random
import sys

from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from tqdm import tqdm

parser = argparse.ArgumentParser(description="Train an AC agent.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--video_interval", type=int, default=2000)
parser.add_argument("--num_envs", type=int, default=1, help="AC script currently uses a single environment.")
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--n_episodes", type=int, default=None)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from torch.utils.tensorboard import SummaryWriter

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

from RL_Algorithm.Function_based.AC import AC as Algorithm

LOCAL_CARTPOLE_USD = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../assets/IsaacLab/Robots/Classic/Cartpole/cartpole.usd")
)
if os.path.isfile(LOCAL_CARTPOLE_USD):
    CARTPOLE_CFG.spawn.usd_path = LOCAL_CARTPOLE_USD

import CartPole.tasks  # noqa: F401


@hydra_task_config(args_cli.task, "sb3_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    if args_cli.num_envs not in (None, 1):
        print(f"[AC] overriding num_envs={args_cli.num_envs} to 1 for single-env training.")
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
    task_name = str(args_cli.task).split("-")[0]
    algorithm_name = "AC"

    # ------------------------------------------------------------------ #
    # Hyperparameters
    # ------------------------------------------------------------------ #
    hidden_dims = [128, 128]
    action_type = "discrete"
    learning_rate = 1e-3
    discount_factor = 0.99
    value_loss_coef = 0.5
    entropy_coef = 0.0
    max_grad_norm = 0.5
    n_episodes = 5000

    # ------------------------------------------------------------------ #
    # CLI overrides
    # ------------------------------------------------------------------ #
    if args_cli.n_episodes is not None:
        n_episodes = args_cli.n_episodes

    # ------------------------------------------------------------------ #
    # Agent construction
    # ------------------------------------------------------------------ #
    agent = Algorithm(
        device=device,
        num_of_action=9,
        action_range=[-25.0, 25.0],
        n_observations=4,
        hidden_dims=hidden_dims,
        activation="relu",
        action_type=action_type,
        init_noise_std=1.0,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        value_loss_coef=value_loss_coef,
        entropy_coef=entropy_coef,
        max_grad_norm=max_grad_norm,
    )

    # ------------------------------------------------------------------ #
    # Save path
    # ------------------------------------------------------------------ #
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    model_dir = os.path.join(project_root, "model", "Function_based", algorithm_name, task_name)
    os.makedirs(model_dir, exist_ok=True)
    run_name = f"{algorithm_name}_ep{n_episodes}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    tensorboard_dir = os.path.join(project_root, "runs", "Function_based", algorithm_name, task_name, run_name)
    writer = SummaryWriter(log_dir=tensorboard_dir)

    cumulative_reward = 0.0
    for episode in tqdm(range(n_episodes)):
        # ========= put your code here ========= #
        episode_return, loss, timestep = agent.learn(env)
        # ====================================== #
        cumulative_reward += episode_return
        writer.add_scalar("reward/episode_reward", episode_return, episode)
        writer.add_scalar("reward/cumulative_reward", cumulative_reward, episode)
        writer.add_scalar("episode/length", timestep, episode)
        writer.add_scalar("loss/total_loss", loss, episode)
        writer.add_scalar("loss/actor_loss", getattr(agent, "last_actor_loss", 0.0), episode)
        writer.add_scalar("loss/critic_loss", getattr(agent, "last_critic_loss", 0.0), episode)
        if episode % 50 == 0:
            print(
                f"[{algorithm_name}] episode {episode} return={episode_return:.2f} steps={timestep} "
                f"loss={loss:.4f} actor={getattr(agent, 'last_actor_loss', 0.0):.4f} "
                f"critic={getattr(agent, 'last_critic_loss', 0.0):.4f}"
            )
        if episode > 0 and episode % 200 == 0:
            agent.save_model(model_dir, f"{algorithm_name}_{episode}.pth")

    agent.save_model(model_dir, f"{algorithm_name}_final.pth")
    # ==================================================================== #
    writer.close()
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
