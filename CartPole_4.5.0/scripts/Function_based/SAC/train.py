"""Train a SAC agent."""

import argparse
from datetime import datetime
import os
import random
import sys

from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from tqdm import tqdm

parser = argparse.ArgumentParser(description="Train a SAC agent.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--video_interval", type=int, default=2000)
parser.add_argument("--num_envs", type=int, default=1, help="SAC script currently uses a single environment.")
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

from RL_Algorithm.Function_based.SAC import SAC as Algorithm

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
    Algorithm_name = "SAC"

    # ------------------------------------------------------------------ #
    # Hyperparameters
    # ------------------------------------------------------------------ #
    num_of_action = 1
    action_range = [-25.0, 25.0]
    n_observations = 4
    hidden_dim = 128
    learning_rate = 3e-4
    alpha_lr = 3e-4
    tau = 0.005
    discount_factor = 0.99
    buffer_size = 100000
    batch_size = 64
    init_alpha = 0.2
    auto_alpha = True
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
        num_of_action=num_of_action,
        action_range=action_range,
        n_observations=n_observations,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        alpha_lr=alpha_lr,
        tau=tau,
        discount_factor=discount_factor,
        buffer_size=buffer_size,
        batch_size=batch_size,
        init_alpha=init_alpha,
        auto_alpha=auto_alpha,
    )

    # ------------------------------------------------------------------ #
    # Save path
    # ------------------------------------------------------------------ #
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    model_dir = os.path.join(project_root, "model", "Function_based", Algorithm_name, task_name)
    os.makedirs(model_dir, exist_ok=True)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"{Algorithm_name}_ep{n_episodes}"
        f"_lr{learning_rate}"
        f"_tau{tau}"
        f"_alpha{init_alpha}"
        f"_{run_timestamp}"
    )
    tensorboard_dir = os.path.join(project_root, "runs", "Function_based", Algorithm_name, task_name, run_name)
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_dir)
    print(f"TensorBoard log dir: {tensorboard_dir}")

    cumulative_reward = 0.0
    for episode in tqdm(range(n_episodes)):
        # ========= put your code here ========= #
        episode_return, timestep, stats = agent.learn(env)
        # ====================================== #
        cumulative_reward += episode_return
        writer.add_scalar("reward/episode_reward", episode_return, episode)
        writer.add_scalar("reward/cumulative_reward", cumulative_reward, episode)
        writer.add_scalar("episode/length", timestep, episode)
        writer.add_scalar("replay/buffer_size", len(agent.memory), episode)
        writer.add_scalar("loss/critic_loss", stats["critic_loss"], episode)
        writer.add_scalar("loss/actor_loss", stats["actor_loss"], episode)
        writer.add_scalar("loss/alpha_loss", stats["alpha_loss"], episode)
        writer.add_scalar("alpha/value", stats["alpha"], episode)
        if episode % 50 == 0:
            print(
                f"[{Algorithm_name}] episode {episode} return={episode_return:.2f} steps={timestep} "
                f"critic={stats['critic_loss']:.4f} actor={stats['actor_loss']:.4f} "
                f"alpha_loss={stats['alpha_loss']:.4f} alpha={stats['alpha']:.4f}"
            )
        if episode > 0 and episode % 200 == 0:
            agent.save_model(model_dir, f"{Algorithm_name}_{episode}.pth")

    agent.save_model(model_dir, f"{Algorithm_name}_final.pth")
    # ==================================================================== #
    writer.close()
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
