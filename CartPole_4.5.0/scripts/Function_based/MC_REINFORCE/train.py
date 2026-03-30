"""Train an MC REINFORCE agent."""

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

parser = argparse.ArgumentParser(description="Train an MC REINFORCE agent.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=1, help="MC_REINFORCE script currently uses a single environment.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--n_episodes", type=int, default=None, help="Number of training episodes.")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
from torch.utils.tensorboard import SummaryWriter

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

from RL_Algorithm.Function_based.MC_REINFORCE import MC_REINFORCE as Algorithm

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
    """Train an MC REINFORCE agent."""

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    if args_cli.num_envs not in (None, 1):
        print(f"[MC_REINFORCE] overriding num_envs={args_cli.num_envs} to 1 for single-env training.")
    env_cfg.scene.num_envs = 1
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    device = torch.device(args_cli.device)
    print("device:", device)

    task_name = str(args_cli.task).split("-")[0]
    algorithm_name = "MC_REINFORCE"

    num_of_action = 9
    action_range = [-25.0, 25.0]
    n_observations = 4
    hidden_dim = 128
    dropout = 0.0
    action_type = "discrete"
    learning_rate = 1e-3
    discount_factor = 0.99
    n_episodes = 5000

    if args_cli.n_episodes is not None:
        n_episodes = args_cli.n_episodes

    agent = Algorithm(
        device=device,
        num_of_action=num_of_action,
        action_range=action_range,
        n_observations=n_observations,
        hidden_dim=hidden_dim,
        dropout=dropout,
        action_type=action_type,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
    )

    save_interval = 200
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    model_dir = os.path.join(project_root, "model", "Function_based", algorithm_name, task_name)
    os.makedirs(model_dir, exist_ok=True)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"{algorithm_name}_ep{n_episodes}"
        f"_lr{learning_rate}"
        f"_gamma{discount_factor}"
        f"_{run_timestamp}"
    )
    tensorboard_dir = os.path.join(project_root, "runs", "Function_based", algorithm_name, task_name, run_name)
    os.makedirs(tensorboard_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=tensorboard_dir)
    print(f"TensorBoard log dir: {tensorboard_dir}")
    reward_history = []
    cumulative_reward = 0.0

    while simulation_app.is_running():
        for episode in tqdm(range(n_episodes)):
            episode_return, loss, trajectory = agent.learn(env)
            timestep = len(trajectory)

            reward_history.append(float(episode_return))
            cumulative_reward += float(episode_return)
            avg_reward = sum(reward_history) / len(reward_history)

            writer.add_scalar("reward/episode_reward", episode_return, episode)
            writer.add_scalar("reward/cumulative_reward", cumulative_reward, episode)
            writer.add_scalar("reward/average_reward", avg_reward, episode)
            writer.add_scalar("episode/length", timestep, episode)
            writer.add_scalar("loss/policy_loss", loss, episode)

            if episode % 50 == 0:
                print(
                    f"[{algorithm_name}] episode {episode} "
                    f"return={episode_return:.2f} steps={timestep} loss={loss:.4f}"
                )

            if episode % save_interval == 0 and episode > 0:
                agent.save_model(model_dir, f"{algorithm_name}_{episode}.pth")

        agent.save_model(model_dir, f"{algorithm_name}_final.pth")
        print("Training complete.")
        break

    writer.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
