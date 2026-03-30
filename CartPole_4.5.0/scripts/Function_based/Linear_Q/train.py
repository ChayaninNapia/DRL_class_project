"""Train a Linear Q-Learning agent."""

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

# add argparse arguments
parser = argparse.ArgumentParser(description="Train a Linear Q-Learning agent.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=1, help="Linear_Q uses a single environment; other values are ignored.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--n_episodes", type=int, default=None, help="Number of training episodes.")

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
from torch.utils.tensorboard import SummaryWriter

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
    """Train a Linear Q-Learning agent."""

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    if args_cli.num_envs not in (None, 1):
        print(f"[Linear_Q] overriding num_envs={args_cli.num_envs} to 1 for single-env training.")
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

    # ------------------------------------------------------------------ #
    # Hyperparameters
    # ------------------------------------------------------------------ #
    num_of_action = 9
    action_range = [-25.0, 25.0]
    learning_rate = 0.01
    discount_factor = 0.99
    n_episodes = 5000
    initial_epsilon = 1.0
    epsilon_decay = 0.9996
    final_epsilon = 0.01

    # ------------------------------------------------------------------ #
    # CLI overrides
    # ------------------------------------------------------------------ #
    if args_cli.n_episodes is not None:
        n_episodes = args_cli.n_episodes

    # ------------------------------------------------------------------ #
    # Agent construction
    # ------------------------------------------------------------------ #
    agent = Algorithm(
        num_of_action=num_of_action,
        action_range=action_range,
        learning_rate=learning_rate,
        initial_epsilon=initial_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
        discount_factor=discount_factor,
    )

    # ------------------------------------------------------------------ #
    # Save path
    # ------------------------------------------------------------------ #
    save_interval = 500
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    model_dir = os.path.join(project_root, "model", "Function_based", Algorithm_name, task_name)
    os.makedirs(model_dir, exist_ok=True)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"{Algorithm_name}_ep{n_episodes}"
        f"_lr{learning_rate}"
        f"_ed{epsilon_decay}"
        f"_fe{final_epsilon}"
        f"_{run_timestamp}"
    )
    tensorboard_dir = os.path.join(project_root, "runs", "Function_based", Algorithm_name, task_name, run_name)
    os.makedirs(tensorboard_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=tensorboard_dir)
    print(f"TensorBoard log dir: {tensorboard_dir}")
    reward_history = []
    episode_length_history = []
    cumulative_reward = 0.0

    while simulation_app.is_running():
        for episode in tqdm(range(n_episodes)):

            # ========= put your code here ========= #
            episode_return, timestep = agent.learn(env)
            agent.decay_epsilon()
            # ====================================== #

            reward_history.append(float(episode_return))
            episode_length_history.append(int(timestep))
            cumulative_reward += float(episode_return)

            avg_reward = sum(reward_history) / len(reward_history)
            avg_episode_length = sum(episode_length_history) / len(episode_length_history)

            writer.add_scalar("reward/episode_reward", episode_return, episode)
            writer.add_scalar("reward/cumulative_reward", cumulative_reward, episode)
            writer.add_scalar("reward/average_reward", avg_reward, episode)
            writer.add_scalar("episode/average_length", avg_episode_length, episode)
            writer.add_scalar("episode/length", timestep, episode)
            writer.add_scalar("exploration/epsilon", agent.epsilon, episode)

            if episode % 100 == 0:
                print(
                    f"[{Algorithm_name}] episode {episode} "
                    f"return={episode_return:.2f} steps={timestep} epsilon={agent.epsilon:.4f}"
                )

            if episode % save_interval == 0 and episode > 0:
                agent.save_model(model_dir, f"{Algorithm_name}_{episode}.npy")

        agent.save_model(model_dir, f"{Algorithm_name}_final.npy")
        print("Training complete.")
        break
    # ==================================================================== #

    writer.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
