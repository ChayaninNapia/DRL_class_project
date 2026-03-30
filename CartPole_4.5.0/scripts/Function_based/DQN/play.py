"""Play a trained DQN agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import random
import sys
import torch

from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")

parser = argparse.ArgumentParser(description="Play a trained DQN agent.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during playing.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=1, help="DQN script currently uses a single environment.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="Unused placeholder for CLI compatibility.")
parser.add_argument("--load_path", type=str, default=None, help="Directory containing the checkpoint.")
parser.add_argument("--load_file", type=str, default=None, help="Checkpoint filename to load.")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

from RL_Algorithm.Function_based.DQN import DQN as Algorithm

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
    """Play with a trained DQN agent."""

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    if args_cli.num_envs not in (None, 1):
        print(f"[DQN] overriding num_envs={args_cli.num_envs} to 1 for single-env play.")
    env_cfg.scene.num_envs = 1
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    device = torch.device(args_cli.device)
    print("device:", device)

    task_name = str(args_cli.task).split("-")[0]
    algorithm_name = "DQN"
    n_episodes = 10

    num_of_action = 9
    action_range = [-25.0, 25.0]
    n_observations = 4
    hidden_dim = 128
    dropout = 0.0
    learning_rate = 1e-3
    tau = 0.005
    discount_factor = 0.99
    initial_epsilon = 0.0
    epsilon_decay = 1.0
    final_epsilon = 0.0
    buffer_size = 10000
    batch_size = 64

    agent = Algorithm(
        device=device,
        num_of_action=num_of_action,
        action_range=action_range,
        n_observations=n_observations,
        hidden_dim=hidden_dim,
        dropout=dropout,
        learning_rate=learning_rate,
        tau=tau,
        initial_epsilon=initial_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
        discount_factor=discount_factor,
        buffer_size=buffer_size,
        batch_size=batch_size,
    )

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    default_model_dir = os.path.join(project_root, "model", "Function_based", algorithm_name, task_name)
    model_dir = os.path.abspath(args_cli.load_path) if args_cli.load_path is not None else default_model_dir

    if args_cli.load_file is not None:
        model_filename = args_cli.load_file
    else:
        final_filename = f"{algorithm_name}_final.pth"
        final_path = os.path.join(model_dir, final_filename)
        if os.path.isfile(final_path):
            model_filename = final_filename
        else:
            checkpoint_files = [
                filename for filename in os.listdir(model_dir)
                if filename.startswith(f"{algorithm_name}_") and filename.endswith(".pth")
            ]
            if not checkpoint_files:
                raise FileNotFoundError(f"No checkpoint files found in {model_dir}")

            def checkpoint_step(filename: str) -> int:
                stem = os.path.splitext(filename)[0]
                suffix = stem.split("_")[-1]
                return int(suffix) if suffix.isdigit() else -1

            model_filename = max(checkpoint_files, key=checkpoint_step)

    print(f"[{algorithm_name}] loading model from: {os.path.join(model_dir, model_filename)}", flush=True)
    agent.load_model(model_dir, model_filename)
    agent.epsilon = 0.0
    print(f"Loaded: {os.path.join(model_dir, model_filename)}", flush=True)

    while simulation_app.is_running():
        with torch.inference_mode():
            for episode in range(n_episodes):
                obs, _ = env.reset()
                episode_return = 0.0
                step = 0

                while True:
                    step += 1
                    env_action, _ = agent.select_action(obs)
                    obs, reward, terminated, truncated, _ = env.step(env_action)

                    reward_value = float(reward.detach().cpu().item()) if isinstance(reward, torch.Tensor) else float(reward)
                    terminated_flag = bool(terminated.detach().cpu().item()) if isinstance(terminated, torch.Tensor) else bool(terminated)
                    truncated_flag = bool(truncated.detach().cpu().item()) if isinstance(truncated, torch.Tensor) else bool(truncated)

                    episode_return += reward_value
                    if terminated_flag or truncated_flag:
                        print(f"[{algorithm_name}] episode {episode} return={episode_return:.2f} steps={step}", flush=True)
                        break

        break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
