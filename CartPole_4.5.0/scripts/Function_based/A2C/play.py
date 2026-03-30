"""Play a trained A2C agent."""

import argparse
import os
import random
import sys

from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

parser = argparse.ArgumentParser(description="Play a trained A2C agent.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--video_interval", type=int, default=2000)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

from RL_Algorithm.Function_based.A2C import A2C as Algorithm

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

    device = torch.device(args_cli.device)
    task_name = str(args_cli.task).split("-")[0]
    algorithm_name = "A2C"
    agent = Algorithm(
        device=device,
        num_of_action=9,
        action_range=[-25.0, 25.0],
        n_observations=4,
        hidden_dims=[128, 128],
        activation="relu",
        action_type="discrete",
        init_noise_std=1.0,
        learning_rate=1e-3,
        discount_factor=0.99,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
    )
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    model_dir = os.path.join(project_root, "model", "Function_based", algorithm_name, task_name)
    agent.load_model(model_dir, f"{algorithm_name}_final.pth")

    with torch.inference_mode():
        for episode in range(10):
            obs, _ = env.reset()
            episode_return = 0.0
            step = 0
            while True:
                step += 1
                env_action = agent.select_action(obs)
                obs, reward, terminated, truncated, _ = env.step(env_action)
                reward_value = float(reward.detach().cpu().item()) if isinstance(reward, torch.Tensor) else float(reward)
                episode_return += reward_value
                done = (
                    bool(terminated.detach().cpu().item()) if isinstance(terminated, torch.Tensor) else bool(terminated)
                ) or (
                    bool(truncated.detach().cpu().item()) if isinstance(truncated, torch.Tensor) else bool(truncated)
                )
                if done:
                    print(f"[{algorithm_name}] episode {episode} return={episode_return:.2f} steps={step}")
                    break

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
