"""Play a trained MC REINFORCE agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import random
import sys

from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

parser = argparse.ArgumentParser(description="Play a trained MC REINFORCE agent.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during playing.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=1, help="MC_REINFORCE script currently uses a single environment.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="Unused placeholder for CLI compatibility.")

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
from torch.distributions import Categorical

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
    """Play with a trained MC REINFORCE agent."""

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    if args_cli.num_envs not in (None, 1):
        print(f"[MC_REINFORCE] overriding num_envs={args_cli.num_envs} to 1 for single-env play.")
    env_cfg.scene.num_envs = 1
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    device = torch.device(args_cli.device)
    print("device:", device)

    task_name = str(args_cli.task).split("-")[0]
    algorithm_name = "MC_REINFORCE"
    n_episodes = 10

    num_of_action = 9
    action_range = [-25.0, 25.0]
    n_observations = 4
    hidden_dim = 128
    dropout = 0.0
    action_type = "discrete"
    learning_rate = 1e-3
    discount_factor = 0.99

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

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    model_dir = os.path.join(project_root, "model", "Function_based", algorithm_name, task_name)
    model_filename = f"{algorithm_name}_final.pth"
    agent.load_model(model_dir, model_filename)
    print(f"Loaded: {os.path.join(model_dir, model_filename)}")

    while simulation_app.is_running():
        with torch.inference_mode():
            for episode in range(n_episodes):
                obs, _ = env.reset()
                episode_return = 0.0
                step = 0

                while True:
                    step += 1
                    if isinstance(obs, dict):
                        obs = obs.get("policy", next(iter(obs.values())))
                    if isinstance(obs, torch.Tensor):
                        obs_tensor = obs.detach().to(device, dtype=torch.float32).view(1, -1)
                    else:
                        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).view(1, -1)

                    logits = agent.policy_net(obs_tensor)
                    action_idx = int(Categorical(logits=logits).probs.argmax(dim=1).item())

                    action_min, action_max = agent.action_range
                    scaled_value = action_min + (action_idx / (agent.num_of_action - 1)) * (action_max - action_min)
                    env_action = torch.tensor([[scaled_value]], dtype=torch.float32, device=device)

                    obs, reward, terminated, truncated, _ = env.step(env_action)

                    reward_value = float(reward.detach().cpu().item()) if isinstance(reward, torch.Tensor) else float(reward)
                    terminated_flag = bool(terminated.detach().cpu().item()) if isinstance(terminated, torch.Tensor) else bool(terminated)
                    truncated_flag = bool(truncated.detach().cpu().item()) if isinstance(truncated, torch.Tensor) else bool(truncated)

                    episode_return += reward_value
                    if terminated_flag or truncated_flag:
                        print(f"[{algorithm_name}] episode {episode} return={episode_return:.2f} steps={step}")
                        break

        break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
