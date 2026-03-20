import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

# Example:
# python scripts/3d_plot.py --q-value-path q_value/Stabilize/MC/MC_1999_7_25_2_9.json --output plots/mc_q_surface.png --show
# python scripts/3d_plot.py --q-value-path q_value/Stabilize/Q_Learning/Q_Learning_999_7_20_2_5.json --output plots/q_learning_surface.png --show


def parse_state_key(state_key: str) -> tuple[int, int, int, int]:
    """Convert a serialized tuple key back to integer state indices."""
    return tuple(int(float(value)) for value in state_key.strip("()").split(", "))


def load_surface_data(q_value_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse velocity dimensions and build a 2D grid in discretized state space."""
    with open(q_value_path, "r") as file:
        q_values = json.load(file)["q_values"]

    surface_map = {}
    cart_values = set()
    pole_values = set()

    for state_key, action_values in q_values.items():
        pose_cart, pose_pole, _, _ = parse_state_key(state_key)
        max_q_value = float(np.max(action_values))
        xy_key = (pose_cart, pose_pole)

        if xy_key not in surface_map or max_q_value > surface_map[xy_key]:
            surface_map[xy_key] = max_q_value

        cart_values.add(pose_cart)
        pole_values.add(pose_pole)

    cart_values = sorted(cart_values)
    pole_values = sorted(pole_values)

    z_grid = np.full((len(pole_values), len(cart_values)), np.nan)
    for (pose_cart, pose_pole), max_q_value in surface_map.items():
        cart_idx = cart_values.index(pose_cart)
        pole_idx = pole_values.index(pose_pole)
        z_grid[pole_idx, cart_idx] = max_q_value

    x_axis = np.array(cart_values, dtype=float)
    y_axis = np.array(pole_values, dtype=float)
    x_grid, y_grid = np.meshgrid(x_axis, y_axis)

    return x_grid, y_grid, z_grid


def main():
    parser = argparse.ArgumentParser(description="Create a 3D surface plot of Q-values.")
    parser.add_argument(
        "--q-value-path",
        type=str,
        default="q_value/Stabilize/MC/MC_1499_10_25_1_5.json",
        help="Path to a saved q_value JSON file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="plots/mc_q_surface.png",
        help="Path to save the output figure.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure window after saving.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.q_value_path):
        raise FileNotFoundError(f"Q-value file not found: {args.q_value_path}")

    x_grid, y_grid, z_grid = load_surface_data(q_value_path=args.q_value_path)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(
        x_grid,
        y_grid,
        z_grid,
        cmap="viridis",
        edgecolor="none",
        antialiased=True,
    )

    ax.set_title("3D Surface Plot of Discretized Q-Values")
    ax.set_xlabel("Discretized Cart Position (bin)")
    ax.set_ylabel("Discretized Pole Position (bin)")
    ax.set_zlabel("Max Q-Value")
    fig.colorbar(surface, ax=ax, shrink=0.7, pad=0.1, label="Max Q-Value")

    plt.tight_layout()
    plt.savefig(args.output, dpi=200)
    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
