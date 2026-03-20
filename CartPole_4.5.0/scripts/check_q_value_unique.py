import argparse
import json
import os


def parse_state_key(state_key: str) -> tuple[int, int, int, int]:
    return tuple(int(float(value)) for value in state_key.strip("()").split(", "))


def main():
    parser = argparse.ArgumentParser(description="Inspect unique discretized state values from a q_value JSON file.")
    parser.add_argument(
        "--q-value-path",
        type=str,
        required=True,
        help="Path to a saved q_value JSON file.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.q_value_path):
        raise FileNotFoundError(f"Q-value file not found: {args.q_value_path}")

    with open(args.q_value_path, "r") as file:
        data = json.load(file)

    q_values = data["q_values"]

    pose_cart = set()
    pose_pole = set()
    vel_cart = set()
    vel_pole = set()

    for state_key in q_values:
        cart_pos, pole_pos, cart_vel, pole_vel = parse_state_key(state_key)
        pose_cart.add(cart_pos)
        pose_pole.add(pole_pos)
        vel_cart.add(cart_vel)
        vel_pole.add(pole_vel)

    print(f"file: {args.q_value_path}")
    print(f"num states: {len(q_values)}")
    print()
    print(f"pose_cart count: {len(pose_cart)}")
    print(f"pose_cart values: {sorted(pose_cart)}")
    print()
    print(f"pose_pole count: {len(pose_pole)}")
    print(f"pose_pole values: {sorted(pose_pole)}")
    print()
    print(f"vel_cart count: {len(vel_cart)}")
    print(f"vel_cart values: {sorted(vel_cart)}")
    print()
    print(f"vel_pole count: {len(vel_pole)}")
    print(f"vel_pole values: {sorted(vel_pole)}")


if __name__ == "__main__":
    main()
