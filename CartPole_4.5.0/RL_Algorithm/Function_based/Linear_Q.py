from __future__ import annotations
import os
import numpy as np
import torch
from RL_Algorithm.RL_base_function import BaseAlgorithm


class Linear_QN(BaseAlgorithm):
    """
    Linear Q-Learning with function approximation.

    Args:
        num_of_action (int): Number of discrete actions.
        action_range (list): [min, max] continuous action range.
        learning_rate (float): TD weight-update step size.
        initial_epsilon (float): Starting exploration rate.
        epsilon_decay (float): Per-step epsilon decay.
        final_epsilon (float): Minimum exploration rate.
        discount_factor (float): Discount factor γ.
    """

    def __init__(
            self,
            num_of_action: int = 2,
            action_range: list = [-2.5, 2.5],
            learning_rate: float = 0.01,
            initial_epsilon: float = 1.0,
            epsilon_decay: float = 1e-3,
            final_epsilon: float = 0.001,
            discount_factor: float = 0.95,
    ) -> None:

        super().__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            initial_epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            discount_factor=discount_factor,
        )

        # ===== Linear weight matrix ===== #
        # Shape: (obs_feature_dim, num_of_action)
        self.w = np.zeros((4, num_of_action))

    # ------------------------------------------------------------------ #
    # Linear Q-value estimation                                           #
    # ------------------------------------------------------------------ #

    def q(self, obs, a=None):
        """
        Return the linearly-estimated Q-value(s) for a given observation.

        Args:
            obs: State feature vector φ(s), shape (obs_dim,).
            a (int | None): Action index. If None, returns Q for all actions
                            as a 1-D array of shape (num_of_action,).

        Returns:
            float | np.ndarray: Q(s, a) scalar, or Q(s, :) array.
        """
        # ========= put your code here ========= #
        if isinstance(obs, dict):
            obs = obs.get("policy", next(iter(obs.values())))
        if isinstance(obs, torch.Tensor):
            obs = obs.detach().cpu().numpy()

        obs = np.asarray(obs, dtype=np.float32).reshape(-1)

        if self.w.shape[0] != obs.shape[0]:
            self.w = np.zeros((obs.shape[0], self.num_of_action), dtype=np.float32)

        q_values = obs @ self.w

        if a is None:
            return q_values

        return float(q_values[int(a)])
        # ====================================== #

    # ------------------------------------------------------------------ #
    # Core algorithm methods                                               #
    # ------------------------------------------------------------------ #

    def update(
        self,
        obs,
        action: int,
        reward: float,
        next_obs,
        next_action: int,
        terminated: bool,
    ):
        """
        Update the weight vector using the TD error.

        Args:
            obs: Current state feature vector φ(s).
            action (int): Action index taken in state s.
            reward (float): Reward received.
            next_obs: Next state feature vector φ(s').
            next_action (int): Next action taken (for SARSA-style update).
            terminated (bool): True if the episode ended.
        """
        # ========= put your code here ========= #
        if isinstance(obs, torch.Tensor):
            obs = obs.detach().cpu().numpy()
        if isinstance(next_obs, torch.Tensor):
            next_obs = next_obs.detach().cpu().numpy()
        if isinstance(reward, torch.Tensor):
            reward = reward.detach().cpu().numpy()
        if isinstance(terminated, torch.Tensor):
            terminated = terminated.detach().cpu().numpy()

        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        next_obs = np.asarray(next_obs, dtype=np.float32).reshape(-1)
        reward = float(np.asarray(reward, dtype=np.float32))
        terminated = bool(np.asarray(terminated).item())

        if self.w.shape[0] != obs.shape[0]:
            self.w = np.zeros((obs.shape[0], self.num_of_action), dtype=np.float32)

        current_q = float(self.q(obs, action))
        next_q_values = np.asarray(self.q(next_obs), dtype=np.float32)
        td_target = reward if terminated else reward + self.discount_factor * float(np.max(next_q_values))
        td_error = td_target - current_q

        self.w[:, int(action)] += self.lr * td_error * obs
        self.training_error.append(td_error)
        return float(td_error)
        # ====================================== #

    def select_action(self, state):
        """
        Select an action using an epsilon-greedy policy over Q(s, :).

        Args:
            state: Current state feature vector φ(s).

        Returns:
            Tuple[Tensor, int]: Scaled continuous action tensor and action index.
        """
        # ========= put your code here ========= #
        if isinstance(state, dict):
            state = state.get("policy", next(iter(state.values())))
        if isinstance(state, torch.Tensor):
            state = state.detach().cpu().numpy()

        state = np.asarray(state, dtype=np.float32).reshape(-1)

        if self.w.shape[0] != state.shape[0]:
            self.w = np.zeros((state.shape[0], self.num_of_action), dtype=np.float32)

        q_values = np.asarray(self.q(state), dtype=np.float32)
        greedy_action = int(np.argmax(q_values))
        if np.random.random() < self.epsilon:
            action = int(np.random.randint(self.num_of_action))
        else:
            action = greedy_action

        action_min, action_max = self.action_range
        if self.num_of_action == 1:
            scaled_action = np.array([[action_min]], dtype=np.float32)
        else:
            scaled_action = action_min + (
                np.array([[action]], dtype=np.float32) / (self.num_of_action - 1)
            ) * (action_max - action_min)

        env_action = torch.tensor(scaled_action, dtype=torch.float32)
        return env_action, action
        # ====================================== #

    def learn(self, env):
        """
        Train the agent for one episode.

        Args:
            env: The environment.

        Returns:
            Tuple[float, int]: (episode_return, timestep)
        """
        # ========= put your code here ========= #
        obs, _ = env.reset()

        if isinstance(obs, dict):
            obs = obs.get("policy", next(iter(obs.values())))
        if isinstance(obs, torch.Tensor):
            obs = obs.detach().cpu().numpy()
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)

        if self.w.shape[0] != obs.shape[0]:
            self.w = np.zeros((obs.shape[0], self.num_of_action), dtype=np.float32)

        episode_return = 0.0
        timestep = 0

        while True:
            timestep += 1
            env_action, action_idx = self.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(env_action)

            if isinstance(next_obs, dict):
                next_obs = next_obs.get("policy", next(iter(next_obs.values())))

            if isinstance(next_obs, torch.Tensor):
                next_obs = next_obs.detach().cpu().numpy()
            if isinstance(reward, torch.Tensor):
                reward = reward.detach().cpu().numpy()
            if isinstance(terminated, torch.Tensor):
                terminated = terminated.detach().cpu().numpy()
            if isinstance(truncated, torch.Tensor):
                truncated = truncated.detach().cpu().numpy()

            next_obs = np.asarray(next_obs, dtype=np.float32).reshape(-1)
            reward = float(np.asarray(reward, dtype=np.float32))
            terminated = bool(np.asarray(terminated).item())
            truncated = bool(np.asarray(truncated).item())
            done = terminated or truncated
            next_q_values = np.asarray(self.q(next_obs), dtype=np.float32)
            next_action_idx = int(np.argmax(next_q_values))

            self.update(obs, action_idx, reward, next_obs, next_action_idx, done)

            episode_return += float(reward)
            obs = next_obs

            if done:
                break

        return float(episode_return), timestep
        # ====================================== #

    # ------------------------------------------------------------------ #
    # Persistence — linear weights only                                    #
    # ------------------------------------------------------------------ #

    def save_model(self, path: str, filename: str) -> None:
        """
        Save the weight matrix self.w to disk as a .npy file.

        Args:
            path (str): Directory to save the file.
            filename (str): File name (e.g., 'linear_q_cartpole.npy').
        """
        # ========= put your code here ========= #
        os.makedirs(path, exist_ok=True)
        # Added logic: normalize filenames so Linear_Q weights are always stored as .npy.
        root, ext = os.path.splitext(filename)
        if ext != ".npy":
            filename = f"{root or filename}.npy"
        save_path = os.path.join(path, filename)
        np.save(save_path, self.w)
        # ====================================== #

    def load_model(self, path: str, filename: str) -> None:
        """
        Load the weight matrix self.w from a .npy file.

        Args:
            path (str): Directory containing the file.
            filename (str): File name (e.g., 'linear_q_cartpole.npy').
        """
        # ========= put your code here ========= #
        # Added logic: normalize filenames so Linear_Q weights are always loaded from .npy.
        root, ext = os.path.splitext(filename)
        if ext != ".npy":
            filename = f"{root or filename}.npy"
        load_path = os.path.join(path, filename)
        self.w = np.load(load_path)
        # ====================================== #
