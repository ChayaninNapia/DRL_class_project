from __future__ import annotations
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from RL_Algorithm.storage.off_policy import OffPolicyAlgorithm


class DQN_network(nn.Module):
    """
    Neural network model for the Deep Q-Network algorithm.

    Args:
        n_observations (int): Number of input features.
        hidden_size (int): Number of hidden neurons.
        n_actions (int): Number of possible actions.
        dropout (float): Dropout rate for regularization.
    """

    def __init__(self, n_observations, hidden_size, n_actions, dropout):
        super(DQN_network, self).__init__()
        # ========= put your code here ========= #
        self.net = nn.Sequential(
            nn.Linear(n_observations, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_actions),
        )
        # ====================================== #

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (Tensor): Input state tensor.

        Returns:
            Tensor: Q-value estimates for each action.
        """
        # ========= put your code here ========= #
        return self.net(x)
        # ====================================== #


class DQN(OffPolicyAlgorithm):
    """
    Deep Q-Network (DQN) — off-policy, value-based.

    Args:
        device: Torch device.
        num_of_action (int): Number of discrete actions.
        action_range (list): [min, max] for continuous action scaling.
        n_observations (int): Observation space dimension.
        hidden_dim (int): Hidden layer width.
        dropout (float): Dropout rate.
        learning_rate (float): Adam learning rate.
        tau (float): Polyak soft-update coefficient for target network.
        initial_epsilon (float): Starting exploration rate.
        epsilon_decay (float): Per-step epsilon decay.
        final_epsilon (float): Minimum exploration rate.
        discount_factor (float): Discount factor γ.
        buffer_size (int): Replay buffer capacity.
        batch_size (int): Mini-batch size per update.
    """

    def __init__(
            self,
            device=None,
            num_of_action: int = None,
            action_range: list = [None, None],
            n_observations: int = None,
            hidden_dim: int = None,
            dropout: float = None,
            learning_rate: float = None,
            tau: float = None,
            initial_epsilon: float = None,
            epsilon_decay: float = None,
            final_epsilon: float = None,
            discount_factor: float = None,
            buffer_size: int = None,
            batch_size: int = None,
    ) -> None:

        # Feel free to add or modify any of the initialized variables above.
        # ========= put your code here ========= #
        if device is None:
            device = torch.device("cpu")

        self.policy_net = DQN_network(n_observations, hidden_dim, num_of_action, dropout).to(device)
        self.target_net = DQN_network(n_observations, hidden_dim, num_of_action, dropout).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.device = device
        self.steps_done = 0
        self.tau = tau

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=learning_rate, amsgrad=True)
        # ====================================== #

        super(DQN, self).__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            initial_epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            discount_factor=discount_factor,
            buffer_size=buffer_size,
            batch_size=batch_size,
        )

    # ------------------------------------------------------------------ #
    # Core algorithm methods                                               #
    # ------------------------------------------------------------------ #

    def select_action(self, state):
        """
        Select an action using an epsilon-greedy policy.

        Args:
            state (Tensor): Current state.

        Returns:
            Tuple[Tensor, int]: Scaled action tensor and action index.
        """
        # ========= put your code here ========= #
        state_device = state.device if isinstance(state, torch.Tensor) else None
        if isinstance(state, dict):
            state = state.get("policy", next(iter(state.values())))
        if isinstance(state, torch.Tensor):
            state = state.detach().to(self.device, dtype=torch.float32).view(1, -1)
        else:
            state = torch.tensor(state, dtype=torch.float32, device=self.device).view(1, -1)

        sample = torch.rand(1).item()
        if sample < self.epsilon:
            action_idx = torch.randint(self.num_of_action, (1,), device=self.device).item()
        else:
            with torch.no_grad():
                action_idx = self.policy_net(state).argmax(dim=1).item()

        action_min, action_max = self.action_range
        if self.num_of_action == 1:
            scaled_action = [[action_min]]
        else:
            scaled_value = action_min + (action_idx / (self.num_of_action - 1)) * (action_max - action_min)
            scaled_action = [[scaled_value]]

        env_action = torch.tensor(
            scaled_action,
            dtype=torch.float32,
            device=state_device if state_device is not None else self.device,
        )
        return env_action, int(action_idx)
        # ====================================== #

    def calculate_loss(self, non_final_mask, non_final_next_states, state_batch, action_batch, reward_batch):
        """
        Compute the Bellman loss for a sampled mini-batch.

        Args:
            non_final_mask (Tensor): True where next state is not terminal.
            non_final_next_states (Tensor): Non-terminal next states.
            state_batch (Tensor): Batch of current states.
            action_batch (Tensor): Batch of action indices.
            reward_batch (Tensor): Batch of rewards.

        Returns:
            Tensor: Scalar Huber / MSE loss.
        """
        # ========= put your code here ========= #
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(state_batch.size(0), device=self.device)
        with torch.no_grad():
            if non_final_next_states.numel() > 0:
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(dim=1).values

        expected_state_action_values = reward_batch + self.discount_factor * next_state_values
        return F.smooth_l1_loss(state_action_values.squeeze(1), expected_state_action_values)
        # ====================================== #

    def generate_sample(self, batch_size=None):
        """
        Sample a mini-batch and unpack it into DQN-ready tensors.

        Returns:
            Tuple or None:
                - non_final_mask (Tensor)
                - non_final_next_states (Tensor)
                - state_batch (Tensor)
                - action_batch (Tensor)
                - reward_batch (Tensor)
            Returns None if the buffer is not ready.
        """
        # ========= put your code here ========= #
        batch = super().generate_sample()
        if batch is None:
            return None
        # ====================================== #

        # Unpack and prepare tensors from the Transition namedtuples
        # ========= put your code here ========= #
        non_final_mask = torch.tensor(
            [not transition.done for transition in batch],
            device=self.device,
            dtype=torch.bool,
        )

        non_final_next_states_list = [
            torch.as_tensor(transition.next_state, dtype=torch.float32, device=self.device).view(-1)
            for transition in batch
            if not transition.done
        ]
        if non_final_next_states_list:
            non_final_next_states = torch.stack(non_final_next_states_list)
        else:
            state_dim = torch.as_tensor(batch[0].state, dtype=torch.float32).numel()
            non_final_next_states = torch.empty((0, state_dim), dtype=torch.float32, device=self.device)

        state_batch = torch.stack([
            torch.as_tensor(transition.state, dtype=torch.float32, device=self.device).view(-1)
            for transition in batch
        ])
        action_batch = torch.tensor(
            [[int(transition.action)] for transition in batch],
            dtype=torch.long,
            device=self.device,
        )
        reward_batch = torch.tensor(
            [float(transition.reward) for transition in batch],
            dtype=torch.float32,
            device=self.device,
        )
        return non_final_mask, non_final_next_states, state_batch, action_batch, reward_batch
        # ====================================== #

    def update_policy(self):
        """Perform one gradient step on the policy network."""
        sample = self.generate_sample()
        if sample is None:
            return
        non_final_mask, non_final_next_states, state_batch, action_batch, reward_batch = sample
        loss = self.calculate_loss(non_final_mask, non_final_next_states, state_batch, action_batch, reward_batch)

        # ========= put your code here ========= #
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100.0)
        self.optimizer.step()
        self.update_target_networks()
        return float(loss.item())
        # ====================================== #

    def update_target_networks(self):
        # ========= put your code here ========= #
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = (
                self.tau * policy_net_state_dict[key] + (1.0 - self.tau) * target_net_state_dict[key]
            )
        self.target_net.load_state_dict(target_net_state_dict)
        # ====================================== #

    def learn(self, env, num_agents: int = 1):
        """
        Train the agent for one episode (single env) or one fixed-length
        run (parallel envs).

        Args:
            env: The Isaac Lab environment.
            num_agents (int): Number of parallel environments.
        Returns:
            Tuple[float, int, float]: (episode_return, timestep, mean_q_loss)
        """

        # ========= put your code here ========= #
        del num_agents  # DQN here is implemented for a single environment.

        obs, _ = env.reset()
        if isinstance(obs, dict):
            obs = obs.get("policy", next(iter(obs.values())))
        if isinstance(obs, torch.Tensor):
            obs = obs.detach().cpu().numpy()

        episode_return = 0.0
        timestep = 0
        losses = []

        while True:
            timestep += 1
            env_action, action_idx = self.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(env_action)

            if isinstance(next_obs, dict):
                next_obs = next_obs.get("policy", next(iter(next_obs.values())))
            if isinstance(next_obs, torch.Tensor):
                next_obs_np = next_obs.detach().cpu().numpy().reshape(-1)
            else:
                next_obs_np = torch.as_tensor(next_obs, dtype=torch.float32).view(-1).cpu().numpy()

            reward_value = float(reward.detach().cpu().item()) if isinstance(reward, torch.Tensor) else float(reward)
            terminated_flag = bool(terminated.detach().cpu().item()) if isinstance(terminated, torch.Tensor) else bool(terminated)
            truncated_flag = bool(truncated.detach().cpu().item()) if isinstance(truncated, torch.Tensor) else bool(truncated)
            done = terminated_flag or truncated_flag

            obs_np = obs.detach().cpu().numpy().reshape(-1) if isinstance(obs, torch.Tensor) else torch.as_tensor(obs, dtype=torch.float32).view(-1).cpu().numpy()
            self.store_transition(obs_np, action_idx, reward_value, next_obs_np, done)

            loss = self.update_policy()
            if loss is not None:
                losses.append(loss)
            self.decay_epsilon()

            episode_return += reward_value
            obs = next_obs_np

            if done:
                break

        mean_q_loss = float(sum(losses) / len(losses)) if losses else 0.0
        return float(episode_return), timestep, mean_q_loss
        # ====================================== #

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save_model(self, path: str, filename: str) -> None:
        """
        Save policy network weights.

        Args:
            path (str): Directory to save.
            filename (str): File name (e.g., 'dqn_cartpole.pth').
        """
        # ========= put your code here ========= #
        os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, filename)
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
            },
            save_path,
        )
        # ====================================== #

    def load_model(self, path: str, filename: str) -> None:
        """
        Load policy network weights and sync to target network.

        Args:
            path (str): Directory of saved model.
            filename (str): File name (e.g., 'dqn_cartpole.pth').
        """
        # ========= put your code here ========= #
        load_path = os.path.join(path, filename)
        checkpoint = torch.load(load_path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon)
        self.target_net.eval()
        # ====================================== #
