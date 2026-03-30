from __future__ import annotations
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from RL_Algorithm.storage.off_policy import OffPolicyAlgorithm


class TD3_Actor(nn.Module):
    """
    Deterministic actor network for TD3.

    Args:
        n_observations (int): Observation space dimension.
        hidden_dim (int): Hidden layer width.
        n_actions (int): Action space dimension.
    """

    def __init__(self, n_observations: int, hidden_dim: int, n_actions: int):
        super(TD3_Actor, self).__init__()
        # ========= put your code here ========= #
        self.net = nn.Sequential(
            nn.Linear(n_observations, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
            nn.Tanh(),
        )
        # ====================================== #

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            state (Tensor): State tensor.

        Returns:
            Tensor: Deterministic action in [-1, 1] (scale externally).
        """
        # ========= put your code here ========= #
        return self.net(state)
        # ====================================== #


class TD3_Critic(nn.Module):
    """
    Q-value network for TD3.
    Args:
        n_observations (int): Observation space dimension.
        n_actions (int): Action space dimension.
        hidden_dim (int): Hidden layer width.
    """

    def __init__(self, n_observations: int, n_actions: int, hidden_dim: int):
        super(TD3_Critic, self).__init__()

        # ===== Q1 network ===== #
        # ========= put your code here ========= #
        input_dim = n_observations + n_actions
        self.q1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # ====================================== #

        # ===== Q2 network (independent weights) ===== #
        # ========= put your code here ========= #
        self.q2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # ====================================== #

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        """
        Compute Q1 and Q2 values for a (state, action) pair.

        Args:
            state (Tensor): State tensor.
            action (Tensor): Action tensor.

        Returns:
            Tuple[Tensor, Tensor]: (Q1, Q2) both of shape (batch, 1).
        """
        # ========= put your code here ========= #
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)
        # ====================================== #

    def Q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Return only Q1 — used for the actor update.

        The actor maximises Q1 only (not min(Q1, Q2)) because at policy
        update time we want the gradient signal from one network, not a
        min operation which would break the gradient flow.

        Args:
            state (Tensor): State tensor.
            action (Tensor): Action tensor.

        Returns:
            Tensor: Q1 value of shape (batch, 1).
        """
        # ========= put your code here ========= #
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa)
        # ====================================== #


class TD3(OffPolicyAlgorithm):
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3).

    Args:
        device: Torch device.
        num_of_action (int): Action space dimension.
        action_range (list): [min, max] for action scaling.
        n_observations (int): Observation space dimension.
        hidden_dim (int): Hidden layer width.
        learning_rate (float): Learning rate for both actor and critics.
        tau (float): Polyak soft-update coefficient.
        discount_factor (float): Discount factor γ.
        buffer_size (int): Replay buffer capacity.
        batch_size (int): Mini-batch size per update.
        exploration_noise (float): Std of Gaussian noise added during interaction.
        target_noise (float): Std of smoothing noise added to target actions.
        target_noise_clip (float): Clip range for target smoothing noise.
        policy_update_freq (int): Critic steps between each actor update.
    """

    def __init__(
            self,
            device=None,
            num_of_action: int = None,
            action_range: list = [None, None],
            n_observations: int = None,
            hidden_dim: int = None,
            learning_rate: float = None,
            tau: float = None,
            discount_factor: float = None,
            buffer_size: int = None,
            batch_size: int = None,
            exploration_noise: float = None,
            target_noise: float = None,
            target_noise_clip: float = None,
            policy_update_freq: int = None,
    ) -> None:

        # Feel free to add or modify any of the initialized variables above.
        # ========= put your code here ========= #
        self.actor = TD3_Actor(n_observations, hidden_dim, num_of_action).to(device)
        self.actor_target = TD3_Actor(n_observations, hidden_dim, num_of_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = TD3_Critic(n_observations, num_of_action, hidden_dim).to(device)
        self.critic_target = TD3_Critic(n_observations, num_of_action, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

        self.device = device
        self.tau = tau
        self.exploration_noise = exploration_noise
        self.target_noise = target_noise
        self.target_noise_clip = target_noise_clip
        self.policy_update_freq = policy_update_freq
        self.total_steps = 0
        self.action_low = float(action_range[0])
        self.action_high = float(action_range[1])
        self.last_actor_loss = 0.0
        self.last_critic_loss = 0.0
        # ====================================== #

        # OffPolicyAlgorithm.__init__ creates self.memory = ReplayBuffer(buffer_size, batch_size)
        super(TD3, self).__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            buffer_size=buffer_size,
            batch_size=batch_size,
        )

    # ------------------------------------------------------------------ #
    # Core algorithm methods                                               #
    # ------------------------------------------------------------------ #

    def select_action(self, state: torch.Tensor, add_noise: bool = True):
        """
        Select a deterministic action with optional Gaussian exploration noise.

        Args:
            state (Tensor): Current state.
            add_noise (bool): Add exploration noise during training.
                              Set False for evaluation / play.

        Returns:
            Tensor: Action tensor of shape (action_dim,).
        """
        # ========= put your code here ========= #
        if isinstance(state, dict):
            state = state.get("policy", next(iter(state.values())))
        if isinstance(state, torch.Tensor):
            state_tensor = state.detach().to(self.device, dtype=torch.float32).view(1, -1)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).view(1, -1)
        with torch.no_grad():
            action = self.actor(state_tensor)
            action = self.action_low + 0.5 * (action + 1.0) * (self.action_high - self.action_low)
            if add_noise:
                action = action + torch.randn_like(action) * self.exploration_noise
            action = torch.clamp(action, self.action_low, self.action_high)
        return action
        # ====================================== #

    def calculate_loss(self, states, actions, rewards, next_states, dones):
        """
        Compute critic and actor loss for one mini-batch.

        Args:
            states (Tensor): Batch of current states.
            actions (Tensor): Batch of actions taken.
            rewards (Tensor): Batch of rewards.
            next_states (Tensor): Batch of next states.
            dones (Tensor): Batch of terminal flags.

        Returns:
            Tuple[Tensor, Tensor | None]: (critic_loss, actor_loss or None)
        """
        # ========= put your code here ========= #
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            next_actions = self.action_low + 0.5 * (next_actions + 1.0) * (self.action_high - self.action_low)
            noise = (torch.randn_like(next_actions) * self.target_noise).clamp(
                -self.target_noise_clip, self.target_noise_clip
            )
            next_actions = torch.clamp(next_actions + noise, self.action_low, self.action_high)
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = rewards + self.discount_factor * (1.0 - dones) * torch.min(target_q1, target_q2)

        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        actor_loss = None
        if self.total_steps % self.policy_update_freq == 0:
            actor_actions = self.actor(states)
            actor_actions = self.action_low + 0.5 * (actor_actions + 1.0) * (self.action_high - self.action_low)
            actor_loss = -self.critic.Q1(states, actor_actions).mean()
        return critic_loss, actor_loss
        # ====================================== #

    def generate_sample(self, batch_size=None):
        """
        Sample a mini-batch and unpack into TD3-ready tensors.

        Returns:
            Tuple or None:
                - states (Tensor)
                - actions (Tensor)
                - rewards (Tensor)
                - next_states (Tensor)
                - dones (Tensor)
        """
        # ========= put your code here ========= #
        batch = super().generate_sample()
        if batch is None:
            return None
        states = torch.tensor(
            [transition.state for transition in batch],
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.tensor(
            [transition.action for transition in batch],
            dtype=torch.float32,
            device=self.device,
        ).view(len(batch), -1)
        rewards = torch.tensor(
            [transition.reward for transition in batch],
            dtype=torch.float32,
            device=self.device,
        ).view(-1, 1)
        next_states = torch.tensor(
            [transition.next_state for transition in batch],
            dtype=torch.float32,
            device=self.device,
        )
        dones = torch.tensor(
            [transition.done for transition in batch],
            dtype=torch.float32,
            device=self.device,
        ).view(-1, 1)
        return states, actions, rewards, next_states, dones
        # ====================================== #

    def update_policy(self):
        """
        Perform one critic update and (if scheduled) one actor update.

        Returns:
            float | None: Critic loss value, or None if buffer not ready.
        """
        sample = self.generate_sample()
        if sample is None:
            return None

        states, actions, rewards, next_states, dones = sample
        critic_loss, actor_loss = self.calculate_loss(
            states, actions, rewards, next_states, dones
        )
        # ========= put your code here ========= #
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss_value = self.last_actor_loss
        if actor_loss is not None:
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self.update_target_networks()
            actor_loss_value = float(actor_loss.item())

        self.last_critic_loss = float(critic_loss.item())
        self.last_actor_loss = actor_loss_value
        # ====================================== #

        self.total_steps += 1
        return {
            "critic_loss": self.last_critic_loss,
            "actor_loss": self.last_actor_loss,
        }

    def update_target_networks(self):
        # ========= put your code here ========= #
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        # ====================================== #

    def learn(self, env, num_agents: int = 1, max_steps: int = 1000):
        """
        Train the agent for one episode (single env) or fixed-length run
        (parallel envs).

        Args:
            env: The Isaac Lab environment.
            num_agents (int): Number of parallel environments.
            max_steps (int): Steps per episode (single) or total steps (parallel).

        Returns:
            Tuple[float, int]: (episode_return, timestep)
        """
        # ========= put your code here ========= #
        del num_agents, max_steps
        obs, _ = env.reset()
        episode_return = 0.0
        timestep = 0
        critic_losses = []
        actor_losses = []

        while True:
            action = self.select_action(obs, add_noise=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)

            if isinstance(obs, dict):
                obs = obs.get("policy", next(iter(obs.values())))
            if isinstance(next_obs, dict):
                next_obs = next_obs.get("policy", next(iter(next_obs.values())))
            obs_array = (
                obs.detach().cpu().numpy().reshape(-1)
                if isinstance(obs, torch.Tensor)
                else torch.as_tensor(obs, dtype=torch.float32).view(-1).cpu().numpy()
            )
            next_obs_array = (
                next_obs.detach().cpu().numpy().reshape(-1)
                if isinstance(next_obs, torch.Tensor)
                else torch.as_tensor(next_obs, dtype=torch.float32).view(-1).cpu().numpy()
            )

            reward_value = float(reward.detach().cpu().item()) if isinstance(reward, torch.Tensor) else float(reward)
            done = (
                bool(terminated.detach().cpu().item()) if isinstance(terminated, torch.Tensor) else bool(terminated)
            ) or (
                bool(truncated.detach().cpu().item()) if isinstance(truncated, torch.Tensor) else bool(truncated)
            )

            self.store_transition(
                obs_array,
                action.squeeze(0).detach().cpu().numpy(),
                reward_value,
                next_obs_array,
                done,
            )
            info = self.update_policy()
            if info is not None:
                critic_losses.append(info["critic_loss"])
                actor_losses.append(info["actor_loss"])

            episode_return += reward_value
            timestep += 1
            obs = next_obs
            if done:
                break

        mean_critic = sum(critic_losses) / len(critic_losses) if critic_losses else 0.0
        mean_actor = sum(actor_losses) / len(actor_losses) if actor_losses else 0.0
        self.last_critic_loss = mean_critic
        self.last_actor_loss = mean_actor
        return episode_return, timestep, {
            "critic_loss": mean_critic,
            "actor_loss": mean_actor,
        }
        # ====================================== #

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save_model(self, path: str, filename: str) -> None:
        """
        Save actor and critic weights.

        Args:
            path (str): Directory to save.
            filename (str): File name (e.g., 'td3_cartpole.pth').
        """
        # ========= put your code here ========= #
        os.makedirs(path, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "total_steps": self.total_steps,
            },
            os.path.join(path, filename),
        )
        # ====================================== #

    def load_model(self, path: str, filename: str) -> None:
        """
        Load actor and critic weights.

        Args:
            path (str): Directory of saved model.
            filename (str): File name (e.g., 'td3_cartpole.pth').
        """
        # ========= put your code here ========= #
        checkpoint = torch.load(os.path.join(path, filename), map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.actor_target.load_state_dict(checkpoint["actor_target"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.total_steps = checkpoint.get("total_steps", 0)
        # ====================================== #
