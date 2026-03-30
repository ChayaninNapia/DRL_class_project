from __future__ import annotations
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from RL_Algorithm.storage.off_policy import OffPolicyAlgorithm


class SAC_Actor(nn.Module):
    """
    Stochastic actor network for SAC using the reparameterisation trick.

    Args:
        n_observations (int): Observation space dimension.
        hidden_dim (int): Hidden layer width.
        n_actions (int): Action space dimension.
        log_std_min (float): Lower bound for log standard deviation.
        log_std_max (float): Upper bound for log standard deviation.
    """

    def __init__(
        self,
        n_observations: int,
        hidden_dim: int,
        n_actions: int,
        log_std_min: float = None,
        log_std_max: float = None,
    ):
        super(SAC_Actor, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # ========= put your code here ========= #
        self.backbone = nn.Sequential(
            nn.Linear(n_observations, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, n_actions)
        self.log_std_head = nn.Linear(hidden_dim, n_actions)
        # ====================================== #

    def forward(self, state: torch.Tensor):
        """
        Compute mean and log_std of the Gaussian policy.

        Args:
            state (Tensor): State tensor.

        Returns:
            Tuple[Tensor, Tensor]: (mean, log_std) both shape (batch, n_actions).
        """
        # ========= put your code here ========= #
        features = self.backbone(state)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features).clamp(self.log_std_min, self.log_std_max)
        return mean, log_std
        # ====================================== #

    def sample(self, state: torch.Tensor):
        """
        Sample an action using the reparameterisation trick and compute
        the corrected log-probability.

        Args:
            state (Tensor): State tensor.

        Returns:
            Tuple[Tensor, Tensor]:
                - action   : Squashed action in (-1, 1), shape (batch, n_actions).
                - log_prob : Corrected log π(a|s),       shape (batch,).
        """
        # ========= put your code here ========= #
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob
        # ====================================== #


class SAC_Critic(nn.Module):
    """
    Twin Q-value network for SAC.

    SAC uses two critics (same as TD3) to reduce overestimation.
    The soft Bellman target uses ``min(Q1, Q2) − α · log π(a'|s')``.

    Args:
        n_observations (int): Observation space dimension.
        n_actions (int): Action space dimension.
        hidden_dim (int): Hidden layer width.
    """

    def __init__(self, n_observations: int, n_actions: int, hidden_dim: int):
        super(SAC_Critic, self).__init__()

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
        Compute both Q-values.

        Args:
            state (Tensor): State tensor.
            action (Tensor): Action tensor.

        Returns:
            Tuple[Tensor, Tensor]: (Q1, Q2) both shape (batch, 1).
        """
        # ========= put your code here ========= #
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)
        # ====================================== #


class SAC(OffPolicyAlgorithm):
    """
    Soft Actor-Critic (SAC) — off-policy, maximum entropy actor-critic.

    Args:
        device: Torch device.
        num_of_action (int): Action space dimension.
        action_range (list): [min, max] for action scaling.
        n_observations (int): Observation space dimension.
        hidden_dim (int): Hidden layer width.
        learning_rate (float): Learning rate for actor and critics.
        alpha_lr (float): Learning rate for automatic temperature tuning.
        tau (float): Polyak soft-update coefficient.
        discount_factor (float): Discount factor γ.
        buffer_size (int): Replay buffer capacity.
        batch_size (int): Mini-batch size per update.
        init_alpha (float): Initial temperature α.
        auto_alpha (bool): Enable automatic α tuning.
        target_entropy (float | None): Target entropy for auto-tuning.
                                       Defaults to −action_dim if None.
    """

    def __init__(
            self,
            device=None,
            num_of_action: int = None,
            action_range: list = [None, None],
            n_observations: int = None,
            hidden_dim: int = None,
            learning_rate: float = None,
            alpha_lr: float = None,
            tau: float = None,
            discount_factor: float = None,
            buffer_size: int = None,
            batch_size: int = None,
            init_alpha: float = None,
            auto_alpha: bool = None,
            target_entropy: float | None = None,
    ) -> None:

        # Feel free to add or modify any of the initialized variables above.
        # ========= put your code here ========= #
        self.actor = SAC_Actor(
            n_observations,
            hidden_dim,
            num_of_action,
            log_std_min=-20.0,
            log_std_max=2.0,
        ).to(device)
        self.critic = SAC_Critic(n_observations, num_of_action, hidden_dim).to(device)
        self.critic_target = SAC_Critic(n_observations, num_of_action, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

        self.device = device
        self.tau = tau
        self.auto_alpha = auto_alpha
        self.action_low = float(action_range[0])
        self.action_high = float(action_range[1])
        self.last_actor_loss = 0.0
        self.last_critic_loss = 0.0
        self.last_alpha_loss = 0.0

        # ===== Automatic temperature tuning ===== #
        # log_alpha is optimised instead of alpha directly to keep alpha > 0.
        # target_entropy is set to -action_dim as a heuristic (Haarnoja et al. 2018).
        self.log_alpha = nn.Parameter(
            torch.log(torch.tensor([float(init_alpha)], dtype=torch.float32, device=device))
        )
        self.alpha = self.log_alpha.exp().item()
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy = target_entropy if target_entropy is not None \
                              else -float(num_of_action)
        # ====================================== #

        # OffPolicyAlgorithm.__init__ creates self.memory = ReplayBuffer(buffer_size, batch_size)
        super(SAC, self).__init__(
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

    def select_action(self, state: torch.Tensor, evaluate: bool = False):
        """
        Sample an action from the stochastic policy.

        Training  (evaluate=False): sample from Normal distribution.
        Inference (evaluate=True) : use the mean (deterministic).

        Args:
            state (Tensor): Current state.
            evaluate (bool): True for deterministic inference.

        Returns:
            Tensor: Scaled action tensor.
        """
        # ========= put your code here ========= #
        if isinstance(state, dict):
            state = state.get("policy", next(iter(state.values())))
        if isinstance(state, torch.Tensor):
            state_tensor = state.detach().to(self.device, dtype=torch.float32).view(1, -1)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).view(1, -1)
        with torch.no_grad():
            if evaluate:
                mean, _ = self.actor(state_tensor)
                action = torch.tanh(mean)
            else:
                action, _ = self.actor.sample(state_tensor)
            action = self.action_low + 0.5 * (action + 1.0) * (self.action_high - self.action_low)
        return torch.clamp(action, self.action_low, self.action_high)
        # ====================================== #

    def calculate_loss(self, states, actions, rewards, next_states, dones):
        """
        Compute SAC losses for critics, actor, and temperature.

        Args:
            states (Tensor): Batch of current states.
            actions (Tensor): Batch of actions taken.
            rewards (Tensor): Batch of rewards.
            next_states (Tensor): Batch of next states.
            dones (Tensor): Batch of terminal flags.

        Returns:
            Tuple[Tensor, Tensor, Tensor | None]:
                (critic_loss, actor_loss, alpha_loss or None)
        """
        with torch.no_grad():
        # ========= put your code here ========= #
            next_actions, next_log_prob = self.actor.sample(next_states)
            next_actions = self.action_low + 0.5 * (next_actions + 1.0) * (self.action_high - self.action_low)
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_v = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_q = rewards + self.discount_factor * (1.0 - dones) * target_v
        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        sampled_actions, log_prob = self.actor.sample(states)
        sampled_actions = self.action_low + 0.5 * (sampled_actions + 1.0) * (self.action_high - self.action_low)
        q1_pi, q2_pi = self.critic(states, sampled_actions)
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha * log_prob - min_q_pi).mean()

        alpha_loss = None
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        return critic_loss, actor_loss, alpha_loss
        # ====================================== #

    def generate_sample(self, batch_size=None):
        """
        Sample a mini-batch and unpack into SAC-ready tensors.

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
        states = torch.as_tensor(
            np.array([transition.state for transition in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.as_tensor(
            np.array([transition.action for transition in batch]),
            dtype=torch.float32,
            device=self.device,
        ).view(len(batch), -1)
        rewards = torch.as_tensor(
            np.array([transition.reward for transition in batch]),
            dtype=torch.float32,
            device=self.device,
        ).view(-1, 1)
        next_states = torch.as_tensor(
            np.array([transition.next_state for transition in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        dones = torch.as_tensor(
            np.array([transition.done for transition in batch]),
            dtype=torch.float32,
            device=self.device,
        ).view(-1, 1)
        return states, actions, rewards, next_states, dones
        # ====================================== #

    def update_policy(self):
        """
        Perform one update step for critics, actor, and temperature.

        Returns:
            float | None: Critic loss, or None if buffer not ready.
        """
        sample = self.generate_sample()
        if sample is None:
            return None

        states, actions, rewards, next_states, dones = sample
        critic_loss, actor_loss, alpha_loss = self.calculate_loss(
            states, actions, rewards, next_states, dones
        )
        # ========= put your code here ========= #
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        for param in self.critic.parameters():
            param.requires_grad = False

        sampled_actions, log_prob = self.actor.sample(states)
        sampled_actions = self.action_low + 0.5 * (sampled_actions + 1.0) * (self.action_high - self.action_low)
        q1_pi, q2_pi = self.critic(states, sampled_actions)
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha * log_prob - min_q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for param in self.critic.parameters():
            param.requires_grad = True

        alpha_loss_value = self.last_alpha_loss
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha_loss_value = float(alpha_loss.item())

        self.last_critic_loss = float(critic_loss.item())
        self.last_actor_loss = float(actor_loss.item())
        self.last_alpha_loss = alpha_loss_value
        # ====================================== #

        self.alpha = self.log_alpha.exp().item()

        self.update_target_networks()
        return {
            "critic_loss": self.last_critic_loss,
            "actor_loss": self.last_actor_loss,
            "alpha_loss": self.last_alpha_loss,
            "alpha": self.alpha,
        }

    def update_target_networks(self):
        """
        Overrides the no-op in OffPolicyAlgorithm.
        """
        # ========= put your code here ========= #
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
        alpha_losses = []

        while True:
            action = self.select_action(obs, evaluate=False)
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
                alpha_losses.append(info["alpha_loss"])

            episode_return += reward_value
            timestep += 1
            obs = next_obs
            if done:
                break

        mean_critic = sum(critic_losses) / len(critic_losses) if critic_losses else 0.0
        mean_actor = sum(actor_losses) / len(actor_losses) if actor_losses else 0.0
        mean_alpha_loss = sum(alpha_losses) / len(alpha_losses) if alpha_losses else 0.0
        self.last_critic_loss = mean_critic
        self.last_actor_loss = mean_actor
        self.last_alpha_loss = mean_alpha_loss
        return episode_return, timestep, {
            "critic_loss": mean_critic,
            "actor_loss": mean_actor,
            "alpha_loss": mean_alpha_loss,
            "alpha": self.alpha,
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
            filename (str): File name (e.g., 'sac_cartpole.pth').
        """
        # ========= put your code here ========= #
        os.makedirs(path, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
                "alpha_optimizer": self.alpha_optimizer.state_dict(),
            },
            os.path.join(path, filename),
        )
        # ====================================== #

    def load_model(self, path: str, filename: str) -> None:
        """
        Load actor and critic weights.

        Args:
            path (str): Directory of saved model.
            filename (str): File name (e.g., 'sac_cartpole.pth').
        """
        # ========= put your code here ========= #
        checkpoint = torch.load(os.path.join(path, filename), map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.log_alpha.data.copy_(checkpoint["log_alpha"].to(self.device))
        self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
        self.alpha = self.log_alpha.exp().item()
        # ====================================== #
