from __future__ import annotations
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from RL_Algorithm.RL_base_function import BaseAlgorithm


# ============================================================ #
# ==================== Policy Network ======================== #
# ============================================================ #

class MC_REINFORCE_network(nn.Module):
    """
    Policy network for the MC REINFORCE algorithm.

    Args:
        n_observations (int): Number of input features.
        hidden_size (int): Number of hidden neurons per layer.
        n_actions (int): Number of output values.
                         Discrete  → number of action choices.
                         Continuous → dimension of the action vector.
        dropout (float): Dropout rate for regularization.
        action_type (str): ``'discrete'`` or ``'continuous'``.
    """

    def __init__(
        self,
        n_observations: int,
        hidden_size: int,
        n_actions: int,
        dropout: float,
        action_type: str = "discrete",
    ):
        super(MC_REINFORCE_network, self).__init__()

        assert action_type in ("discrete", "continuous"), \
            f"action_type must be 'discrete' or 'continuous', got '{action_type}'"

        self.action_type = action_type

        # ===== Shared MLP body ===== #
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

        # ===== Learnable log_std (continuous only) ===== #
        # Initialise to 0 so std starts at exp(0) = 1.0
        if self.action_type == "continuous":
            self.log_std = nn.Parameter(torch.zeros(n_actions))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Discrete  : returns logits of shape ``(batch, n_actions)``.
        Continuous: returns action mean of shape ``(batch, n_actions)``.
                    Use ``self.log_std`` separately to build the Normal distribution.

        Args:
            x (Tensor): Input state tensor of shape ``(batch, n_observations)``.

        Returns:
            Tensor: Logits (discrete) or action mean (continuous).
        """
        # ========= put your code here ========= #
        return self.net(x)
        # ====================================== #


# ============================================================ #
# =================== MC REINFORCE Agent ===================== #
# ============================================================ #

class MC_REINFORCE(BaseAlgorithm):
    """
    Monte-Carlo REINFORCE policy gradient algorithm supporting both
    discrete and continuous action spaces.

    Args:
        device: Torch device.
        num_of_action (int): Action dim (continuous) or number of choices (discrete).
        action_range (list): [min, max] for continuous action scaling.
                             Ignored for discrete.
        n_observations (int): Observation space dimension.
        hidden_dim (int): Hidden layer width.
        dropout (float): Dropout rate.
        action_type (str): ``'discrete'`` or ``'continuous'``.
        learning_rate (float): AdamW learning rate.
        discount_factor (float): Discount factor γ.
    """

    def __init__(
            self,
            device=None,
            num_of_action: int = None,
            action_range: list = [None, None],
            n_observations: int = None,
            hidden_dim: int = None,
            dropout: float = None,
            action_type: str = None,
            learning_rate: float = None,
            discount_factor: float = None,
    ) -> None:

        assert action_type in ("discrete", "continuous"), \
            f"action_type must be 'discrete' or 'continuous', got '{action_type}'"

        # Feel free to add or modify any of the initialized variables above.
        # ========= put your code here ========= #
        if device is None:
            device = torch.device("cpu")
        self.action_type = action_type
        self.LR = learning_rate

        self.policy_net = MC_REINFORCE_network(
            n_observations, hidden_dim, num_of_action, dropout, action_type
        ).to(device)
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=learning_rate)

        self.device = device
        self.steps_done = 0
        # ====================================== #

        super(MC_REINFORCE, self).__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
        )

    # ------------------------------------------------------------------ #
    # Distribution helpers                                                 #
    # ------------------------------------------------------------------ #

    def _get_distribution(self, obs: torch.Tensor):
        """
        Build the action distribution from the current observation.

        Args:
            obs (Tensor): State tensor of shape ``(batch, obs_dim)``.

        Returns:
            torch.distributions.Distribution: Categorical or Normal distribution.
        """
        # ========= put your code here ========= #
        logits_or_mean = self.policy_net(obs)
        if self.action_type == "discrete":
            return Categorical(logits=logits_or_mean)

        std = self.policy_net.log_std.exp().expand_as(logits_or_mean)
        return Normal(logits_or_mean, std)
        # ====================================== #

    def _sample_action(self, dist) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample an action from the distribution and compute its log-probability.

        Args:
            dist: A ``Categorical`` or ``Normal`` distribution object.

        Returns:
            Tuple[Tensor, Tensor]:
                - action  : Discrete: shape ``(batch, 1)``.
                            Continuous: shape ``(batch, action_dim)``.
                - log_prob: Shape ``(batch,)``.
        """
        # ========= put your code here ========= #
        action = dist.sample()
        if self.action_type == "discrete":
            return action.view(-1, 1), dist.log_prob(action)

        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob
        # ====================================== #

    # ------------------------------------------------------------------ #
    # Core algorithm methods                                               #
    # ------------------------------------------------------------------ #

    def calculate_stepwise_returns(self, rewards: list) -> torch.Tensor:
        """
        Compute normalised discounted returns G_t for each timestep.

        Args:
            rewards (list): Rewards collected in the episode.

        Returns:
            Tensor: Normalised return tensor of shape ``(T,)``.
        """
        # ========= put your code here ========= #
        discounted_returns = []
        running_return = 0.0
        for reward in reversed(rewards):
            running_return = reward + self.discount_factor * running_return
            discounted_returns.insert(0, running_return)

        returns = torch.tensor(discounted_returns, dtype=torch.float32, device=self.device)
        if returns.numel() > 1:
            returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)
        return returns
        # ====================================== #

    def generate_trajectory(self, env):
        """
        Run one full episode and collect the trajectory.

        Args:
            env: The environment.

        Returns:
            Tuple:
                - episode_return (float)
                - stepwise_returns (Tensor): shape ``(T,)``
                - log_prob_actions (Tensor): shape ``(T,)``
                - trajectory (list): ``[(state, action, reward), ...]``
        """
        # ========= put your code here ========= #
        obs, _ = env.reset()
        if isinstance(obs, dict):
            obs = obs.get("policy", next(iter(obs.values())))
        if isinstance(obs, torch.Tensor):
            obs_np = obs.detach().cpu().numpy().reshape(-1)
        else:
            obs_np = torch.as_tensor(obs, dtype=torch.float32).view(-1).cpu().numpy()

        rewards = []
        log_probs = []
        trajectory = []
        episode_return = 0.0

        while True:
            obs_tensor = torch.tensor(obs_np, dtype=torch.float32, device=self.device).view(1, -1)
            dist = self._get_distribution(obs_tensor)
            action_tensor, log_prob = self._sample_action(dist)

            if self.action_type == "discrete":
                action_idx = int(action_tensor.item())
                action_min, action_max = self.action_range
                if self.num_of_action == 1:
                    env_action = torch.tensor([[action_min]], dtype=torch.float32, device=self.device)
                else:
                    scaled_value = action_min + (action_idx / (self.num_of_action - 1)) * (action_max - action_min)
                    env_action = torch.tensor([[scaled_value]], dtype=torch.float32, device=self.device)
            else:
                env_action = action_tensor
                action_idx = env_action.detach().cpu().numpy()

            next_obs, reward, terminated, truncated, _ = env.step(env_action)
            if isinstance(next_obs, dict):
                next_obs = next_obs.get("policy", next(iter(next_obs.values())))

            reward_value = float(reward.detach().cpu().item()) if isinstance(reward, torch.Tensor) else float(reward)
            terminated_flag = bool(terminated.detach().cpu().item()) if isinstance(terminated, torch.Tensor) else bool(terminated)
            truncated_flag = bool(truncated.detach().cpu().item()) if isinstance(truncated, torch.Tensor) else bool(truncated)
            done = terminated_flag or truncated_flag

            rewards.append(reward_value)
            log_probs.append(log_prob.squeeze(0))
            trajectory.append((obs_np.copy(), action_idx, reward_value))
            episode_return += reward_value

            if isinstance(next_obs, torch.Tensor):
                obs_np = next_obs.detach().cpu().numpy().reshape(-1)
            else:
                obs_np = torch.as_tensor(next_obs, dtype=torch.float32).view(-1).cpu().numpy()

            if done:
                break

        stepwise_returns = self.calculate_stepwise_returns(rewards)
        log_prob_actions = torch.stack(log_probs)
        return episode_return, stepwise_returns, log_prob_actions, trajectory
        # ====================================== #

    def calculate_loss(
        self,
        stepwise_returns: torch.Tensor,
        log_prob_actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute REINFORCE policy gradient loss.

        Args:
            stepwise_returns (Tensor): shape ``(T,)``.
            log_prob_actions (Tensor): shape ``(T,)``.

        Returns:
            Tensor: Scalar loss.
        """
        # ========= put your code here ========= #
        return -(stepwise_returns * log_prob_actions).sum()
        # ====================================== #

    def update_policy(
        self,
        stepwise_returns: torch.Tensor,
        log_prob_actions: torch.Tensor,
    ) -> float:
        """
        Backpropagate the REINFORCE loss and update the policy network.

        Args:
            stepwise_returns (Tensor): shape ``(T,)``.
            log_prob_actions (Tensor): shape ``(T,)``.

        Returns:
            float: Loss value after the update.
        """
        # ========= put your code here ========= #
        loss = self.calculate_loss(stepwise_returns, log_prob_actions)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())
        # ====================================== #

    def learn(self, env, num_agents: int = 1):
        """
        Train the agent for one episode.

        Args:
            env: The environment.
            num_agents (int): Number of parallel agents (>1 for vectorised envs).

        Returns:
            Tuple: (episode_return, loss, trajectory)
        """
        self.policy_net.train()

        # ========= put your code here ========= #
        del num_agents  # MC_REINFORCE here is implemented for a single environment.
        episode_return, stepwise_returns, log_prob_actions, trajectory = self.generate_trajectory(env)
        loss = self.update_policy(stepwise_returns, log_prob_actions)
        return episode_return, loss, trajectory
        # ====================================== #

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save_model(self, path: str, filename: str) -> None:
        """
        Save policy network weights to disk.

        Args:
            path (str): Directory to save.
            filename (str): File name (e.g., ``'reinforce_cartpole.pth'``).
        """
        # ========= put your code here ========= #
        os.makedirs(path, exist_ok=True)
        torch.save(self.policy_net.state_dict(), os.path.join(path, filename))
        # ====================================== #

    def load_model(self, path: str, filename: str) -> None:
        """
        Load policy network weights from disk.

        Args:
            path (str): Directory of saved model.
            filename (str): File name (e.g., ``'reinforce_cartpole.pth'``).
        """
        # ========= put your code here ========= #
        self.policy_net.load_state_dict(torch.load(os.path.join(path, filename), map_location=self.device))
        self.policy_net.to(self.device)
        self.policy_net.eval()
        # ====================================== #
