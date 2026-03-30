from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from RL_Algorithm.storage.on_policy import OnPolicyAlgorithm
from RL_Algorithm.networks.mlp import MLP


# ============================================================ #
# =================== Actor-Critic Network =================== #
# ============================================================ #

class ActorCritic(nn.Module):
    """
    Combined Actor-Critic network supporting continuous and discrete actions.

    ``action_type='continuous'`` → ``Normal(mean, std)``, learnable ``self.std``
    ``action_type='discrete'``   → ``Categorical(logits)``, no ``self.std``

    Args:
        state_dim (int): Observation space dimension.
        action_dim (int): Action vector dim (continuous) or # choices (discrete).
        hidden_dims (list[int]): MLP hidden layer sizes.
        activation (str): Activation function.
        action_type (str): ``'continuous'`` or ``'discrete'``.
        init_noise_std (float): Initial std for continuous distribution.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list[int] = [None],
        activation: str = None,
        action_type: str = None,
        init_noise_std: float = None,
    ):
        super().__init__()

        assert action_type in ("continuous", "discrete"), \
            f"action_type must be 'continuous' or 'discrete', got '{action_type}'"

        self.action_type = action_type
        self.action_dim  = action_dim

        self.actor  = MLP(state_dim, action_dim, hidden_dims, activation)
        self.critic = MLP(state_dim, 1,          hidden_dims, activation)

        if self.action_type == "continuous":
            self.std = nn.Parameter(init_noise_std * torch.ones(action_dim))

        self.distribution: Normal | Categorical | None = None

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def action_mean(self) -> torch.Tensor:
        if self.action_type == "continuous":
            return self.distribution.mean
        return self.distribution.probs

    @property
    def action_std(self) -> torch.Tensor:
        if self.action_type == "continuous":
            return self.distribution.stddev
        return torch.ones_like(self.distribution.probs)

    @property
    def entropy(self) -> torch.Tensor:
        if self.action_type == "continuous":
            return self.distribution.entropy().sum(dim=-1)
        return self.distribution.entropy()

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError("Use act() or evaluate().")

    def _update_distribution(self, obs: torch.Tensor) -> None:
        """
        Build the action distribution from current observations.

        Continuous: ``Normal(mean, std)``
        Discrete  : ``Categorical(logits)``
        """
        # ========= put your code here ========= #
        actor_output = self.actor(obs)
        if self.action_type == "continuous":
            std = torch.clamp(self.std, min=1e-3).expand_as(actor_output)
            self.distribution = Normal(actor_output, std)
        else:
            self.distribution = Categorical(logits=actor_output)
        # ====================================== #

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Sample an action.

        Continuous: shape (batch, action_dim).
        Discrete  : shape (batch, 1).
        """
        # ========= put your code here ========= #
        self._update_distribution(obs)
        if self.action_type == "continuous":
            return self.distribution.sample()
        return self.distribution.sample().view(-1, 1)
        # ====================================== #

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic action: actor mean (continuous) or argmax (discrete)."""
        # ========= put your code here ========= #
        if self.action_type == "continuous":
            return self.actor(obs)
        return self.actor(obs).argmax(dim=-1, keepdim=True)
        # ====================================== #

    def evaluate(self, obs: torch.Tensor) -> torch.Tensor:
        """Critic value estimate V(s), shape (batch, 1)."""
        # ========= put your code here ========= #
        return self.critic(obs)
        # ====================================== #

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Log-probability of given actions under the current distribution.

        Continuous: sum over action dims → shape (batch,).
        Discrete  : scalar log-prob → shape (batch,).
        """
        # ========= put your code here ========= #
        if self.action_type == "continuous":
            return self.distribution.log_prob(actions).sum(dim=-1)
        return self.distribution.log_prob(actions.squeeze(-1))
        # ====================================== #


# ============================================================ #
# ====================== AC Agent ============================ #
# ============================================================ #

class AC(OnPolicyAlgorithm):
    """
    Advantage Actor-Critic (A2C) — on-policy, episodic.

    Args:
        device: Torch device.
        num_of_action (int): Action dim (continuous) or # choices (discrete).
        action_range (list): [min, max] for continuous action scaling.
        n_observations (int): Observation space dimension.
        hidden_dims (list[int]): MLP hidden layer sizes.
        activation (str): Activation function.
        action_type (str): ``'continuous'`` or ``'discrete'``.
        init_noise_std (float): Initial std for continuous policy.
        learning_rate (float): Adam learning rate.
        discount_factor (float): Discount factor γ.
        value_loss_coef (float): Coefficient for value loss.
        entropy_coef (float): Coefficient for entropy bonus.
        max_grad_norm (float): Gradient clipping norm.
    """

    def __init__(
        self,
        device=None,
        num_of_action: int = None,
        action_range: list = [None, None],
        n_observations: int = None,
        hidden_dims: list[int] = [None],
        activation: str = None,
        action_type: str = None,
        init_noise_std: float = None,
        learning_rate: float = None,
        discount_factor: float = None,
        value_loss_coef: float = None,
        entropy_coef: float = None,
        max_grad_norm: float = None,
    ) -> None:

        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Feel free to add or modify any of the initialized variables above.
        # ========= put your code here ========= #
        self.policy = ActorCritic(
            state_dim=n_observations,
            action_dim=num_of_action,
            hidden_dims=hidden_dims,
            activation=activation,
            action_type=action_type,
            init_noise_std=init_noise_std,
        ).to(self.device)
        # ====================================== #

        self.optimizer       = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.action_type     = action_type
        self.value_loss_coef = value_loss_coef
        self.entropy_coef    = entropy_coef
        self.max_grad_norm   = max_grad_norm

        super(AC, self).__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
        )

    # ------------------------------------------------------------------ #
    # Trajectory Collection                                                #
    # ------------------------------------------------------------------ #

    def generate_trajectory(self, env) -> tuple:
        """
        Run one full episode and collect the trajectory as lists.

        Args:
            env: The environment.

        Returns:
            Tuple: (episode_return, log_prob_actions, values, rewards, timestep)
        """
        # ========= put your code here ========= #
        obs, _ = env.reset()
        if isinstance(obs, dict):
            obs = obs.get("policy", next(iter(obs.values())))
        if isinstance(obs, torch.Tensor):
            obs_np = obs.detach().cpu().numpy().reshape(-1)
        else:
            obs_np = torch.as_tensor(obs, dtype=torch.float32).view(-1).cpu().numpy()

        log_prob_actions = []
        values = []
        rewards = []
        episode_return = 0.0
        timestep = 0

        while True:
            timestep += 1
            obs_tensor = torch.tensor(obs_np, dtype=torch.float32, device=self.device).view(1, -1)
            action = self.policy.act(obs_tensor)
            value = self.policy.evaluate(obs_tensor).squeeze(0)
            log_prob = self.policy.get_actions_log_prob(action).squeeze(0)

            if self.action_type == "continuous":
                env_action = torch.clamp(action, self.action_range[0], self.action_range[1])
            else:
                action_idx = int(action.item())
                action_min, action_max = self.action_range
                scaled_value = action_min + (action_idx / (self.num_of_action - 1)) * (action_max - action_min)
                env_action = torch.tensor([[scaled_value]], dtype=torch.float32, device=self.device)

            next_obs, reward, terminated, truncated, _ = env.step(env_action)
            if isinstance(next_obs, dict):
                next_obs = next_obs.get("policy", next(iter(next_obs.values())))

            reward_value = float(reward.detach().cpu().item()) if isinstance(reward, torch.Tensor) else float(reward)
            done = (
                bool(terminated.detach().cpu().item()) if isinstance(terminated, torch.Tensor) else bool(terminated)
            ) or (
                bool(truncated.detach().cpu().item()) if isinstance(truncated, torch.Tensor) else bool(truncated)
            )

            log_prob_actions.append(log_prob)
            values.append(value.squeeze(-1))
            rewards.append(reward_value)
            episode_return += reward_value

            if isinstance(next_obs, torch.Tensor):
                obs_np = next_obs.detach().cpu().numpy().reshape(-1)
            else:
                obs_np = torch.as_tensor(next_obs, dtype=torch.float32).view(-1).cpu().numpy()

            if done:
                break

        return (
            episode_return,
            torch.stack(log_prob_actions),
            torch.stack(values),
            torch.tensor(rewards, dtype=torch.float32, device=self.device),
            timestep,
        )
        # ====================================== #

    # ------------------------------------------------------------------ #
    # Return & Loss                                                        #
    # ------------------------------------------------------------------ #

    def compute_returns(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Compute discounted Monte-Carlo returns G_t = r_t + γ·r_{t+1} + ...

        Args:
            rewards (Tensor): shape (T,).

        Returns:
            Tensor: Normalised return tensor of shape (T,).
        """
        # ========= put your code here ========= #
        returns = []
        running_return = 0.0
        for reward in reversed(rewards.tolist()):
            running_return = reward + self.discount_factor * running_return
            returns.insert(0, running_return)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        if returns.numel() > 1:
            returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)
        return returns
        # ====================================== #

    def calculate_loss(self, log_prob_actions, values, returns):
        """
        Compute actor and critic losses.

        Args:
            log_prob_actions (Tensor): shape (T,).
            values (Tensor): shape (T,).
            returns (Tensor): shape (T,).

        Returns:
            Tuple[Tensor, Tensor]: (actor_loss, critic_loss)
        """
        # ========= put your code here ========= #
        advantages = returns - values.detach()
        actor_loss = -(log_prob_actions * advantages).mean()
        critic_loss = torch.nn.functional.mse_loss(values, returns)
        return actor_loss, critic_loss
        # ====================================== #

    def update_policy(self, log_prob_actions, values, returns) -> float:
        """
        Backpropagate and update.

        Returns:
            float: Total combined loss.
        """
        # ========= put your code here ========= #
        actor_loss, critic_loss = self.calculate_loss(log_prob_actions, values, returns)
        total_loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * 0.0
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.last_actor_loss = float(actor_loss.item())
        self.last_critic_loss = float(critic_loss.item())
        return float(total_loss.item())
        # ====================================== #

    # ------------------------------------------------------------------ #
    # Main Training Loop                                                   #
    # ------------------------------------------------------------------ #

    def learn(self, env, max_steps: int = 1000, num_agents: int = 1) -> tuple:
        """
        Train the agent for one episode.

        Args:
            env: The environment.
            max_steps (int): Maximum steps per episode.
            num_agents (int): Number of parallel agents.

        Returns:
            Tuple: (episode_return, loss, timestep)
        """
        self.policy.train()

        # ========= put your code here ========= #
        del max_steps, num_agents
        episode_return, log_prob_actions, values, rewards, timestep = self.generate_trajectory(env)
        returns = self.compute_returns(rewards)
        loss = self.update_policy(log_prob_actions, values, returns)
        return episode_return, loss, timestep
        # ====================================== #

    # ------------------------------------------------------------------ #
    # Inference & Persistence                                              #
    # ------------------------------------------------------------------ #

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Required by OnPolicyAlgorithm interface.
        For episodic AC, delegates to self.policy.act().
        """
        return self.policy.act(obs)

    def process_env_step(self, rewards, dones) -> None:
        """Not used by episodic AC — no RolloutBuffer to write to."""
        pass

    def select_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic action for evaluation."""
        # ========= put your code here ========= #
        if isinstance(obs, dict):
            obs = obs.get("policy", next(iter(obs.values())))
        if isinstance(obs, torch.Tensor):
            obs_tensor = obs.detach().to(self.device, dtype=torch.float32).view(1, -1)
        else:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).view(1, -1)
        action = self.policy.act_inference(obs_tensor)
        if self.action_type == "continuous":
            return torch.clamp(action, self.action_range[0], self.action_range[1])
        action_idx = int(action.item())
        scaled_value = self.action_range[0] + (action_idx / (self.num_of_action - 1)) * (
            self.action_range[1] - self.action_range[0]
        )
        return torch.tensor([[scaled_value]], dtype=torch.float32, device=self.device)
        # ====================================== #

    def save_model(self, path: str, filename: str) -> None:
        """
        Save actor-critic weights.

        Args:
            path (str): Directory to save.
            filename (str): File name (e.g., 'ac_cartpole.pth').
        """
        # ========= put your code here ========= #
        import os
        os.makedirs(path, exist_ok=True)
        torch.save(self.policy.state_dict(), os.path.join(path, filename))
        # ====================================== #

    def load_model(self, path: str, filename: str) -> None:
        """
        Load actor-critic weights.

        Args:
            path (str): Directory of saved model.
            filename (str): File name (e.g., 'ac_cartpole.pth').
        """
        # ========= put your code here ========= #
        import os
        self.policy.load_state_dict(torch.load(os.path.join(path, filename), map_location=self.device))
        self.policy.to(self.device)
        self.policy.eval()
        # ====================================== #
