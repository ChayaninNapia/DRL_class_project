from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim
from RL_Algorithm.storage.on_policy import OnPolicyAlgorithm
from RL_Algorithm.storage.buffers import RolloutBuffer
from RL_Algorithm.Function_based.AC import ActorCritic


class PPO(OnPolicyAlgorithm):
    """
    Proximal Policy Optimization (PPO) — on-policy, clipped surrogate.

    Args:
        device: Torch device.
        num_of_action (int): Action dim (continuous) or number of choices (discrete).
        action_range (list): [min, max] for continuous action scaling.
        n_observations (int): Observation space dimension.
        hidden_dims (list[int]): MLP hidden layer sizes.
        activation (str): Activation function.
        action_type (str): ``'continuous'`` or ``'discrete'``.
        init_noise_std (float): Initial std for continuous policy.
        num_learning_epochs (int): Epochs per PPO update.
        num_mini_batches (int): Mini-batches per epoch.
        clip_param (float): PPO clipping ε.
        gamma (float): Discount factor γ.
        lam (float): GAE lambda λ.
        value_loss_coef (float): Coefficient for value loss.
        entropy_coef (float): Coefficient for entropy bonus.
        learning_rate (float): Adam learning rate.
        max_grad_norm (float): Gradient clipping norm.
        desired_kl (float): KL target for adaptive LR (0 to disable; use 0 for discrete).
        normalize_advantage_per_mini_batch (bool): Normalise advantages per mini-batch.
        use_clipped_value_loss (bool): Apply clipped value loss.
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
        num_learning_epochs: int = None,
        num_mini_batches: int = None,
        clip_param: float = None,
        gamma: float = None,
        lam: float = None,
        value_loss_coef: float = None,
        entropy_coef: float = None,
        learning_rate: float = None,
        max_grad_norm: float = None,
        desired_kl: float = None,
        normalize_advantage_per_mini_batch: bool = False,
        use_clipped_value_loss: bool = True,
    ) -> None:

        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # ===== Build ActorCritic network (imported from AC.py) ===== #
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

        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        # ===== PPO hyperparameters ===== #
        self.action_type                        = action_type
        self.clip_param                         = clip_param
        self.num_learning_epochs                = num_learning_epochs
        self.num_mini_batches                   = num_mini_batches
        self.value_loss_coef                    = value_loss_coef
        self.entropy_coef                       = entropy_coef
        self.gamma                              = gamma
        self.lam                                = lam
        self.max_grad_norm                      = max_grad_norm
        self.desired_kl                         = desired_kl
        self.learning_rate                      = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch
        self.use_clipped_value_loss             = use_clipped_value_loss
        self.last_actor_loss                    = 0.0
        self.last_critic_loss                   = 0.0
        self.last_entropy                       = 0.0
        self.last_total_loss                    = 0.0

        super(PPO, self).__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
        )

    # ------------------------------------------------------------------ #
    # Rollout collection                                                   #
    # ------------------------------------------------------------------ #

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Sample actions for all parallel envs and populate self.transition.

        Continuous: actions shape (num_envs, action_dim).
        Discrete  : actions shape (num_envs, 1).

        Args:
            obs (Tensor): shape (num_envs, obs_dim).

        Returns:
            Tensor: Sampled actions.
        """
        # ========= put your code here ========= #
        actions = self.policy.act(obs)
        values = self.policy.evaluate(obs)
        log_prob = self.policy.get_actions_log_prob(actions).view(-1, 1)
        self.transition.observations = obs.detach()
        self.transition.actions = actions.detach().float()
        self.transition.values = values.detach()
        self.transition.actions_log_prob = log_prob.detach()
        if self.action_type == "continuous":
            self.transition.action_mean = self.policy.action_mean.detach()
            self.transition.action_sigma = self.policy.action_std.detach()
        else:
            self.transition.action_mean = actions.detach().float()
            self.transition.action_sigma = torch.ones_like(actions, dtype=torch.float32, device=self.device)
        # ====================================== #

        return self.transition.actions

    def process_env_step(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        """
        Write rewards and dones into self.transition, then flush to storage.

        Args:
            rewards (Tensor): shape (num_envs,) or (num_envs, 1).
            dones (Tensor): shape (num_envs,) or (num_envs, 1).
        """
        # ========= put your code here ========= #
        self.transition.rewards = rewards.view(-1, 1).detach().to(self.device)
        self.transition.dones = dones.view(-1, 1).detach().to(self.device)
        # ====================================== #

        # Flush transition into RolloutBuffer via inherited add_transition()
        self.add_transition()

    # ------------------------------------------------------------------ #
    # Return & Advantage Computation                                       #
    # ------------------------------------------------------------------ #

    def compute_returns(self, last_obs: torch.Tensor) -> None:
        """
        Compute GAE returns and advantages over the collected rollout.

        Args:
            last_obs (Tensor): Observation after the final rollout step.
                               Shape: (num_envs, obs_dim).
        """
        # ========= put your code here ========= #
        with torch.no_grad():
            last_values = self.policy.evaluate(last_obs)

        advantage = torch.zeros_like(last_values)
        for step in reversed(range(self.storage.num_transitions_per_env)):
            if step == self.storage.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.storage.values[step + 1]
            done = self.storage.dones[step].float()
            delta = self.storage.rewards[step] + self.gamma * next_values * (1.0 - done) - self.storage.values[step]
            advantage = delta + self.gamma * self.lam * (1.0 - done) * advantage
            self.storage.advantages[step] = advantage
            self.storage.returns[step] = advantage + self.storage.values[step]

        advantages = self.storage.advantages[: self.storage.step]
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        self.storage.advantages[: self.storage.step] = advantages
        # ====================================== #

    # ------------------------------------------------------------------ #
    # Policy Update                                                        #
    # ------------------------------------------------------------------ #

    def update(self) -> dict:
        """
        Perform PPO updates over the collected rollout.

        Calls ``self.storage.mini_batch_generator()`` which now lives in
        ``RolloutBuffer`` (storage/buffers.py) and yields 8-tuples.

        Returns:
            dict: Mean losses {'value', 'surrogate', 'entropy'}.
        """
        mean_value_loss     = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy        = 0.0
        mean_total_loss     = 0.0

        generator = self.storage.mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs
        )

        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
        ) in generator:
            # ========= put your code here ========= #
            del old_mu_batch, old_sigma_batch
            self.policy._update_distribution(obs_batch)
            new_log_prob = self.policy.get_actions_log_prob(actions_batch).view(-1, 1)
            values = self.policy.evaluate(obs_batch)
            entropy = self.policy.entropy.mean()

            adv = advantages_batch
            if self.normalize_advantage_per_mini_batch:
                adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

            ratio = torch.exp(new_log_prob - old_actions_log_prob_batch)
            surrogate_1 = ratio * adv
            surrogate_2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv
            surrogate_loss = -torch.min(surrogate_1, surrogate_2).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (values - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (values - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = 0.5 * (returns_batch - values).pow(2).mean()

            total_loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += float(value_loss.item())
            mean_surrogate_loss += float(surrogate_loss.item())
            mean_entropy += float(entropy.item())
            mean_total_loss += float(total_loss.item())
            # ====================================== #

        num_updates          = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss     /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy        /= num_updates
        mean_total_loss     /= num_updates

        self.storage.clear()   # on-policy: discard rollout after update
        self.last_critic_loss = mean_value_loss
        self.last_actor_loss = mean_surrogate_loss
        self.last_entropy = mean_entropy
        self.last_total_loss = mean_total_loss

        return {
            "value":     mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy":   mean_entropy,
            "total":     mean_total_loss,
        }

    # ------------------------------------------------------------------ #
    # Main Training Loop                                                   #
    # ------------------------------------------------------------------ #

    def learn(
        self,
        env,
        num_envs: int,
        num_transitions_per_env: int,
        max_episodes: int = 10000,
    ) -> None:
        """
        Main PPO parallel training loop.

        Calls ``_init_storage()`` (from OnPolicyAlgorithm) to create the buffer.

        Continuous: actions_shape = (num_of_action,)
        Discrete  : actions_shape = (1,)

        Args:
            env: Isaac Lab vectorised environment.
            num_envs (int): Number of parallel environments.
            num_transitions_per_env (int): Rollout horizon per env.
            max_episodes (int): Total number of training rollouts.
        """
        # ========= put your code here ========= #
        del max_episodes
        if num_envs != 1:
            raise ValueError("PPO implementation currently supports only a single environment.")

        if self.storage is None or self.storage.num_transitions_per_env != num_transitions_per_env:
            actions_shape = (self.num_of_action,) if self.action_type == "continuous" else (1,)
            self._init_storage(
                num_envs=1,
                num_transitions_per_env=num_transitions_per_env,
                obs_shape=(self.policy.actor[0].in_features,),
                actions_shape=actions_shape,
                device=self.device,
            )

        obs, _ = env.reset()
        rollout_reward = 0.0
        timestep = 0

        while self.storage.step < num_transitions_per_env:
            if isinstance(obs, dict):
                obs = obs.get("policy", next(iter(obs.values())))
            obs_tensor = (
                obs.detach().to(self.device, dtype=torch.float32).view(1, -1)
                if isinstance(obs, torch.Tensor)
                else torch.tensor(obs, dtype=torch.float32, device=self.device).view(1, -1)
            )
            actions = self.act(obs_tensor)
            if self.action_type == "continuous":
                env_action = torch.clamp(actions, self.action_range[0], self.action_range[1])
            else:
                action_idx = int(actions.item())
                scaled = self.action_range[0] + (action_idx / (self.num_of_action - 1)) * (
                    self.action_range[1] - self.action_range[0]
                )
                env_action = torch.tensor([[scaled]], dtype=torch.float32, device=self.device)

            next_obs, reward, terminated, truncated, _ = env.step(env_action)
            reward_value = float(reward.detach().cpu().item()) if isinstance(reward, torch.Tensor) else float(reward)
            done = (
                bool(terminated.detach().cpu().item()) if isinstance(terminated, torch.Tensor) else bool(terminated)
            ) or (
                bool(truncated.detach().cpu().item()) if isinstance(truncated, torch.Tensor) else bool(truncated)
            )
            self.process_env_step(
                torch.tensor([reward_value], dtype=torch.float32, device=self.device),
                torch.tensor([done], dtype=torch.uint8, device=self.device),
            )
            rollout_reward += reward_value
            timestep += 1
            if done:
                obs, _ = env.reset()
            else:
                obs = next_obs

        if isinstance(obs, dict):
            obs = obs.get("policy", next(iter(obs.values())))
        last_obs = (
            obs.detach().to(self.device, dtype=torch.float32).view(1, -1)
            if isinstance(obs, torch.Tensor)
            else torch.tensor(obs, dtype=torch.float32, device=self.device).view(1, -1)
        )
        self.compute_returns(last_obs)
        stats = self.update()
        return rollout_reward, timestep, stats
        # ====================================== #


    # ------------------------------------------------------------------ #
    # Inference & Persistence                                              #
    # ------------------------------------------------------------------ #

    def select_action(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Deterministic action for evaluation.

        Continuous: actor mean. Discrete: argmax of logits.

        Args:
            obs (Tensor): shape (1, obs_dim) or (obs_dim,).
        """
        # ========= put your code here ========= #
        if isinstance(obs, dict):
            obs = obs.get("policy", next(iter(obs.values())))
        obs_tensor = (
            obs.detach().to(self.device, dtype=torch.float32).view(1, -1)
            if isinstance(obs, torch.Tensor)
            else torch.tensor(obs, dtype=torch.float32, device=self.device).view(1, -1)
        )
        action = self.policy.act_inference(obs_tensor)
        if self.action_type == "continuous":
            return torch.clamp(action, self.action_range[0], self.action_range[1])
        action_idx = int(action.item())
        scaled = self.action_range[0] + (action_idx / (self.num_of_action - 1)) * (
            self.action_range[1] - self.action_range[0]
        )
        return torch.tensor([[scaled]], dtype=torch.float32, device=self.device)
        # ====================================== #

    def save_model(self, path: str, filename: str) -> None:
        """
        Save actor-critic weights.

        Args:
            path (str): Directory to save.
            filename (str): File name (e.g., 'ppo_cartpole.pth').
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
            filename (str): File name (e.g., 'ppo_cartpole.pth').
        """
        # ========= put your code here ========= #
        import os
        self.policy.load_state_dict(torch.load(os.path.join(path, filename), map_location=self.device))
        self.policy.to(self.device)
        self.policy.eval()
        # ====================================== #
