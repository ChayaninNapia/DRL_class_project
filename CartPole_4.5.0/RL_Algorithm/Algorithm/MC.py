from __future__ import annotations
import numpy as np
from RL_Algorithm.RL_base import BaseAlgorithm, ControlType

class MC(BaseAlgorithm):
    def __init__(
            self,
            num_of_action: int,
            action_range: list,
            discretize_state_weight: list,
            learning_rate: float,
            initial_epsilon: float,
            epsilon_decay: float,
            final_epsilon: float,
            discount_factor: float,
    ) -> None:
        """
        Initialize the Monte Carlo algorithm.

        Args:
            num_of_action (int): Number of possible actions.
            action_range (list): Scaling factor for actions.
            discretize_state_weight (list): Scaling factor for discretizing states.
            learning_rate (float): Learning rate for Q-value updates.
            initial_epsilon (float): Initial value for epsilon in epsilon-greedy policy.
            epsilon_decay (float): Rate at which epsilon decays.
            final_epsilon (float): Minimum value for epsilon.
            discount_factor (float): Discount factor for future rewards.
        """
        super().__init__(
            control_type=ControlType.MONTE_CARLO,
            num_of_action=num_of_action,
            action_range=action_range,
            discretize_state_weight=discretize_state_weight,
            learning_rate=learning_rate,
            initial_epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            discount_factor=discount_factor,
        )
        
    def update(
        self,
        obs,
        action_idx,
        reward_value,
        done,
    ):
        """
        Update Q-values using Monte Carlo.

        This method applies the Monte Carlo update rule to improve policy decisions by updating the Q-table.
        """
        state = self.discretize_state(obs)

        self.obs_hist.append(state)
        self.action_hist.append(action_idx)
        self.reward_hist.append(reward_value)

        if not done:
            return
        
        G = 0.0
        returns = [0.0] * len(self.reward_hist)

        # Compute return G_t for each step
        for t in reversed(range(len(self.reward_hist))):
            G = self.reward_hist[t] + self.discount_factor * G
            returns[t] = G

        visited_state_actions = set()

        # First-visit Monte Carlo update
        for t in range(len(self.obs_hist)):
            state = self.obs_hist[t]
            action = self.action_hist[t]
            state_action = (state, action)

            if state_action in visited_state_actions:
                continue

            visited_state_actions.add(state_action)

            self.n_values[state][action] += 1
            self.q_values[state][action] += (
                returns[t] - self.q_values[state][action]
            ) / self.n_values[state][action]

        # Clear episode history
        self.obs_hist.clear()
        self.action_hist.clear()
        self.reward_hist.clear()