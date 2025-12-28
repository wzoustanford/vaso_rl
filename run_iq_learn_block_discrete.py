#!/usr/bin/env python3
"""
IQ-Learn Block Discrete Training Script
Inverse Q-Learning for reward recovery from clinician demonstrations.

Based on: "IQ-Learn: Inverse soft-Q Learning for Imitation" (Garg et al., 2021)
https://arxiv.org/abs/2106.12142

Key idea: Learn Q-function such that expert trajectories have high implicit rewards
where r(s,a) = Q(s,a) - gamma*V(s')

Action space:
- VP1: Binary (0 or 1)
- VP2: Discretized into N bins (0 to 0.5 mcg/kg/min)

Hyperparameters for discrete action space:
- init_temp = 0.001 (soft-value temperature)
- method_alpha = 0.5 (chi2 regularization)
- div = "chi" (chi2 divergence)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import os
import sys
from typing import Dict, Tuple, Optional

# Import our unified pipeline
from integrated_data_pipeline_v2 import IntegratedDataPipelineV2

# Force unbuffered output
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__


# ============================================================================
# Q-Network for Block Discrete Actions
# ============================================================================

class BlockDiscreteQNetwork(nn.Module):
    """
    Q-network for block discrete actions using Q(s,a) -> R architecture.
    Takes state and discrete action index as input, outputs single Q-value.

    VP1: 2 actions (binary)
    VP2: vp2_bins actions (discretized continuous)
    Total: 2 * vp2_bins possible actions
    """

    def __init__(self, state_dim: int, vp2_bins: int = 5):
        super().__init__()
        self.state_dim = state_dim
        self.vp2_bins = vp2_bins
        self.total_actions = 2 * vp2_bins

        # Network takes state + one-hot encoded action
        input_dim = state_dim + self.total_actions

        # Architecture: input -> 128 -> 64 -> 32 -> 1 (aligned with MaxEnt IRL)
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

        # Xavier initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)

    def forward(self, state: torch.Tensor, action_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: [batch_size, state_dim]
            action_idx: [batch_size] discrete action indices
        Returns:
            q_value: [batch_size, 1]
        """
        action_one_hot = F.one_hot(action_idx.long(), num_classes=self.total_actions).float()
        x = torch.cat([state, action_one_hot], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

    def get_all_q_values(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get Q-values for all actions given states.

        Args:
            state: [batch_size, state_dim]
        Returns:
            q_values: [batch_size, total_actions]
        """
        batch_size = state.shape[0]

        all_actions = torch.arange(self.total_actions, device=state.device)
        all_actions = all_actions.unsqueeze(0).expand(batch_size, -1)

        state_expanded = state.unsqueeze(1).expand(-1, self.total_actions, -1)
        state_expanded = state_expanded.reshape(-1, self.state_dim)
        actions_flat = all_actions.reshape(-1)

        q_values = self.forward(state_expanded, actions_flat)
        q_values = q_values.reshape(batch_size, self.total_actions)

        return q_values


# ============================================================================
# IQ-Learn Agent for Block Discrete Actions
# ============================================================================

class IQLearnBlockDiscrete:
    """
    IQ-Learn agent for inverse reinforcement learning with block discrete actions.

    Learns Q-function from expert demonstrations without explicit rewards.
    The implicit reward is: r(s,a) = Q(s,a) - gamma*V(s')
    """

    def __init__(
        self,
        state_dim: int,
        vp2_bins: int = 5,
        gamma: float = 0.99,
        init_temp: float = 0.001,
        method_alpha: float = 0.5,
        tau: float = 0.005,
        lr: float = 1e-4,
        grad_clip: float = 1.0,
        use_target: bool = True,
        div: str = "chi",
        loss_type: str = "value_expert",
        regularize: bool = False,
        # grad_pen: bool = False,  # Commented out - for Wasserstein distance
        # lambda_gp: float = 10.0,  # Commented out - for Wasserstein distance
    ):
        """
        Args:
            state_dim: Dimension of state space
            vp2_bins: Number of bins for VP2 discretization
            gamma: Discount factor
            init_temp: Soft-value temperature (small for discrete actions)
            method_alpha: chi2 regularization coefficient
            tau: Target network soft update rate
            lr: Learning rate
            grad_clip: Gradient clipping value
            use_target: Whether to use target network for next_V
            div: Divergence type ("chi", "kl", "kl2", "kl_fix", "js", "hellinger")
            loss_type: Value loss type ("value_expert", "value", "v0")
            regularize: Whether to use chi2 regularization on all states (works online)
        """
        self.state_dim = state_dim
        self.vp2_bins = vp2_bins
        self.total_actions = 2 * vp2_bins
        self.gamma = gamma
        self.init_temp = init_temp
        self.method_alpha = method_alpha
        self.tau = tau
        self.grad_clip = grad_clip
        self.use_target = use_target
        self.div = div
        self.loss_type = loss_type
        self.regularize = regularize
        # self.grad_pen = grad_pen  # Commented out - for Wasserstein distance
        # self.lambda_gp = lambda_gp  # Commented out - for Wasserstein distance

        # VP2 bin edges (0 to 0.5)
        self.vp2_bin_edges = np.linspace(0, 0.5, vp2_bins + 1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize Q-networks (double Q for stability)
        self.q1 = BlockDiscreteQNetwork(state_dim, vp2_bins).to(self.device)
        self.q2 = BlockDiscreteQNetwork(state_dim, vp2_bins).to(self.device)

        # Initialize target networks
        self.q1_target = BlockDiscreteQNetwork(state_dim, vp2_bins).to(self.device)
        self.q2_target = BlockDiscreteQNetwork(state_dim, vp2_bins).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Single optimizer for both Q-networks
        self.optimizer = optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=lr
        )

    def continuous_to_discrete_action(self, continuous_action: np.ndarray) -> int:
        """Convert continuous action [vp1, vp2] to discrete action index."""
        vp1, vp2 = continuous_action
        vp1_idx = int(vp1)
        vp2_bin = np.digitize(vp2, self.vp2_bin_edges) - 1
        vp2_bin = np.clip(vp2_bin, 0, self.vp2_bins - 1)
        action_idx = vp1_idx * self.vp2_bins + vp2_bin
        return action_idx

    def continuous_to_discrete_batch(self, actions: torch.Tensor) -> torch.Tensor:
        """Convert batch of continuous actions to discrete indices."""
        batch_size = actions.shape[0]
        action_indices = []
        for i in range(batch_size):
            action_np = actions[i].cpu().numpy()
            action_idx = self.continuous_to_discrete_action(action_np)
            action_indices.append(action_idx)
        return torch.LongTensor(action_indices).to(self.device)

    def discrete_to_continuous_action(self, action_idx: int) -> np.ndarray:
        """Convert discrete action index to continuous action [vp1, vp2]."""
        vp1_idx = action_idx // self.vp2_bins
        vp2_bin = action_idx % self.vp2_bins
        vp2_value = (self.vp2_bin_edges[vp2_bin] + self.vp2_bin_edges[vp2_bin + 1]) / 2
        return np.array([float(vp1_idx), vp2_value])

    def getV(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Compute soft-value function from Q-values.
        V = alpha * logsumexp(Q / alpha) where alpha = init_temp

        Args:
            obs: [batch_size, state_dim]
        Returns:
            v: [batch_size, 1]
        """
        q1_all = self.q1.get_all_q_values(obs)
        q2_all = self.q2.get_all_q_values(obs)
        q_all = torch.min(q1_all, q2_all)
        v = self.init_temp * torch.logsumexp(q_all / self.init_temp, dim=1, keepdim=True)
        return v

    def get_targetV(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Compute soft-value function using target networks.

        Args:
            obs: [batch_size, state_dim]
        Returns:
            v: [batch_size, 1]
        """
        q1_all = self.q1_target.get_all_q_values(obs)
        q2_all = self.q2_target.get_all_q_values(obs)
        q_all = torch.min(q1_all, q2_all)
        v = self.init_temp * torch.logsumexp(q_all / self.init_temp, dim=1, keepdim=True)
        return v

    def select_action(self, state, return_discrete: bool = False):
        """
        Select best action using argmax over Q-values.

        Args:
            state: np.ndarray [batch, state_dim] or [state_dim],
                   or torch.Tensor [batch, state_dim]
            return_discrete: If True, return discrete tensor indices

        Returns:
            If return_discrete: torch.Tensor [batch] discrete action indices
            Else: np.ndarray [batch, 2] or [2] continuous actions
        """
        with torch.no_grad():
            # Handle input type
            if isinstance(state, np.ndarray):
                if state.ndim == 1:
                    state = state.reshape(1, -1)
                state_tensor = torch.FloatTensor(state).to(self.device)
                single_input = (state.shape[0] == 1)
            else:
                state_tensor = state
                single_input = False

            batch_size = state_tensor.shape[0]

            # Get Q-values for all actions
            q1_all = self.q1.get_all_q_values(state_tensor)
            q2_all = self.q2.get_all_q_values(state_tensor)
            q_all = torch.min(q1_all, q2_all)

            # Get best action (argmax)
            best_action_indices = q_all.argmax(dim=1)

            if return_discrete:
                return best_action_indices

            # Convert to continuous actions
            indices_np = best_action_indices.cpu().numpy()
            vp1_actions = (indices_np // self.vp2_bins).astype(float)
            vp2_bin_indices = indices_np % self.vp2_bins
            vp2_bin_centers = (self.vp2_bin_edges[:-1] + self.vp2_bin_edges[1:]) / 2
            vp2_actions = vp2_bin_centers[vp2_bin_indices]

            actions = np.stack([vp1_actions, vp2_actions], axis=1)
            return actions[0] if single_input else actions

    def iq_loss(
        self,
        current_Q: torch.Tensor,
        current_v: torch.Tensor,
        next_v: torch.Tensor,
        done: torch.Tensor,
        is_expert: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Full IQ-Learn loss function.

        Operates on concatenated (policy_batch, expert_batch).
        Uses is_expert mask to distinguish expert vs policy samples.

        Args:
            current_Q: [batch, 1] Q(s, a) values
            current_v: [batch, 1] V(s) values
            next_v: [batch, 1] V(s') values
            done: [batch] terminal flags
            is_expert: [batch] boolean mask (True = expert sample)

        Returns:
            loss: scalar loss tensor
            loss_dict: dictionary of loss components
        """
        loss_dict = {}

        # y = (1 - done) * gamma * V(s')
        y = (1 - done.unsqueeze(1)) * self.gamma * next_v

        # Implicit reward: r = Q(s,a) - gamma*V(s')
        reward = current_Q - y

        # Track value of initial/expert states
        v0 = current_v[is_expert].mean()
        loss_dict['v0'] = v0.item()

        # ==============================================================
        # Term 1: -E_expert[Q(s,a) - gamma*V(s')] with divergence weight
        # ==============================================================

        expert_reward = reward[is_expert]

        with torch.no_grad():
            # Different divergence functions
            # (For chi2 divergence, we add a third term instead)
            if self.div == "hellinger":
                phi_grad = 1 / (1 + expert_reward.clamp(min=-0.99)) ** 2
            elif self.div == "kl":
                # original dual form for kl divergence (with clamping for stability)
                phi_grad = torch.exp(-expert_reward.clamp(-20, 20) - 1)
            elif self.div == "kl2":
                # biased dual form for kl divergence
                phi_grad = F.softmax(-expert_reward, dim=0) * expert_reward.shape[0]
            elif self.div == "kl_fix":
                # unbiased form for fixing kl divergence (with clamping for stability)
                phi_grad = torch.exp(-expert_reward.clamp(-20, 20))
            elif self.div == "js":
                # jensen-shannon (with clamping to prevent div by zero)
                exp_neg_r = torch.exp(-expert_reward.clamp(-20, 20))
                phi_grad = exp_neg_r / (2 - exp_neg_r.clamp(max=1.99))
            else:
                phi_grad = 1

        softq_loss = -(phi_grad * expert_reward).mean()
        loss_dict['softq_loss'] = softq_loss.item()
        loss = softq_loss

        # ==============================================================
        # Term 2: Value regularization (different strategies)
        # ==============================================================

        if self.loss_type == "value_expert":
            # Sample using only expert states (works offline)
            value_loss = (current_v - y)[is_expert].mean()
            loss += value_loss
            loss_dict['value_loss'] = value_loss.item()

        elif self.loss_type == "value":
            # Sample using expert and policy states (works online)
            value_loss = (current_v - y).mean()
            loss += value_loss
            loss_dict['value_loss'] = value_loss.item()

        elif self.loss_type == "v0":
            # Use only initial states (works offline, usually suboptimal)
            # (1 - gamma) * E[V(s0)]
            v0_loss = (1 - self.gamma) * v0
            loss += v0_loss
            loss_dict['v0_loss'] = v0_loss.item()

        # ==============================================================
        # Optional: Gradient penalty for Wasserstein distance
        # (Commented out - requires _compute_grad_penalty implementation)
        # ==============================================================

        # if self.grad_pen:
        #     # Gradient penalty is needed for Wasserstein-1 metric.
        #     # This computes the gradient penalty between expert and policy
        #     # distributions to enforce the Lipschitz constraint.
        #     # Implementation: _compute_grad_penalty(expert_obs, expert_action,
        #     #                                       policy_obs, policy_action,
        #     #                                       lambda_gp)
        #     # gp_loss = self._compute_grad_penalty(...)
        #     # loss += gp_loss
        #     # loss_dict['gp_loss'] = gp_loss.item()
        #     pass

        # ==============================================================
        # chi2 divergence regularization
        # ==============================================================

        if self.div == "chi":
            # chi2 regularization using expert states (works offline)
            chi2_loss = (1 / (4 * self.method_alpha)) * (expert_reward ** 2).mean()
            loss += chi2_loss
            loss_dict['chi2_loss'] = chi2_loss.item()

        if self.regularize:
            # chi2 regularization using ALL states (works online)
            reg_loss = (1 / (4 * self.method_alpha)) * (reward ** 2).mean()
            loss += reg_loss
            loss_dict['regularize_loss'] = reg_loss.item()

        loss_dict['total_loss'] = loss.item()

        return loss, loss_dict

    def iq_update(
        self,
        states: torch.Tensor,
        expert_actions: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor
    ) -> Dict[str, float]:
        """
        IQ-Learn update step.

        Creates concatenated batch of (policy_samples, expert_samples):
        - Expert batch: states with clinician actions
        - Policy batch: same states with policy actions (argmax Q)

        Args:
            states: [batch, state_dim] expert states
            expert_actions: [batch, 2] expert continuous actions
            next_states: [batch, state_dim] next states
            dones: [batch] terminal flags

        Returns:
            Dictionary of loss metrics
        """
        batch_size = states.shape[0]

        # ==============================================================
        # Create policy batch and expert batch
        # ==============================================================

        # Expert actions (from clinician demonstrations)
        expert_action_indices = self.continuous_to_discrete_batch(expert_actions)

        # Policy actions (from current Q-function via argmax)
        policy_action_indices = self.select_action(states, return_discrete=True)

        # ==============================================================
        # Concatenate batches: [policy_batch, expert_batch]
        # ==============================================================

        # States: same for both policy and expert
        concat_states = torch.cat([states, states], dim=0)
        concat_next_states = torch.cat([next_states, next_states], dim=0)
        concat_dones = torch.cat([dones, dones], dim=0)

        # Actions: policy actions first, then expert actions
        concat_actions = torch.cat([policy_action_indices, expert_action_indices], dim=0)

        # is_expert mask: [False, False, ..., True, True, ...]
        is_expert = torch.cat([
            torch.zeros(batch_size, dtype=torch.bool, device=self.device),
            torch.ones(batch_size, dtype=torch.bool, device=self.device)
        ], dim=0)

        # ==============================================================
        # Compute value functions on concatenated batch
        # ==============================================================

        current_v = self.getV(concat_states)

        if self.use_target:
            with torch.no_grad():
                next_v = self.get_targetV(concat_next_states)
        else:
            next_v = self.getV(concat_next_states)

        # ==============================================================
        # Compute Q-values and IQ loss (Double Q)
        # ==============================================================

        current_Q1 = self.q1(concat_states, concat_actions)
        current_Q2 = self.q2(concat_states, concat_actions)

        # IQ loss for each Q-network
        q1_loss, loss_dict1 = self.iq_loss(
            current_Q1, current_v, next_v, concat_dones, is_expert
        )
        q2_loss, loss_dict2 = self.iq_loss(
            current_Q2, current_v, next_v, concat_dones, is_expert
        )

        # Combined loss (average of both Q losses)
        critic_loss = 0.5 * (q1_loss + q2_loss)

        # Merge loss dicts (average values)
        loss_dict = {
            k: 0.5 * (loss_dict1.get(k, 0) + loss_dict2.get(k, 0))
            for k in set(loss_dict1) | set(loss_dict2)
        }

        # ==============================================================
        # Optimization step
        # ==============================================================

        self.optimizer.zero_grad()
        critic_loss.backward()

        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.q1.parameters(), self.grad_clip)
            torch.nn.utils.clip_grad_norm_(self.q2.parameters(), self.grad_clip)

        self.optimizer.step()

        # ==============================================================
        # Soft update target networks
        # ==============================================================

        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # ==============================================================
        # Additional metrics
        # ==============================================================

        with torch.no_grad():
            loss_dict['q1_mean'] = current_Q1.mean().item()
            loss_dict['q2_mean'] = current_Q2.mean().item()
            loss_dict['v_mean'] = current_v.mean().item()
            loss_dict['reward_expert_mean'] = (current_Q1[is_expert] -
                (1 - concat_dones[is_expert].unsqueeze(1)) * self.gamma * next_v[is_expert]).mean().item()

        return loss_dict

    def get_recovered_reward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the recovered/implicit reward: r(s,a) = Q(s,a) - gamma*V(s')

        Args:
            states: [batch, state_dim]
            actions: [batch, 2] continuous actions
            next_states: [batch, state_dim]
            dones: [batch]

        Returns:
            rewards: [batch] implicit rewards
        """
        with torch.no_grad():
            action_indices = self.continuous_to_discrete_batch(actions)

            q1 = self.q1(states, action_indices)
            q2 = self.q2(states, action_indices)
            q = torch.min(q1, q2)

            next_v = self.get_targetV(next_states)
            y = (1 - dones.unsqueeze(1)) * self.gamma * next_v

            reward = (q - y).squeeze()

        return reward

    def save(self, filepath: str):
        """Save model checkpoint."""
        torch.save({
            'q1_state_dict': self.q1.state_dict(),
            'q2_state_dict': self.q2.state_dict(),
            'q1_target_state_dict': self.q1_target.state_dict(),
            'q2_target_state_dict': self.q2_target.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'state_dim': self.state_dim,
            'vp2_bins': self.vp2_bins,
            'gamma': self.gamma,
            'init_temp': self.init_temp,
            'method_alpha': self.method_alpha,
            'tau': self.tau,
            'div': self.div,
            'loss_type': self.loss_type,
            'regularize': self.regularize,
        }, filepath)
        print(f"Model saved to: {filepath}")

    @classmethod
    def load(cls, filepath: str, device: Optional[torch.device] = None) -> 'IQLearnBlockDiscrete':
        """Load model from checkpoint."""
        checkpoint = torch.load(filepath, map_location=device)

        agent = cls(
            state_dim=checkpoint['state_dim'],
            vp2_bins=checkpoint['vp2_bins'],
            gamma=checkpoint['gamma'],
            init_temp=checkpoint['init_temp'],
            method_alpha=checkpoint['method_alpha'],
            tau=checkpoint['tau'],
            div=checkpoint['div'],
            loss_type=checkpoint['loss_type'],
            regularize=checkpoint['regularize'],
        )

        agent.q1.load_state_dict(checkpoint['q1_state_dict'])
        agent.q2.load_state_dict(checkpoint['q2_state_dict'])
        agent.q1_target.load_state_dict(checkpoint['q1_target_state_dict'])
        agent.q2_target.load_state_dict(checkpoint['q2_target_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print(f"Model loaded from: {filepath}")
        return agent


# ============================================================================
# Training Function
# ============================================================================

def train_iq_learn(
    vp2_bins: int = 5,
    epochs: int = 500,
    batch_size: int = 128,
    gamma: float = 0.99,
    init_temp: float = 0.001,
    method_alpha: float = 0.5,
    tau: float = 0.005,
    lr: float = 1e-4,
    div: str = "chi",
    loss_type: str = "value_expert",
    regularize: bool = False,
    experiment_prefix: str = "iq_learn",
    save_dir: str = "experiment/iq_learn"
):
    """Train IQ-Learn agent for inverse RL reward recovery."""

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 70, flush=True)
    print(" IQ-LEARN BLOCK DISCRETE TRAINING", flush=True)
    print(" Inverse Q-Learning for Reward Recovery", flush=True)
    print("=" * 70, flush=True)

    # Initialize data pipeline
    print("\nInitializing data pipeline...", flush=True)
    pipeline = IntegratedDataPipelineV2(model_type='dual', random_seed=42)
    train_data, val_data, test_data = pipeline.prepare_data()

    state_dim = train_data['states'].shape[1]

    # Print settings
    print("\n" + "=" * 70, flush=True)
    print("IQ-LEARN SETTINGS:", flush=True)
    print(f"  State dimension: {state_dim}", flush=True)
    print(f"  Action: VP1 (binary) x VP2 ({vp2_bins} bins) = {2 * vp2_bins} actions", flush=True)
    print(f"  gamma = {gamma}", flush=True)
    print(f"  init_temp = {init_temp} (soft-value temperature)", flush=True)
    print(f"  method_alpha = {method_alpha} (chi2 regularization)", flush=True)
    print(f"  tau = {tau} (target network update)", flush=True)
    print(f"  lr = {lr}", flush=True)
    print(f"  div = {div}", flush=True)
    print(f"  loss_type = {loss_type}", flush=True)
    print(f"  regularize = {regularize}", flush=True)
    print(f"  batch_size = {batch_size}", flush=True)
    print(f"  epochs = {epochs}", flush=True)
    print("=" * 70, flush=True)

    # Initialize agent
    agent = IQLearnBlockDiscrete(
        state_dim=state_dim,
        vp2_bins=vp2_bins,
        gamma=gamma,
        init_temp=init_temp,
        method_alpha=method_alpha,
        tau=tau,
        lr=lr,
        div=div,
        loss_type=loss_type,
        regularize=regularize,
    )

    print(f"\nDevice: {agent.device}", flush=True)

    # Training loop
    print(f"\nTraining for {epochs} epochs...", flush=True)
    start_time = time.time()

    os.makedirs('experiment/iq_learn', exist_ok=True)

    best_val_reward = float('-inf')

    for epoch in range(epochs):
        # Training phase
        agent.q1.train()
        agent.q2.train()

        train_metrics = {
            'total_loss': 0, 'softq_loss': 0, 'value_loss': 0,
            'chi2_loss': 0, 'q1_mean': 0, 'q2_mean': 0,
            'v_mean': 0, 'reward_expert_mean': 0
        }

        n_batches = len(train_data['states']) // batch_size

        for _ in range(n_batches):
            batch = pipeline.get_batch(batch_size=batch_size, split='train')

            states = torch.FloatTensor(batch['states']).to(agent.device)
            actions = torch.FloatTensor(batch['actions']).to(agent.device)
            next_states = torch.FloatTensor(batch['next_states']).to(agent.device)
            dones = torch.FloatTensor(batch['dones']).to(agent.device)
            # Note: We do NOT use batch['rewards'] - IQ-Learn learns rewards implicitly

            metrics = agent.iq_update(states, actions, next_states, dones)

            for key in train_metrics:
                if key in metrics:
                    train_metrics[key] += metrics[key]

        for key in train_metrics:
            train_metrics[key] /= n_batches

        # Validation phase
        agent.q1.eval()
        agent.q2.eval()

        val_metrics = {
            'total_loss': 0, 'softq_loss': 0, 'value_loss': 0,
            'chi2_loss': 0, 'q1_mean': 0, 'q2_mean': 0,
            'v_mean': 0, 'reward_expert_mean': 0
        }
        n_val_batches = 10

        with torch.no_grad():
            for _ in range(n_val_batches):
                batch = pipeline.get_batch(batch_size=batch_size, split='val')

                states = torch.FloatTensor(batch['states']).to(agent.device)
                actions = torch.FloatTensor(batch['actions']).to(agent.device)
                next_states = torch.FloatTensor(batch['next_states']).to(agent.device)
                dones = torch.FloatTensor(batch['dones']).to(agent.device)

                # Compute validation metrics (same as training but no backward)
                expert_action_indices = agent.continuous_to_discrete_batch(actions)
                policy_action_indices = agent.select_action(states, return_discrete=True)

                concat_states = torch.cat([states, states], dim=0)
                concat_next_states = torch.cat([next_states, next_states], dim=0)
                concat_dones = torch.cat([dones, dones], dim=0)
                concat_actions = torch.cat([policy_action_indices, expert_action_indices], dim=0)

                is_expert = torch.cat([
                    torch.zeros(batch_size, dtype=torch.bool, device=agent.device),
                    torch.ones(batch_size, dtype=torch.bool, device=agent.device)
                ], dim=0)

                current_v = agent.getV(concat_states)
                next_v = agent.get_targetV(concat_next_states)

                current_Q1 = agent.q1(concat_states, concat_actions)
                current_Q2 = agent.q2(concat_states, concat_actions)

                _, loss_dict1 = agent.iq_loss(current_Q1, current_v, next_v, concat_dones, is_expert)
                _, loss_dict2 = agent.iq_loss(current_Q2, current_v, next_v, concat_dones, is_expert)

                # Average the two Q losses
                for key in val_metrics:
                    val_metrics[key] += 0.5 * (loss_dict1.get(key, 0) + loss_dict2.get(key, 0))

                # Compute reward_expert_mean
                y = (1 - concat_dones[is_expert].unsqueeze(1)) * agent.gamma * next_v[is_expert]
                val_metrics['reward_expert_mean'] += (current_Q1[is_expert] - y).mean().item() / n_val_batches
                val_metrics['v_mean'] += current_v.mean().item() / n_val_batches

        for key in val_metrics:
            if key not in ['reward_expert_mean', 'v_mean']:
                val_metrics[key] /= n_val_batches

        # Save best model (based on validation loss, lower is better)
        val_loss = val_metrics['total_loss']
        if -val_loss > best_val_reward:  # Using negative loss for "best"
            best_val_reward = -val_loss
            agent.save(f'{save_dir}/{experiment_prefix}_q_model.pt')

        # Print progress
        if (epoch + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1:4d} | "
                  f"Tra [L:{train_metrics['total_loss']:.3f} "
                  f"SQ:{train_metrics['softq_loss']:.3f} "
                  f"Val:{train_metrics.get('value_loss', 0):.3f} "
                  f"Chi:{train_metrics.get('chi2_loss', 0):.3f} "
                  f"R:{train_metrics['reward_expert_mean']:.3f} "
                  f"V:{train_metrics['v_mean']:.3f}] | "
                  f"Val [L:{val_metrics['total_loss']:.3f} "
                  f"SQ:{val_metrics['softq_loss']:.3f} "
                  f"Val:{val_metrics.get('value_loss', 0):.3f} "
                  f"Chi:{val_metrics.get('chi2_loss', 0):.3f} "
                  f"R:{val_metrics['reward_expert_mean']:.3f} "
                  f"V:{val_metrics['v_mean']:.3f}] | "
                  f"{elapsed/60:.1f}min", flush=True)

    # Save final model
    agent.save(f'{save_dir}/{experiment_prefix}_final.pt')

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.1f} minutes!", flush=True)
    print(f"Best validation reward: {best_val_reward:.4f}", flush=True)

    return agent, pipeline


# ============================================================================
# Main
# ============================================================================

def main():
    """Train IQ-Learn with specified hyperparameters."""
    import argparse

    parser = argparse.ArgumentParser(description='Train IQ-Learn for Inverse RL')
    parser.add_argument('--vp2_bins', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--init_temp', type=float, default=0.001)
    parser.add_argument('--method_alpha', type=float, default=0.5)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--div', type=str, default='chi',
                        choices=['chi', 'kl', 'kl2', 'kl_fix', 'js', 'hellinger'])
    parser.add_argument('--loss_type', type=str, default='value_expert',
                        choices=['value_expert', 'value', 'v0'])
    parser.add_argument('--regularize', action='store_true', default=False)
    parser.add_argument('--prefix', type=str, default='iq_learn')
    parser.add_argument('--save_dir', type=str, default='experiment/iq_learn')

    args = parser.parse_args()

    print("=" * 70, flush=True)
    print(" IQ-LEARN: INVERSE Q-LEARNING FOR IMITATION", flush=True)
    print("=" * 70, flush=True)
    print(f"\nConfiguration:", flush=True)
    print(f"  VP1: Binary (0 or 1)", flush=True)
    print(f"  VP2: {args.vp2_bins} bins", flush=True)
    print(f"  div: {args.div}", flush=True)
    print(f"  loss_type: {args.loss_type}", flush=True)
    print(f"  init_temp: {args.init_temp}", flush=True)
    print(f"  method_alpha: {args.method_alpha}", flush=True)
    print(f"  regularize: {args.regularize}", flush=True)

    agent, pipeline = train_iq_learn(
        vp2_bins=args.vp2_bins,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gamma=args.gamma,
        init_temp=args.init_temp,
        method_alpha=args.method_alpha,
        tau=args.tau,
        lr=args.lr,
        div=args.div,
        loss_type=args.loss_type,
        regularize=args.regularize,
        experiment_prefix=args.prefix,
        save_dir=args.save_dir
    )

    print("\n" + "=" * 70, flush=True)
    print(" TRAINING COMPLETE", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
