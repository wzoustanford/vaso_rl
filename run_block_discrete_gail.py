#!/usr/bin/env python3
"""
Block Discrete GAIL (Generative Adversarial Imitation Learning) Training Script
Offline GAIL baseline for imitation learning comparison.
Discretizes VP2 continuous action space into bins:
- VP1: Binary (0 or 1)
- VP2: Discretized into N bins (0 to 0.5 mcg/kg/min)
- Total discrete actions: 2 * vp2_bins

Reference: Ho & Ermon, "Generative Adversarial Imitation Learning", NeurIPS 2016
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import time
import os
import sys

# Import our unified pipeline
from integrated_data_pipeline_v3 import IntegratedDataPipelineV3

# Force unbuffered output
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__


# ============================================================================
# GAIL Networks
# ============================================================================

class GAILPolicyNetwork(nn.Module):
    """
    Policy network for block discrete actions.
    Outputs categorical distribution over 2 * vp2_bins actions.
    """

    def __init__(self, state_dim: int, vp2_bins: int = 5, hidden_dim: int = 128):
        super().__init__()
        self.vp2_bins = vp2_bins
        self.total_actions = 2 * vp2_bins

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 64)
        self.fc4 = nn.Linear(64, self.total_actions)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Returns action probabilities [batch_size, total_actions]."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return F.softmax(self.fc4(x), dim=-1)

    def get_distribution(self, state: torch.Tensor) -> Categorical:
        """Returns Categorical distribution over actions."""
        probs = self.forward(state)
        return Categorical(probs)


class GAILDiscriminator(nn.Module):
    """
    Discriminator for GAIL.
    Takes (state, one-hot action) and outputs D(s,a) in [0, 1].
    D(s,a) -> 1 for expert, D(s,a) -> 0 for policy.
    """

    def __init__(self, state_dim: int, vp2_bins: int = 5, hidden_dim: int = 128):
        super().__init__()
        self.vp2_bins = vp2_bins
        self.total_actions = 2 * vp2_bins

        input_dim = state_dim + self.total_actions

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 64)
        self.fc4 = nn.Linear(64, 1)

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
            D(s,a): [batch_size, 1] in [0, 1]
        """
        action_one_hot = F.one_hot(action_idx.long(), num_classes=self.total_actions).float()
        x = torch.cat([state, action_one_hot], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return torch.sigmoid(self.fc4(x))


# ============================================================================
# Offline GAIL Agent
# ============================================================================

class BlockDiscreteGAIL:
    """
    Offline GAIL agent with block discrete action space.
    - Policy trained via REINFORCE with discriminator reward
    - Discriminator trained to distinguish expert vs policy actions
    - Transition-level (no temporal structure)
    """

    def __init__(self, state_dim: int, vp2_bins: int = 5,
                 lr_policy: float = 1e-3, lr_discriminator: float = 1e-3,
                 entropy_coeff: float = 0.01, grad_clip: float = 1.0):
        self.state_dim = state_dim
        self.vp2_bins = vp2_bins
        self.total_actions = 2 * vp2_bins
        self.entropy_coeff = entropy_coeff
        self.grad_clip = grad_clip

        # VP2 bin edges for action conversion
        self.vp2_bin_edges = np.linspace(0, 0.5, vp2_bins + 1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize networks
        self.policy = GAILPolicyNetwork(state_dim, vp2_bins).to(self.device)
        self.discriminator = GAILDiscriminator(state_dim, vp2_bins).to(self.device)

        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr_policy)
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr_discriminator)

    def continuous_to_discrete_action(self, continuous_action: np.ndarray) -> int:
        """Convert continuous action [vp1, vp2] to discrete action index."""
        vp1, vp2 = continuous_action
        vp1_idx = int(vp1)
        vp2_bin = np.digitize(vp2, self.vp2_bin_edges) - 1
        vp2_bin = np.clip(vp2_bin, 0, self.vp2_bins - 1)
        return vp1_idx * self.vp2_bins + vp2_bin

    def continuous_to_discrete_batch(self, actions: np.ndarray) -> np.ndarray:
        """Vectorized conversion of continuous actions to discrete indices."""
        vp1 = actions[:, 0].astype(int)
        vp2 = np.clip(actions[:, 1], 0, 0.5)
        vp2_bin = np.digitize(vp2, self.vp2_bin_edges) - 1
        vp2_bin = np.clip(vp2_bin, 0, self.vp2_bins - 1)
        return vp1 * self.vp2_bins + vp2_bin

    def discrete_to_continuous_action(self, action_idx: int) -> np.ndarray:
        """Convert discrete action index to continuous action [vp1, vp2]."""
        vp1_idx = action_idx // self.vp2_bins
        vp2_bin = action_idx % self.vp2_bins
        vp2_value = (self.vp2_bin_edges[vp2_bin] + self.vp2_bin_edges[vp2_bin + 1]) / 2
        return np.array([float(vp1_idx), vp2_value])

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select best action (argmax of policy) for evaluation."""
        with torch.no_grad():
            if state.ndim == 1:
                state = state.reshape(1, -1)
            batch_size = state.shape[0]
            state_tensor = torch.FloatTensor(state).to(self.device)
            probs = self.policy(state_tensor)
            best_indices = probs.argmax(dim=1).cpu().numpy()

            vp1_actions = (best_indices // self.vp2_bins).astype(float)
            vp2_bin_indices = best_indices % self.vp2_bins
            vp2_bin_centers = (self.vp2_bin_edges[:-1] + self.vp2_bin_edges[1:]) / 2
            vp2_actions = vp2_bin_centers[vp2_bin_indices]
            actions = np.stack([vp1_actions, vp2_actions], axis=1)
            return actions if batch_size > 1 else actions[0]

    def update(self, states: torch.Tensor, actions: torch.Tensor,
               n_discriminator_steps: int = 1) -> dict:
        """
        One GAIL update step.
        Args:
            states: [batch_size, state_dim] expert states
            actions: [batch_size, 2] expert continuous actions
            n_discriminator_steps: number of discriminator updates per policy update
        """
        batch_size = states.shape[0]

        # Convert expert actions to discrete indices
        expert_indices = torch.LongTensor(
            self.continuous_to_discrete_batch(actions.cpu().numpy())
        ).to(self.device)

        # ---- Train Discriminator ----
        d_losses = []
        for _ in range(n_discriminator_steps):
            # Sample actions from current policy (no grad for policy here)
            with torch.no_grad():
                dist = self.policy.get_distribution(states)
                policy_actions = dist.sample()

            # Discriminator predictions
            d_expert = self.discriminator(states, expert_indices)
            d_policy = self.discriminator(states, policy_actions)

            # BCE loss: expert -> 1, policy -> 0
            loss_d = -torch.mean(torch.log(d_expert + 1e-8)) - torch.mean(torch.log(1 - d_policy + 1e-8))

            self.discriminator_optimizer.zero_grad()
            loss_d.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.grad_clip)
            self.discriminator_optimizer.step()
            d_losses.append(loss_d.item())

        # ---- Train Policy (REINFORCE) ----
        dist = self.policy.get_distribution(states)
        sampled_actions = dist.sample()
        log_probs = dist.log_prob(sampled_actions)

        # Reward: log(D(s, a_policy)) — policy wants to fool discriminator
        with torch.no_grad():
            d_score = self.discriminator(states, sampled_actions).squeeze()
            reward = torch.log(d_score + 1e-8)

        # Baseline subtraction (batch mean) to reduce variance
        advantage = reward - reward.mean()

        # REINFORCE loss + entropy bonus
        policy_loss = -torch.mean(log_probs * advantage)
        entropy = dist.entropy().mean()
        total_policy_loss = policy_loss - self.entropy_coeff * entropy

        self.policy_optimizer.zero_grad()
        total_policy_loss.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        self.policy_optimizer.step()

        # Compute metrics
        with torch.no_grad():
            d_expert_mean = self.discriminator(states, expert_indices).mean().item()
            d_policy_mean = self.discriminator(states, sampled_actions).mean().item()
            # Policy accuracy: how often does policy pick the same action as expert?
            policy_argmax = dist.probs.argmax(dim=1)
            expert_match = (policy_argmax == expert_indices).float().mean().item()

        return {
            'd_loss': np.mean(d_losses),
            'policy_loss': policy_loss.item(),
            'entropy': entropy.item(),
            'd_expert': d_expert_mean,
            'd_policy': d_policy_mean,
            'expert_match_rate': expert_match,
        }

    def save(self, filepath: str):
        """Save model checkpoint."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'state_dim': self.state_dim,
            'vp2_bins': self.vp2_bins,
            'entropy_coeff': self.entropy_coeff,
        }, filepath)


# ============================================================================
# Training Function
# ============================================================================

def train_block_discrete_gail(
    vp2_bins: int = 5,
    epochs: int = 100,
    lr_policy: float = 1e-3,
    lr_discriminator: float = 1e-3,
    entropy_coeff: float = 0.01,
    n_discriminator_steps: int = 1,
    suffix: str = "",
    save_dir: str = "experiment/gail",
    combined_or_train_data_path: str = None,
    eval_data_path: str = None,
):
    """Train Block Discrete GAIL.

    Args:
        vp2_bins: Number of bins for VP2 discretization
        epochs: Number of training epochs
        lr_policy: Learning rate for policy network
        lr_discriminator: Learning rate for discriminator
        entropy_coeff: Entropy bonus coefficient
        n_discriminator_steps: Discriminator updates per policy update
        suffix: Suffix for experiment naming
        save_dir: Directory to save models
        combined_or_train_data_path: Path to training dataset
        eval_data_path: Path to evaluation dataset
    """
    os.makedirs(save_dir, exist_ok=True)

    # Initialize data pipeline (manual reward — GAIL doesn't use rewards)
    print("\nInitializing Block Discrete GAIL data pipeline...", flush=True)
    pipeline = IntegratedDataPipelineV3(
        model_type='dual', reward_source='manual', random_seed=42,
        combined_or_train_data_path=combined_or_train_data_path,
        eval_data_path=eval_data_path
    )

    experiment_prefix = f"gail_{suffix}" if suffix else "gail"

    print("=" * 70, flush=True)
    print(f" BLOCK DISCRETE GAIL TRAINING", flush=True)
    print(f" Prefix: {experiment_prefix}", flush=True)
    print("=" * 70, flush=True)

    train_data, val_data, test_data = pipeline.prepare_data()
    state_dim = train_data['states'].shape[1]

    print("\n" + "=" * 70, flush=True)
    print("SETTINGS:", flush=True)
    print(f"  State dimension: {state_dim}", flush=True)
    print(f"  Action dimension: 2 (VP1: binary, VP2: {vp2_bins} bins)", flush=True)
    print(f"  Total discrete actions: {2 * vp2_bins}", flush=True)
    print(f"  LR (policy): {lr_policy}", flush=True)
    print(f"  LR (discriminator): {lr_discriminator}", flush=True)
    print(f"  Entropy coeff: {entropy_coeff}", flush=True)
    print(f"  Discriminator steps per update: {n_discriminator_steps}", flush=True)
    print(f"  Batch size: 128", flush=True)
    print(f"  Epochs: {epochs}", flush=True)
    print("=" * 70, flush=True)

    # Initialize agent
    agent = BlockDiscreteGAIL(
        state_dim=state_dim,
        vp2_bins=vp2_bins,
        lr_policy=lr_policy,
        lr_discriminator=lr_discriminator,
        entropy_coeff=entropy_coeff,
        grad_clip=1.0,
    )

    # Training loop
    batch_size = 128
    print(f"\nTraining for {epochs} epochs with batch size {batch_size}...", flush=True)
    start_time = time.time()

    best_expert_match = 0.0

    for epoch in range(epochs):
        # Training phase
        agent.policy.train()
        agent.discriminator.train()

        epoch_metrics = {
            'd_loss': 0, 'policy_loss': 0, 'entropy': 0,
            'd_expert': 0, 'd_policy': 0, 'expert_match_rate': 0,
        }

        n_batches = len(train_data['states']) // batch_size

        for _ in range(n_batches):
            batch = pipeline.get_batch(batch_size=batch_size, split='train')
            states = torch.FloatTensor(batch['states']).to(agent.device)
            actions = torch.FloatTensor(batch['actions']).to(agent.device)

            metrics = agent.update(states, actions, n_discriminator_steps=n_discriminator_steps)

            for key in epoch_metrics:
                epoch_metrics[key] += metrics.get(key, 0)

        for key in epoch_metrics:
            epoch_metrics[key] /= n_batches

        # Validation: compute expert match rate on val set
        agent.policy.eval()
        val_match_rates = []
        with torch.no_grad():
            for _ in range(10):
                batch = pipeline.get_batch(batch_size=batch_size, split='val')
                states = torch.FloatTensor(batch['states']).to(agent.device)
                actions = torch.FloatTensor(batch['actions']).to(agent.device)

                expert_indices = torch.LongTensor(
                    agent.continuous_to_discrete_batch(actions.cpu().numpy())
                ).to(agent.device)

                probs = agent.policy(states)
                policy_argmax = probs.argmax(dim=1)
                match_rate = (policy_argmax == expert_indices).float().mean().item()
                val_match_rates.append(match_rate)

        val_match = np.mean(val_match_rates)

        # Save best model (by validation expert match rate)
        if val_match > best_expert_match:
            best_expert_match = val_match
            save_path = f'{save_dir}/{experiment_prefix}_bins{vp2_bins}_best.pt'
            print(f'  Saving best model (val match={val_match:.4f}) at {save_path}')
            agent.save(save_path)

        # Print progress
        if (epoch + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1}: "
                  f"D_loss={epoch_metrics['d_loss']:.4f}, "
                  f"π_loss={epoch_metrics['policy_loss']:.4f}, "
                  f"entropy={epoch_metrics['entropy']:.3f}, "
                  f"D(expert)={epoch_metrics['d_expert']:.3f}, "
                  f"D(policy)={epoch_metrics['d_policy']:.3f}, "
                  f"train_match={epoch_metrics['expert_match_rate']:.3f}, "
                  f"val_match={val_match:.3f}, "
                  f"Time={elapsed/60:.1f}min", flush=True)

    # Save final model
    final_path = f'{save_dir}/{experiment_prefix}_bins{vp2_bins}_final.pt'
    agent.save(final_path)

    total_time = time.time() - start_time
    print(f"\nGAIL training completed in {total_time/60:.1f} minutes!", flush=True)
    print(f"Best val expert match rate: {best_expert_match:.4f}", flush=True)
    print("Models saved:", flush=True)
    print(f"  - {save_dir}/{experiment_prefix}_bins{vp2_bins}_best.pt", flush=True)
    print(f"  - {final_path}", flush=True)

    return agent, pipeline, experiment_prefix


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Train Block Discrete GAIL')
    parser.add_argument('--vp2_bins', type=int, default=5,
                        help='Number of bins for VP2 discretization (default: 5)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--lr_policy', type=float, default=1e-3,
                        help='Policy learning rate (default: 1e-3)')
    parser.add_argument('--lr_discriminator', type=float, default=1e-3,
                        help='Discriminator learning rate (default: 1e-3)')
    parser.add_argument('--entropy_coeff', type=float, default=0.01,
                        help='Entropy bonus coefficient (default: 0.01)')
    parser.add_argument('--n_discriminator_steps', type=int, default=1,
                        help='Discriminator updates per policy update (default: 1)')
    parser.add_argument('--suffix', type=str, default='',
                        help='Suffix for experiment naming')
    parser.add_argument('--save_dir', type=str, default='experiment/gail',
                        help='Directory to save models (default: experiment/gail)')
    parser.add_argument('--combined_or_train_data_path', type=str, default=None,
                        help='Path to training dataset')
    parser.add_argument('--eval_data_path', type=str, default=None,
                        help='Path to evaluation dataset')
    args = parser.parse_args()

    print("=" * 70, flush=True)
    print(" BLOCK DISCRETE GAIL TRAINING", flush=True)
    print("=" * 70, flush=True)

    print("\nConfiguration:", flush=True)
    print("  - VP1: Binary (0 or 1)", flush=True)
    print(f"  - VP2: Discretized into {args.vp2_bins} bins (0 to 0.5 mcg/kg/min)", flush=True)
    print(f"  - Total discrete actions: {2 * args.vp2_bins}", flush=True)
    print(f"  - Epochs: {args.epochs}", flush=True)
    print(f"  - LR (policy): {args.lr_policy}", flush=True)
    print(f"  - LR (discriminator): {args.lr_discriminator}", flush=True)
    print(f"  - Entropy coeff: {args.entropy_coeff}", flush=True)
    print(f"  - Discriminator steps: {args.n_discriminator_steps}", flush=True)
    if args.eval_data_path:
        print(f"  - Dataset mode: DUAL-DATASET", flush=True)
        print(f"    Train data: {args.combined_or_train_data_path or 'default'}", flush=True)
        print(f"    Eval data:  {args.eval_data_path}", flush=True)
    else:
        print(f"  - Dataset mode: SINGLE-DATASET", flush=True)
        print(f"    Data path: {args.combined_or_train_data_path or 'default'}", flush=True)

    agent, pipeline, exp_prefix = train_block_discrete_gail(
        vp2_bins=args.vp2_bins,
        epochs=args.epochs,
        lr_policy=args.lr_policy,
        lr_discriminator=args.lr_discriminator,
        entropy_coeff=args.entropy_coeff,
        n_discriminator_steps=args.n_discriminator_steps,
        suffix=args.suffix,
        save_dir=args.save_dir,
        combined_or_train_data_path=args.combined_or_train_data_path,
        eval_data_path=args.eval_data_path,
    )

    print("\n" + "=" * 70, flush=True)
    print(" TRAINING COMPLETE", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
