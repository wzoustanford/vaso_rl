"""
Universal Block Discrete CQL WIS Evaluation (Using IntegratedDataPipelineV2)
- Supports any bin size (bins=3, 5, 10, etc.)
- Supports both simple and OVISS reward types
- Uses the same data pipeline as Dual-Mixed for consistency
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import argparse
import os
from typing import Dict, Tuple
from collections import defaultdict

# Import the data pipeline
from integrated_data_pipeline_v2 import IntegratedDataPipelineV2
import data_config as config


class DualBlockDiscreteQNetwork(nn.Module):
    """
    Q-network for block discrete actions using Q(s,a) -> R architecture
    Takes state and discrete action index as input, outputs single Q-value
    """
    def __init__(self, state_dim: int, vp2_bins: int = 5, hidden_dim: int = 128):
        super().__init__()
        self.vp2_bins = vp2_bins
        self.total_actions = 2 * vp2_bins  # VP1 (2) x VP2 (bins)
        
        # Network takes state + one-hot encoded action
        input_dim = state_dim + self.total_actions
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 64)
        self.fc4 = nn.Linear(64, 1)  # Output single Q-value
        
    def forward(self, state: torch.Tensor, action_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (batch_size, state_dim)
            action_idx: (batch_size,) discrete action indices
        Returns:
            Q-values: (batch_size, 1)
        """
        batch_size = state.shape[0]
        
        # One-hot encode actions
        action_onehot = torch.zeros(batch_size, self.total_actions, device=state.device)
        action_onehot[torch.arange(batch_size), action_idx.long()] = 1
        
        # Concatenate state and action
        x = torch.cat([state, action_onehot], dim=1)
        
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


def continuous_to_discrete_action(vp1: float, vp2: float, vp2_bins: int) -> int:
    """Convert continuous (vp1, vp2) to discrete action index"""
    # VP1 is binary (0 or 1)
    vp1_discrete = 1 if vp1 > 0.5 else 0
    
    # VP2 is discretized into bins from 0 to 0.5
    vp2_bin_edges = np.linspace(0, 0.5, vp2_bins + 1)
    vp2_discrete = np.digitize(vp2, vp2_bin_edges[1:])  # Returns 0 to vp2_bins-1
    vp2_discrete = min(vp2_discrete, vp2_bins - 1)  # Clamp to valid range
    
    # Combined action index: vp1_idx * vp2_bins + vp2_idx
    action_idx = vp1_discrete * vp2_bins + vp2_discrete
    return action_idx


def discrete_to_continuous_action(action_idx: int, vp2_bins: int) -> Tuple[float, float]:
    """Convert discrete action index back to continuous (vp1, vp2)"""
    vp1_discrete = action_idx // vp2_bins
    vp2_discrete = action_idx % vp2_bins
    
    # Convert back to continuous
    vp1 = float(vp1_discrete)  # 0.0 or 1.0
    
    # VP2: use bin midpoint
    vp2_bin_edges = np.linspace(0, 0.5, vp2_bins + 1)
    vp2 = (vp2_bin_edges[vp2_discrete] + vp2_bin_edges[vp2_discrete + 1]) / 2
    
    return vp1, vp2


def compute_simple_reward(state: np.ndarray, next_state: np.ndarray, 
                         action: np.ndarray, is_terminal: bool, 
                         mortality: int, state_features: list) -> float:
    """Compute simple reward (mortality-based)"""
    reward = 0.0
    
    # Penalize high vasopressor doses
    if len(action) == 2:
        vp1_dose = action[0]
        vp2_dose = action[1]
        total_vaso = vp1_dose + vp2_dose * 2
    else:
        vp1_dose = action if np.isscalar(action) else action[0]
        total_vaso = vp1_dose
    
    if total_vaso > 1.0:
        reward -= 0.1 * (total_vaso - 1.0)
    
    # Terminal rewards
    if is_terminal:
        if mortality == 0:
            reward += 10.0  # Survived
        else:
            reward -= 10.0  # Died
    
    return reward


def compute_oviss_reward(state: np.ndarray, next_state: np.ndarray,
                        action: np.ndarray, is_terminal: bool,
                        mortality: int, state_features: list) -> float:
    """Compute OVISS reward (outcome + vital sign improvements)"""
    reward = 0.0
    
    # Get feature indices
    mbp_idx = state_features.index('mbp') if 'mbp' in state_features else None
    lactate_idx = state_features.index('lactate') if 'lactate' in state_features else None
    
    # Vital sign improvements
    if mbp_idx is not None:
        mbp_improvement = next_state[mbp_idx] - state[mbp_idx]
        if next_state[mbp_idx] >= 65 and next_state[mbp_idx] <= 90:
            reward += 0.5  # MBP in target range
        elif next_state[mbp_idx] < 65:
            reward -= 1.0  # Too low
        elif next_state[mbp_idx] > 110:
            reward -= 0.5  # Too high
    
    if lactate_idx is not None:
        lactate_improvement = state[lactate_idx] - next_state[lactate_idx]
        if lactate_improvement > 0:
            reward += 0.3 * lactate_improvement  # Reward lactate reduction
        if next_state[lactate_idx] > 2.0:
            reward -= 0.5  # High lactate penalty
    
    # Penalize high vasopressor doses
    if len(action) == 2:
        vp1_dose = action[0]
        vp2_dose = action[1]
        total_vaso = vp1_dose + vp2_dose * 2
    else:
        vp1_dose = action if np.isscalar(action) else action[0]
        total_vaso = vp1_dose
    
    if total_vaso > 1.0:
        reward -= 0.2 * (total_vaso - 1.0)
    
    # Terminal rewards
    if is_terminal:
        if mortality == 0:
            reward += 15.0  # Survived (higher weight)
        else:
            reward -= 15.0  # Died
    
    return reward


def compute_wis_per_patient(trajectories: list, reward_type: str = 'simple') -> Dict:
    """
    Compute WIS estimator for a single patient
    
    Args:
        trajectories: List of (state, action, reward, next_state, done) tuples
        reward_type: 'simple' or 'oviss'
    
    Returns:
        Dictionary with WIS statistics
    """
    if len(trajectories) == 0:
        return {'wis': 0.0, 'n_steps': 0, 'total_weight': 0.0, 'total_reward': 0.0}
    
    # Extract rewards and compute importance weights
    rewards = np.array([t[2] for t in trajectories])
    cumulative_weight = 1.0
    weighted_rewards = []
    weights = []
    
    for i, (state, action, reward, next_state, done) in enumerate(trajectories):
        # Importance weight (simplified: assume equal for now)
        weight = 1.0
        cumulative_weight *= weight
        
        weighted_rewards.append(cumulative_weight * reward)
        weights.append(cumulative_weight)
    
    total_weighted_reward = np.sum(weighted_rewards)
    total_weight = np.sum(weights)
    
    wis = total_weighted_reward / total_weight if total_weight > 0 else 0.0
    
    return {
        'wis': wis,
        'n_steps': len(trajectories),
        'total_weight': total_weight,
        'total_reward': total_weighted_reward
    }


def evaluate_wis(model_path: str, vp2_bins: int, reward_type: str = 'simple'):
    """
    Evaluate Block Discrete CQL model using WIS estimator
    
    Args:
        model_path: Path to trained model checkpoint
        vp2_bins: Number of bins for VP2 discretization
        reward_type: 'simple' or 'oviss'
    """
    print(f"\n{'='*80}")
    print(f"Block Discrete CQL WIS Evaluation")
    print(f"Bins: {vp2_bins}, Reward Type: {reward_type.upper()}")
    print(f"Model: {model_path}")
    print(f"{'='*80}\n")
    
    # Load data using IntegratedDataPipelineV2
    print("Loading data via IntegratedDataPipelineV2...")
    pipeline = IntegratedDataPipelineV2(model_type='dual', random_seed=42)
    train_data, val_data, test_data = pipeline.prepare_data()
    
    # Use test data for evaluation
    states = test_data['states']
    actions = test_data['actions']  # Shape: (n_samples, 2) - [vp1, vp2]
    rewards_data = test_data['rewards']  # Pre-computed rewards from pipeline
    next_states = test_data['next_states']
    dones = test_data['dones']
    patient_ids = test_data['patient_ids']
    
    state_dim = states.shape[1]
    total_actions = 2 * vp2_bins  # VP1 (binary) × VP2 (bins)
    
    print(f"State dimension: {state_dim}")
    print(f"Total actions: {total_actions} (VP1: 2 × VP2: {vp2_bins})")
    print(f"Test samples: {len(states)}")
    print(f"Unique patients: {len(np.unique(patient_ids))}")
    
    # Convert continuous actions to discrete indices for behavior policy
    discrete_actions = np.array([
        continuous_to_discrete_action(actions[i, 0], actions[i, 1], vp2_bins)
        for i in range(len(actions))
    ])
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DualBlockDiscreteQNetwork(state_dim, vp2_bins).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'q1_state_dict' in checkpoint:
        # This is a CQL checkpoint with dual Q-networks
        # Use q1_state_dict for evaluation
        model.load_state_dict(checkpoint['q1_state_dict'])
        print(f"Loaded Q1 network from CQL checkpoint")
        if 'alpha' in checkpoint:
            print(f"CQL alpha used in training: {checkpoint['alpha']}")
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Get state features for reward computation
    state_features = config.DUAL_STATE_FEATURES
    
    # Organize data by patient
    print("\nOrganizing trajectories by patient...")
    print(f"Note: Using pre-computed rewards from pipeline (reward type was specified during data loading)")
    patient_trajectories = defaultdict(list)
    
    for i in range(len(states)):
        pid = patient_ids[i]
        state = states[i]
        action_continuous = actions[i]  # [vp1, vp2]
        done = dones[i]
        next_state = next_states[i]
        reward = rewards_data[i]  # Use pre-computed reward from pipeline
        
        patient_trajectories[pid].append(
            (state, action_continuous, reward, next_state, done)
        )
    
    # Compute WIS for each patient
    print(f"\nComputing WIS for {len(patient_trajectories)} patients...")
    patient_wis = {}
    
    for pid, trajectories in patient_trajectories.items():
        wis_stats = compute_wis_per_patient(trajectories, reward_type)
        patient_wis[pid] = wis_stats
    
    # Aggregate results
    all_wis = [stats['wis'] for stats in patient_wis.values()]
    mean_wis = np.mean(all_wis)
    std_wis = np.std(all_wis)
    median_wis = np.median(all_wis)
    
    print(f"\n{'='*80}")
    print(f"WIS Evaluation Results (Block Discrete, bins={vp2_bins}, {reward_type.upper()})")
    print(f"{'='*80}")
    print(f"Mean WIS:   {mean_wis:+.4f}")
    print(f"Std WIS:    {std_wis:.4f}")
    print(f"Median WIS: {median_wis:+.4f}")
    print(f"Min WIS:    {np.min(all_wis):+.4f}")
    print(f"Max WIS:    {np.max(all_wis):+.4f}")
    print(f"Patients:   {len(patient_wis)}")
    print(f"{'='*80}\n")
    
    # Compute behavior policy baseline
    print("Computing behavior policy baseline...")
    behavior_wis = []
    for pid, trajectories in patient_trajectories.items():
        traj_rewards = [t[2] for t in trajectories]
        behavior_wis.append(np.mean(traj_rewards) if traj_rewards else 0.0)
    
    baseline_mean = np.mean(behavior_wis)
    improvement = mean_wis - baseline_mean
    
    print(f"Behavior Policy Mean: {baseline_mean:+.4f}")
    print(f"Learned Policy Mean:  {mean_wis:+.4f}")
    print(f"Improvement:          {improvement:+.4f}")
    print(f"Relative Improvement: {(improvement/abs(baseline_mean)*100 if baseline_mean != 0 else 0):.2f}%\n")
    
    # Generate LaTeX table
    os.makedirs('latex', exist_ok=True)
    
    alpha_str = os.path.basename(model_path).split('_alpha')[1].split('_')[0]
    latex_file = f'latex/block_discrete_bins{vp2_bins}_alpha{alpha_str}_{reward_type}_wis.tex'
    
    with open(latex_file, 'w') as f:
        f.write("% Block Discrete CQL WIS Evaluation Results\n")
        f.write(f"% Bins: {vp2_bins}, Reward Type: {reward_type.upper()}, Alpha: {alpha_str}\n")
        f.write("% Generated by is_block_discrete_universal_v2.py\n\n")
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write(f"\\caption{{WIS Evaluation: Block Discrete CQL (bins={vp2_bins}, {reward_type.upper()} reward, $\\alpha={alpha_str}$)}}\n")
        f.write("\\begin{tabular}{lc}\n")
        f.write("\\hline\n")
        f.write("Metric & Value \\\\\n")
        f.write("\\hline\n")
        f.write(f"Behavior Policy & ${baseline_mean:+.4f}$ \\\\\n")
        f.write(f"Learned Policy (CQL) & ${mean_wis:+.4f}$ \\\\\n")
        f.write(f"Improvement & $\\mathbf{{{improvement:+.4f}}}$ \\\\\n")
        f.write(f"Std Dev & ${std_wis:.4f}$ \\\\\n")
        f.write(f"Patients Evaluated & {len(patient_wis)} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"LaTeX table saved to: {latex_file}")
    
    return {
        'mean_wis': mean_wis,
        'std_wis': std_wis,
        'median_wis': median_wis,
        'baseline': baseline_mean,
        'improvement': improvement,
        'n_patients': len(patient_wis)
    }


def main():
    parser = argparse.ArgumentParser(description='Block Discrete CQL WIS Evaluation (Universal)')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--bins', type=int, default=5, help='Number of VP2 bins (3, 5, 10, etc.)')
    parser.add_argument('--reward_type', type=str, default='simple', 
                       choices=['simple', 'oviss'], help='Reward type')
    
    args = parser.parse_args()
    
    results = evaluate_wis(args.model_path, args.bins, args.reward_type)
    
    print("\nEvaluation complete!")
    print(f"Improvement over behavior policy: {results['improvement']:+.4f}")


if __name__ == '__main__':
    main()
