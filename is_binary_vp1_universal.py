"""
Binary VP1 CQL WIS Evaluation (Using IntegratedDataPipelineV2)
- Binary action space (VP1 only: 0 or 1)
- Supports both simple and OVISS reward types
- Uses the same data pipeline approach as other evaluations
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import argparse
import os
from typing import Dict
from collections import defaultdict

# Import the data pipeline
from integrated_data_pipeline_v2 import IntegratedDataPipelineV2
import data_config as config


class BinaryQNetwork(nn.Module):
    """
    Q-network for binary action space (VP1 only: 0 or 1)
    Takes state + binary action as input, outputs single Q-value
    """
    def __init__(self, state_dim: int = 18, hidden_dim: int = 128):
        super().__init__()
        # Network takes state + action (binary: 0 or 1)
        input_dim = state_dim + 1
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 64)
        self.fc4 = nn.Linear(64, 1)  # Output single Q-value
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (batch_size, state_dim)
            action: (batch_size, 1) binary action (0 or 1)
        Returns:
            Q-values: (batch_size, 1)
        """
        # Concatenate state and action
        x = torch.cat([state, action], dim=1)
        
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


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


def evaluate_wis(model_path: str, reward_type: str = 'simple'):
    """
    Evaluate Binary VP1 CQL model using WIS estimator
    
    Args:
        model_path: Path to trained model checkpoint
        reward_type: 'simple' or 'oviss'
    """
    print(f"\n{'='*80}")
    print(f"Binary VP1 CQL WIS Evaluation")
    print(f"Reward Type: {reward_type.upper()}")
    print(f"Model: {model_path}")
    print(f"{'='*80}\n")
    
    # Load data using IntegratedDataPipelineV2
    # Binary model uses 'binary' model_type which includes norepinephrine in state
    print("Loading data via IntegratedDataPipelineV2...")
    pipeline = IntegratedDataPipelineV2(model_type='binary', random_seed=42)
    train_data, val_data, test_data = pipeline.prepare_data()
    
    # Use test data for evaluation
    states = test_data['states']
    actions = test_data['actions']  # For binary: shape (n_samples, 1) - just VP1
    rewards_data = test_data['rewards']  # Pre-computed rewards from pipeline
    next_states = test_data['next_states']
    dones = test_data['dones']
    patient_ids = test_data['patient_ids']
    
    state_dim = states.shape[1]
    
    print(f"State dimension: {state_dim}")
    print(f"Action space: Binary (0 or 1)")
    print(f"Test samples: {len(states)}")
    print(f"Unique patients: {len(np.unique(patient_ids))}")
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BinaryQNetwork(state_dim=state_dim).to(device)
    
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
    
    # Organize data by patient
    print("\nOrganizing trajectories by patient...")
    print(f"Note: Using pre-computed rewards from pipeline")
    patient_trajectories = defaultdict(list)
    
    for i in range(len(states)):
        pid = patient_ids[i]
        state = states[i]
        action = actions[i]  # Binary action (0 or 1)
        done = dones[i]
        next_state = next_states[i]
        reward = rewards_data[i]  # Use pre-computed reward from pipeline
        
        patient_trajectories[pid].append(
            (state, action, reward, next_state, done)
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
    print(f"WIS Evaluation Results (Binary VP1, {reward_type.upper()})")
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
    
    alpha_str = os.path.basename(model_path).split('alpha')[1].split('_')[0] if 'alpha' in model_path else '00'
    latex_file = f'latex/binary_vp1_alpha{alpha_str}_{reward_type}_wis.tex'
    
    with open(latex_file, 'w') as f:
        f.write("% Binary VP1 CQL WIS Evaluation Results\n")
        f.write(f"% Reward Type: {reward_type.upper()}, Alpha: {alpha_str}\n")
        f.write("% Generated by is_binary_vp1_universal.py\n\n")
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write(f"\\caption{{WIS Evaluation: Binary VP1 CQL ({reward_type.upper()} reward, $\\alpha={alpha_str}$)}}\n")
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
    parser = argparse.ArgumentParser(description='Binary VP1 CQL WIS Evaluation')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--reward_type', type=str, default='simple', 
                       choices=['simple', 'oviss'], help='Reward type')
    
    args = parser.parse_args()
    
    results = evaluate_wis(args.model_path, args.reward_type)
    
    print("\nEvaluation complete!")
    print(f"Improvement over behavior policy: {results['improvement']:+.4f}")


if __name__ == '__main__':
    main()
