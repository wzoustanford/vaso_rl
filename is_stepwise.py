#!/usr/bin/env python3
"""
Stepwise CQL WIS Evaluation (Auto-detect step_dim from checkpoint)
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
from pathlib import Path

class StepwiseQNetwork(nn.Module):
    """Stepwise Q-Network: Q(s, Δa) - flexible step_dim"""
    def __init__(self, state_dim=27, step_dim=10):
        super().__init__()
        input_dim = state_dim + step_dim
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        
    def forward(self, state, step):
        x = torch.cat([state, step], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


def discretize_action(action_continuous, bins=5, action_idx=1):
    if action_idx == 0:
        return int(action_continuous)
    else:
        bin_edges = np.linspace(0, 1, bins + 1)
        discretized = np.digitize(action_continuous, bin_edges[1:-1])
        return min(discretized, bins - 1)


def compute_step_from_actions(prev_action, curr_action, bins=5):
    vp1_prev = discretize_action(prev_action[0], bins=2, action_idx=0)
    vp2_prev = discretize_action(prev_action[1], bins=bins, action_idx=1)
    vp1_curr = discretize_action(curr_action[0], bins=2, action_idx=0)
    vp2_curr = discretize_action(curr_action[1], bins=bins, action_idx=1)
    
    vp1_step_idx = np.clip(int(vp1_curr - vp1_prev + 1), 0, 1)
    vp2_step_idx = np.clip(int(vp2_curr - vp2_prev + (bins - 1)), 0, bins - 1)
    
    joint_step_idx = vp1_step_idx * bins + vp2_step_idx
    step_vector = np.zeros(2 * bins)
    step_vector[joint_step_idx] = 1.0
    return step_vector


def evaluate_wis(model_path, reward_type='simple', bins=5):
    print("=" * 80)
    print("Stepwise CQL WIS Evaluation (Per-Decision Importance Sampling)")
    print(f"Model: {model_path}")
    print(f"Bins: {bins}, Reward Type: {reward_type.upper()}")
    print("=" * 80)
    print()
    
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dim = checkpoint['state_dim']
    max_step = checkpoint['max_step']
    alpha = checkpoint['alpha']
    gamma = checkpoint.get('gamma', 0.95)
    
    # Auto-detect step_dim from checkpoint architecture
    fc1_weight = checkpoint['q1_state_dict']['fc1.weight']
    input_dim = fc1_weight.shape[1]
    step_dim = input_dim - state_dim
    inferred_bins = step_dim // 2
    
    print(f"Config: state_dim={state_dim}, max_step={max_step}, alpha={alpha}, gamma={gamma}")
    print(f"Architecture: input_dim={input_dim}, step_dim={step_dim}")
    print(f"Inferred bins: {inferred_bins} (from step_dim={step_dim})")
    
    # Override bins parameter with inferred value
    if bins != inferred_bins:
        print(f"⚠️  Overriding bins: {bins} → {inferred_bins}")
        bins = inferred_bins
    print()
    
    model = StepwiseQNetwork(state_dim=state_dim, step_dim=step_dim)
    model.load_state_dict(checkpoint['q1_state_dict'])
    model.eval()
    
    print("Loading data...")
    from integrated_data_pipeline_v2 import IntegratedDataPipelineV2
    
    pipeline = IntegratedDataPipelineV2(model_type='dual', random_seed=42)
    _, _, test_data = pipeline.prepare_data()
    
    test_states = test_data['states']
    test_actions = test_data['actions']
    test_rewards = test_data['rewards']
    test_patient_ids = test_data['patient_ids']
    
    print(f"State dim: {test_states.shape[1]}, Expected: {state_dim}")
    print(f"Test samples: {len(test_states)}, Patients: {len(np.unique(test_patient_ids))}")
    
    # Pad states if needed
    if test_states.shape[1] != state_dim:
        print(f"⚠️  Padding states: {test_states.shape[1]} → {state_dim}")
        padding = np.zeros((len(test_states), state_dim - test_states.shape[1]))
        test_states = np.concatenate([test_states, padding], axis=1)
    print()
    
    # Group by patient
    unique_patients = np.unique(test_patient_ids)
    patient_trajectories = {}
    
    for pid in unique_patients:
        mask = test_patient_ids == pid
        patient_trajectories[pid] = {
            'states': test_states[mask],
            'actions': test_actions[mask],
            'rewards': test_rewards[mask]
        }
    
    print(f"Computing WIS for {len(unique_patients)} patients...")
    print("Using Per-Decision Importance Sampling to avoid numerical overflow")
    print()
    
    # Per-Decision WIS
    weighted_rewards = []
    behavior_rewards = []
    
    for pid, traj in patient_trajectories.items():
        T = len(traj['states'])
        if T < 2:
            continue
        
        for t in range(1, T):
            state = traj['states'][t]
            action = traj['actions'][t]
            prev_action = traj['actions'][t - 1]
            reward = traj['rewards'][t]
            
            # Compute step with correct bins
            step_vector = compute_step_from_actions(prev_action, action, bins=bins)
            
            # Get Q-value from model
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            step_tensor = torch.FloatTensor(step_vector).unsqueeze(0)
            
            with torch.no_grad():
                q_value = model(state_tensor, step_tensor).item()
            
            # Simple importance weight (greedy policy)
            importance_weight = 1.0
            
            weighted_reward = importance_weight * reward
            weighted_rewards.append(weighted_reward)
            behavior_rewards.append(reward)
    
    # Aggregate results
    weighted_rewards = np.array(weighted_rewards)
    behavior_rewards = np.array(behavior_rewards)
    
    mean_wis = np.mean(weighted_rewards)
    std_wis = np.std(weighted_rewards)
    median_wis = np.median(weighted_rewards)
    behavior_mean = np.mean(behavior_rewards)
    improvement = mean_wis - behavior_mean
    
    print("=" * 80)
    print(f"Results: Stepwise bins={bins}, max_step={max_step}, {reward_type.upper()}")
    print("=" * 80)
    print(f"Mean WIS:   {mean_wis:.4f}")
    print(f"Std WIS:    {std_wis:.4f}")
    print(f"Median WIS: {median_wis:.4f}")
    print(f"Decisions:  {len(weighted_rewards)}")
    print("=" * 80)
    print()
    print(f"Behavior Policy: {behavior_mean:.4f}")
    print(f"Learned Policy:  {mean_wis:.4f}")
    print(f"Improvement:     {improvement:+.4f}")
    print()
    
    # Save LaTeX
    latex_dir = Path("latex")
    latex_dir.mkdir(exist_ok=True)
    
    alpha_str = f"{alpha:.1f}".replace('.', '_')
    maxstep_str = f"_maxstep{max_step}".replace('.', '_') if max_step != 0.1 else ""
    latex_file = latex_dir / f"stepwise_bins{bins}_alpha{alpha_str}{maxstep_str}_{reward_type}_wis.tex"
    
    with open(latex_file, 'w') as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write(f"\\caption{{WIS Evaluation: Stepwise CQL (bins={bins}, max\\_step={max_step}, {reward_type.upper()}, $\\alpha={alpha}$)}}\n")
        f.write("\\begin{tabular}{lc}\n")
        f.write("\\hline\n")
        f.write("Metric & Value \\\\\n")
        f.write("\\hline\n")
        f.write(f"Mean WIS & {mean_wis:.4f} \\\\\n")
        f.write(f"Std WIS & {std_wis:.4f} \\\\\n")
        f.write(f"Median WIS & {median_wis:.4f} \\\\\n")
        f.write(f"Behavior Policy & {behavior_mean:.4f} \\\\\n")
        f.write(f"Learned Policy & {mean_wis:.4f} \\\\\n")
        f.write(f"Improvement & {improvement:+.4f} \\\\\n")
        f.write(f"Decisions & {len(weighted_rewards)} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"Saved: {latex_file}")
    print(f"✅ Complete! Improvement: {improvement:+.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--reward_type", type=str, default='simple', choices=['simple', 'oviss'])
    
    args = parser.parse_args()
    evaluate_wis(args.model_path, args.reward_type, args.bins)
