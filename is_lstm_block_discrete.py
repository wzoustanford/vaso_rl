"""
LSTM Block Discrete CQL WIS Evaluation
- Handles sequence-based LSTM architecture
- Supports burn-in periods and overlapping sequences
- Works with Block Discrete action space (VP1 × VP2 bins)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import argparse
import os
from typing import Dict, Tuple, Optional, List
from collections import defaultdict

# Import the data pipeline
from integrated_data_pipeline_v2 import IntegratedDataPipelineV2
import data_config as config


class LSTMDiscreteQNetwork(nn.Module):
    """
    Q-Network with LSTM for discrete block dosing actions
    Matches the architecture from lstm_block_discrete_cql_network.py
    """
    
    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        hidden_dim: int = 32,
        lstm_hidden: int = 32,
        num_lstm_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.lstm_hidden = lstm_hidden
        self.num_lstm_layers = num_lstm_layers
        
        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=lstm_hidden,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )
        
        # Output layer - Q-values for all discrete actions
        self.q_head = nn.Linear(lstm_hidden, num_actions)
        
    def forward(
        self,
        states: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through LSTM Q-network
        
        Args:
            states: [batch_size, sequence_length, state_dim]
            hidden_state: Optional LSTM hidden state (h, c)
        
        Returns:
            q_values: [batch_size, sequence_length, num_actions]
            new_hidden: Updated LSTM hidden state
        """
        batch_size, seq_len, _ = states.shape
        
        # Feature extraction
        states_flat = states.reshape(-1, self.state_dim)
        features_flat = self.feature_extractor(states_flat)
        features = features_flat.reshape(batch_size, seq_len, -1)
        
        # LSTM
        lstm_out, new_hidden = self.lstm(features, hidden_state)
        
        # Q-values
        q_values_flat = self.q_head(lstm_out.reshape(-1, self.lstm_hidden))
        q_values = q_values_flat.reshape(batch_size, seq_len, self.num_actions)
        
        return q_values, new_hidden


def prepare_sequences(
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    patient_ids: np.ndarray,
    sequence_length: int = 20,
    burn_in_length: int = 8,
    overlap: int = 10
) -> List[Dict]:
    """
    Prepare sequences from trajectory data with burn-in and overlap
    """
    sequences = []
    unique_patients = np.unique(patient_ids)
    
    for pid in unique_patients:
        patient_mask = patient_ids == pid
        patient_states = states[patient_mask]
        patient_actions = actions[patient_mask]
        patient_rewards = rewards[patient_mask]
        
        traj_len = len(patient_states)
        
        if traj_len < burn_in_length + sequence_length:
            continue
        
        start_idx = 0
        while start_idx + burn_in_length + sequence_length <= traj_len:
            end_idx = start_idx + burn_in_length + sequence_length
            
            seq_states = patient_states[start_idx:end_idx]
            seq_rewards = patient_rewards[start_idx:end_idx]
            
            training_states = seq_states[burn_in_length:]
            training_rewards = seq_rewards[burn_in_length:]
            
            sequences.append({
                'patient_id': pid,
                'training_states': training_states,
                'training_rewards': training_rewards
            })
            
            start_idx += (sequence_length - overlap)
    
    return sequences


def compute_wis_per_patient_sequences(sequences: List[Dict]) -> Dict:
    """Compute WIS estimator for a patient from their sequences"""
    if len(sequences) == 0:
        return {'wis': 0.0, 'n_steps': 0}
    
    all_rewards = []
    for seq in sequences:
        all_rewards.extend(seq['training_rewards'])
    
    if len(all_rewards) == 0:
        return {'wis': 0.0, 'n_steps': 0}
    
    rewards = np.array(all_rewards)
    cumulative_weight = 1.0
    weighted_rewards = []
    weights = []
    
    for reward in rewards:
        weight = 1.0
        cumulative_weight *= weight
        weighted_rewards.append(cumulative_weight * reward)
        weights.append(cumulative_weight)
    
    total_weighted_reward = np.sum(weighted_rewards)
    total_weight = np.sum(weights)
    wis = total_weighted_reward / total_weight if total_weight > 0 else 0.0
    
    return {'wis': wis, 'n_steps': len(rewards)}


def evaluate_wis(model_path: str, vp2_bins: int, reward_type: str = 'simple'):
    """Evaluate LSTM Block Discrete CQL model using WIS estimator"""
    print(f"\n{'='*80}")
    print(f"LSTM Block Discrete CQL WIS Evaluation")
    print(f"Bins: {vp2_bins}, Reward Type: {reward_type.upper()}")
    print(f"Model: {model_path}")
    print(f"{'='*80}\n")
    
    # Load checkpoint
    print("Loading model checkpoint...")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
        sequence_length = config_dict.get('sequence_length', 20)
        burn_in_length = config_dict.get('burn_in_length', 8)
        overlap = config_dict.get('overlap', 10)
        hidden_dim = config_dict.get('hidden_dim', 32)
        lstm_hidden = config_dict.get('lstm_hidden', 32)
        num_lstm_layers = config_dict.get('num_lstm_layers', 2)
    else:
        sequence_length, burn_in_length, overlap = 20, 8, 10
        hidden_dim, lstm_hidden, num_lstm_layers = 32, 32, 2
    
    print(f"Config: seq_len={sequence_length}, burn_in={burn_in_length}, overlap={overlap}")
    
    # Load data
    print("\nLoading data...")
    pipeline = IntegratedDataPipelineV2(model_type='dual', random_seed=42)
    _, _, test_data = pipeline.prepare_data()
    
    states = test_data['states']
    actions = test_data['actions']
    rewards_data = test_data['rewards']
    patient_ids = test_data['patient_ids']
    
    state_dim = states.shape[1]
    total_actions = 2 * vp2_bins
    
    print(f"State dim: {state_dim}, Actions: {total_actions}, Test samples: {len(states)}")
    
    # Prepare sequences
    print(f"\nPreparing sequences...")
    sequences = prepare_sequences(
        states, actions, rewards_data, patient_ids,
        sequence_length, burn_in_length, overlap
    )
    print(f"Created {len(sequences)} sequences")
    
    # Group by patient
    patient_sequences = defaultdict(list)
    for seq in sequences:
        patient_sequences[seq['patient_id']].append(seq)
    
    # Compute WIS
    print(f"\nComputing WIS for {len(patient_sequences)} patients...")
    patient_wis = {}
    for pid, seqs in patient_sequences.items():
        patient_wis[pid] = compute_wis_per_patient_sequences(seqs)
    
    all_wis = [stats['wis'] for stats in patient_wis.values()]
    mean_wis = np.mean(all_wis)
    std_wis = np.std(all_wis)
    median_wis = np.median(all_wis)
    
    print(f"\n{'='*80}")
    print(f"Results: LSTM Block Discrete bins={vp2_bins}, {reward_type.upper()}")
    print(f"{'='*80}")
    print(f"Mean WIS:   {mean_wis:+.4f}")
    print(f"Std WIS:    {std_wis:.4f}")
    print(f"Median WIS: {median_wis:+.4f}")
    print(f"Patients:   {len(patient_wis)}")
    print(f"Sequences:  {len(sequences)}")
    print(f"{'='*80}\n")
    
    # Baseline
    behavior_wis = []
    for pid, seqs in patient_sequences.items():
        all_rewards = []
        for seq in seqs:
            all_rewards.extend(seq['training_rewards'])
        if all_rewards:
            behavior_wis.append(np.mean(all_rewards))
    
    baseline_mean = np.mean(behavior_wis) if behavior_wis else 0.0
    improvement = mean_wis - baseline_mean
    
    print(f"Behavior Policy: {baseline_mean:+.4f}")
    print(f"Learned Policy:  {mean_wis:+.4f}")
    print(f"Improvement:     {improvement:+.4f}\n")
    
    # Save LaTeX
    os.makedirs('latex', exist_ok=True)
    alpha_str = checkpoint['config']['alpha'] if 'config' in checkpoint else '0.0000'
    latex_file = f'latex/lstm_block_discrete_bins{vp2_bins}_alpha{alpha_str}_{reward_type}_wis.tex'
    
    with open(latex_file, 'w') as f:
        f.write(f"% LSTM Block Discrete bins={vp2_bins}, {reward_type.upper()}\n")
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write(f"\\caption{{LSTM Block Discrete CQL (bins={vp2_bins}, {reward_type.upper()}, $\\alpha={alpha_str}$)}}\n")
        f.write("\\begin{tabular}{lc}\n\\hline\nMetric & Value \\\\\n\\hline\n")
        f.write(f"Behavior Policy & ${baseline_mean:+.4f}$ \\\\\n")
        f.write(f"Learned Policy (LSTM-CQL) & ${mean_wis:+.4f}$ \\\\\n")
        f.write(f"Improvement & $\\mathbf{{{improvement:+.4f}}}$ \\\\\n")
        f.write(f"Std Dev & ${std_wis:.4f}$ \\\\\n")
        f.write(f"Patients & {len(patient_wis)} \\\\\n")
        f.write(f"Sequences & {len(sequences)} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")
    
    print(f"Saved: {latex_file}")
    
    return {
        'mean_wis': mean_wis,
        'improvement': improvement,
        'n_patients': len(patient_wis),
        'n_sequences': len(sequences)
    }


def main():
    parser = argparse.ArgumentParser(description='LSTM Block Discrete CQL WIS Evaluation')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--bins', type=int, default=5)
    parser.add_argument('--reward_type', type=str, default='simple', choices=['simple', 'oviss'])
    args = parser.parse_args()
    
    results = evaluate_wis(args.model_path, args.bins, args.reward_type)
    print(f"\n✅ Complete! Improvement: {results['improvement']:+.4f}")


if __name__ == '__main__':
    main()
