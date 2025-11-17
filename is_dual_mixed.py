#!/usr/bin/env python3
"""
WIS/OPE Evaluation for Dual-Mixed CQL
Dual-Mixed: VP1 binary (0 or 1), VP2 continuous (discretized for WIS)
"""

import torch
import numpy as np
import argparse
import sys
import os
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

from integrated_data_pipeline_v2 import IntegratedDataPipelineV2
from run_dualmixed_cql_allalphas import DualMixedCQL

def parse_args():
    parser = argparse.ArgumentParser(description='WIS/OPE for Dual-Mixed CQL')
    parser.add_argument('--alpha', type=float, required=True)
    parser.add_argument('--reward-type', type=str, required=True, choices=['simple', 'oviss'])
    parser.add_argument('--vp2-bins', type=int, default=5, help='Number of bins for VP2 discretization')
    parser.add_argument('--output-dir', type=str, default='latex/')
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*70)
    print(" DUAL-MIXED CQL - WIS/OPE EVALUATION")
    print("="*70)
    print(f"  Alpha: {args.alpha}, Reward: {args.reward_type}, VP2 bins: {args.vp2_bins}")
    
    # Load model
    if args.reward_type == 'simple':
        alpha_str = f"{int(args.alpha*100):02d}"
        if args.alpha == 0.001:
            alpha_str = "001"
        model_path = f'experiment/dual_cql_unified_alpha{alpha_str}_best.pt'
    else:
        model_path = f'experiment/dual_rev_cql_alpha{args.alpha:.4f}_best.pt'
    
    print(f"\nLoading: {model_path}")
    if not os.path.exists(model_path):
        print(f"ERROR: Not found")
        sys.exit(1)
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Infer state_dim from checkpoint (fc1 weight has shape [hidden, state+action])
    fc1_weight = checkpoint['q1_state_dict']['fc1.weight']
    input_dim = fc1_weight.shape[1]  # state_dim + action_dim
    action_dim = 2  # Dual-Mixed uses action_dim=2 (VP1 + VP2)
    state_dim = input_dim - action_dim
    print(f"  Inferred state_dim: {state_dim} (from checkpoint)")
    
    agent = DualMixedCQL(state_dim=state_dim, alpha=args.alpha)
    agent.q1.load_state_dict(checkpoint['q1_state_dict'])
    agent.q2.load_state_dict(checkpoint['q2_state_dict'])
    agent.q1.eval()
    agent.q2.eval()
    agent.q1.to(device)
    agent.q2.to(device)
    print("✅ Model loaded")
    
    # Load data
    pipeline = IntegratedDataPipelineV2(model_type='dual', random_seed=42)
    train_data, _, test_data = pipeline.prepare_data()
    print(f"✅ Data loaded")
    
    # Define VP2 bins for discretization (uniform bins from 0 to max VP2)
    train_vp2 = train_data['actions'][:, 1]
    vp2_min, vp2_max = 0.0, train_vp2.max()
    vp2_bin_edges = np.linspace(vp2_min, vp2_max, args.vp2_bins + 1)
    print(f"  VP2 range: [{vp2_min:.4f}, {vp2_max:.4f}], bins: {args.vp2_bins}")
    
    # Generate actions with VP2 discretized
    def select_actions(states):
        """Generate continuous actions from model, then discretize VP2"""
        with torch.no_grad():
            st = torch.FloatTensor(states).to(device)
            # For binary VP1, evaluate both 0 and 1
            batch_size = len(states)
            actions_continuous = []
            
            for i in range(batch_size):
                s = st[i:i+1]
                best_q = -float('inf')
                best_action = None
                
                # Try VP1=0 and VP1=1 with continuous VP2
                for vp1 in [0.0, 1.0]:
                    # Sample a few VP2 values to find best
                    vp2_samples = np.linspace(vp2_min, vp2_max, 20)
                    for vp2 in vp2_samples:
                        a = torch.FloatTensor([[vp1, vp2]]).to(device)
                        q = min(agent.q1(s, a).item(), agent.q2(s, a).item())
                        if q > best_q:
                            best_q = q
                            best_action = [vp1, vp2]
                
                actions_continuous.append(best_action)
            
            actions_continuous = np.array(actions_continuous)
            
            # Discretize VP2 for behavior policy
            vp1 = actions_continuous[:, 0]
            vp2_continuous = actions_continuous[:, 1]
            vp2_discrete = np.digitize(vp2_continuous, vp2_bin_edges[1:-1])  # bins 0 to vp2_bins-1
            vp2_discrete = np.clip(vp2_discrete, 0, args.vp2_bins - 1)  # ensure valid bin indices
            
            return vp1, vp2_discrete, actions_continuous
    
    print("\n⏳ Generating model actions (this may take a moment)...")
    train_vp1, train_vp2_discrete, _ = select_actions(train_data['states'])
    test_vp1, test_vp2_discrete, test_actions_continuous = select_actions(test_data['states'])
    print("✅ Actions generated")
    
    # Discretize clinician VP2 actions
    train_clin_vp1 = train_data['actions'][:, 0]
    train_clin_vp2 = train_data['actions'][:, 1]
    train_clin_vp2_discrete = np.digitize(train_clin_vp2, vp2_bin_edges[1:-1])
    train_clin_vp2_discrete = np.clip(train_clin_vp2_discrete, 0, args.vp2_bins - 1)
    
    test_clin_vp1 = test_data['actions'][:, 0]
    test_clin_vp2 = test_data['actions'][:, 1]
    test_clin_vp2_discrete = np.digitize(test_clin_vp2, vp2_bin_edges[1:-1])
    test_clin_vp2_discrete = np.clip(test_clin_vp2_discrete, 0, args.vp2_bins - 1)
    
    # Train behavior policies (separate for VP1 and VP2)
    print("\n⏳ Training behavior policies...")
    
    # Debug: check VP2 classes
    print(f"  Train VP2 classes: {np.unique(train_vp2_discrete)}")
    print(f"  Train Clin VP2 classes: {np.unique(train_clin_vp2_discrete)}")
    print(f"  Test VP2 classes: {np.unique(test_vp2_discrete)}")
    print(f"  Test Clin VP2 classes: {np.unique(test_clin_vp2_discrete)}")
    
    # VP1 behavior policy (binary)
    clf_vp1_model = LogisticRegression(max_iter=1000, random_state=42)
    clf_vp1_model.fit(train_data['states'], train_vp1)
    
    clf_vp1_clin = LogisticRegression(max_iter=1000, random_state=42)
    clf_vp1_clin.fit(train_data['states'], train_clin_vp1)
    
    # VP2 behavior policy (discrete) - ensure all classes 0 to vp2_bins-1 are known
    # Add dummy samples for missing classes to ensure classifier knows about all bins
    all_classes = np.arange(args.vp2_bins)
    missing_model = np.setdiff1d(all_classes, np.unique(train_vp2_discrete))
    missing_clin = np.setdiff1d(all_classes, np.unique(train_clin_vp2_discrete))
    
    train_states_vp2_model = train_data['states']
    train_vp2_discrete_full = train_vp2_discrete.copy()
    if len(missing_model) > 0:
        # Add one dummy sample per missing class (use mean state)
        dummy_state = train_data['states'].mean(axis=0, keepdims=True)
        train_states_vp2_model = np.vstack([train_states_vp2_model] + [dummy_state] * len(missing_model))
        train_vp2_discrete_full = np.concatenate([train_vp2_discrete_full, missing_model])
    
    train_states_vp2_clin = train_data['states']
    train_clin_vp2_discrete_full = train_clin_vp2_discrete.copy()
    if len(missing_clin) > 0:
        dummy_state = train_data['states'].mean(axis=0, keepdims=True)
        train_states_vp2_clin = np.vstack([train_states_vp2_clin] + [dummy_state] * len(missing_clin))
        train_clin_vp2_discrete_full = np.concatenate([train_clin_vp2_discrete_full, missing_clin])
    
    clf_vp2_model = LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial')
    clf_vp2_model.fit(train_states_vp2_model, train_vp2_discrete_full)
    
    clf_vp2_clin = LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial')
    clf_vp2_clin.fit(train_states_vp2_clin, train_clin_vp2_discrete_full)
    print("✅ Behavior policies trained")
    print(f"  VP2 model classes: {clf_vp2_model.classes_}")
    print(f"  VP2 clin classes: {clf_vp2_clin.classes_}")
    
    # Compute IS weights (product of VP1 and VP2 probabilities)
    test_probs_vp1_model = clf_vp1_model.predict_proba(test_data['states'])
    test_probs_vp1_clin = clf_vp1_clin.predict_proba(test_data['states'])
    
    test_probs_vp2_model = clf_vp2_model.predict_proba(test_data['states'])
    test_probs_vp2_clin = clf_vp2_clin.predict_proba(test_data['states'])
    
    # Get probabilities for clinician actions
    test_clin_vp1_int = test_clin_vp1.astype(int)
    test_clin_vp2_int = test_clin_vp2_discrete.astype(int)
    
    prob_vp1_model = test_probs_vp1_model[np.arange(len(test_clin_vp1_int)), test_clin_vp1_int]
    prob_vp1_clin = test_probs_vp1_clin[np.arange(len(test_clin_vp1_int)), test_clin_vp1_int]
    
    prob_vp2_model = test_probs_vp2_model[np.arange(len(test_clin_vp2_int)), test_clin_vp2_int]
    prob_vp2_clin = test_probs_vp2_clin[np.arange(len(test_clin_vp2_int)), test_clin_vp2_int]
    
    # Combined IS weights (independent actions assumption)
    is_weights = (prob_vp1_model * prob_vp2_model) / ((prob_vp1_clin * prob_vp2_clin) + 1e-10)
    print(f"✅ IS weights computed (mean={is_weights.mean():.4f})")
    
    # Compute WIS per patient
    test_rewards = test_data['rewards']
    test_pids = test_data['patient_ids']
    unique_pids = np.unique(test_pids)
    
    wis_rewards = []
    clin_rewards = []
    for pid in unique_pids:
        mask = test_pids == pid
        w = is_weights[mask]
        r = test_rewards[mask]
        wis_rewards.append((w * r).sum() / w.sum() if w.sum() > 0 else 0)
        clin_rewards.append(r.sum())
    
    wis_reward = np.mean(wis_rewards)
    clin_reward = np.mean(clin_rewards)
    improvement = wis_reward - clin_reward
    
    print(f"\nResults:")
    print(f"  Clinician: {clin_reward:.4f}")
    print(f"  Model WIS: {wis_reward:.4f}")
    print(f"  Improvement: {improvement:.4f}")
    
    # Save LaTeX
    os.makedirs(args.output_dir, exist_ok=True)
    latex = f"""\\begin{{table}}[h]
\\centering
\\caption{{Dual-Mixed CQL ($\\alpha={args.alpha:.4f}$, {args.reward_type}, VP2 bins={args.vp2_bins})}}
\\begin{{tabular}}{{lccc}}
\\hline
\\textbf{{Policy}} & \\textbf{{WIS Reward}} & \\textbf{{Improvement}} & \\textbf{{95\\% CI}} \\\\
\\hline
Clinician & {clin_reward:.4f} & -- & -- \\\\
Dual-Mixed & {wis_reward:.4f} & \\textbf{{{improvement:.4f}}} & [TBD] \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
"""
    
    outfile = os.path.join(args.output_dir, f'dual_mixed_alpha{args.alpha:.4f}_{args.reward_type}_wis.tex')
    with open(outfile, 'w') as f:
        f.write(latex)
    
    print(f"✅ Saved to: {outfile}")
    print("="*70)

if __name__ == "__main__":
    main()
