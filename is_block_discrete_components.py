"""
Component-wise Weighted Importance Sampling (WIS) Evaluation

Computes WIS for each clinician reward component separately:
- Base (Survival)
- Lactate improvement
- Blood pressure (MBP) improvement
- SOFA score decrease
- Norepinephrine decrease
- Mortality penalty
- Total (sum of all components)

Usage:
    python is_block_discrete_components.py --model_path <path_to_model.pt> [--vp2_bins 5]

Requires: Same model checkpoint as is_block_discrete.py
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import os
import argparse
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# Import project modules
from integrated_data_pipeline_v3 import IntegratedDataPipelineV3, compute_outcome_reward_components
from data_loader import DataLoader, DataSplitter
import data_config as config

# ============================================================================
# Configuration
# ============================================================================

RANDOM_SEED = 42
STATE_DIM = 17
DEFAULT_VP2_BINS = 5

# IRL model paths (for reference - these were used to train the Q-learning models)
IRL_MODEL_PATHS = {
    'unet_tanh': 'experiments/unet_reward_gen_h64_tanh/model_epoch_100.pt',
    'semi_sup_unet': 'experiments/semi_supervised_unet_64/model_epoch_100.pt',
    'maxent_irl': 'experiment/irl/maxent_reward_model.pt',
    'gcl': '/scratch/zouwil/code/vaso_rl/experiment/irl/gcl_tau0.005_lr0.001_clr0.01_cost_model.pt',
    'iq_learn': '/scratch/zouwil/code/vaso_rl/experiment/irl/iq_learn_temp0.001_tau0.005_lr0.001_divjs_q_model.pt',
}

# Q-learning model paths (trained with each IRL reward)
# Format: experiment/ql/{algorithm}_alpha0.0000_bins{n}_best.pt
QL_MODEL_PATHS = {
    'unet_tanh': 'experiment/ql/unet_alpha0.0000_bins5_best.pt',
    'semi_sup_unet': 'experiment/ql/semi_supervised_unet_alpha0.0000_bins5_best.pt',
    'maxent_irl': 'experiment/ql/maxent_alpha0.0000_bins5_best.pt',
    'gcl': 'experiment/ql/gcl_tau0.005_lr0.001_clr0.01_alpha0.0000_bins5_best.pt',
    'iq_learn': 'experiment/ql/iq_learn_temp0.001_tau0.005_lr0.001_divjs_alpha0.0000_bins5_best.pt',
}

# Model display names for tables
MODEL_DISPLAY_NAMES = {
    'unet_tanh': 'U-Net (Tanh)',
    'semi_sup_unet': 'Semi-Sup U-Net',
    'maxent_irl': 'MaxEnt IRL',
    'gcl': 'GCL',
    'iq_learn': 'IQ-Learn',
}

# Order for display
MODEL_ORDER = ['unet_tanh', 'semi_sup_unet', 'maxent_irl', 'gcl', 'iq_learn']

# Reward components to analyze
REWARD_COMPONENTS = ['base', 'lactate', 'mbp', 'sofa', 'norepinephrine', 'mortality', 'total']

COMPONENT_DISPLAY_NAMES = {
    'base': 'Base (Survival)',
    'lactate': 'Lactate Improvement',
    'mbp': 'Blood Pressure (MBP)',
    'sofa': 'SOFA Score Decrease',
    'norepinephrine': 'Norepinephrine Decrease',
    'mortality': 'Mortality Penalty',
    'total': 'Total Reward',
}


# ============================================================================
# Argument Parsing
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Component-wise WIS evaluation for Block Discrete CQL')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained CQL model checkpoint')
    parser.add_argument('--vp2_bins', type=int, default=5,
                       help='Number of bins for VP2 discretization (default: 5)')
    parser.add_argument('--eval_set', type=str, default='test', choices=['val', 'test'],
                       help='Which data split to evaluate on (default: test)')
    parser.add_argument('--output_dir', type=str, default='irl_analysis',
                       help='Directory to save results (default: irl_analysis)')
    parser.add_argument('--combined_or_train_data_path', type=str, default=None,
                       help='Path to training dataset. If eval_data_path is also provided, '
                            'all patients from this dataset are used for training. '
                            'If eval_data_path is None, this dataset is split into train/val/test.')
    parser.add_argument('--eval_data_path', type=str, default=None,
                       help='Path to evaluation dataset (for val/test). If provided, enables '
                            'dual-dataset mode where this dataset is split 50/50 into val/test.')
    parser.add_argument('--use_lstm', action='store_true',
                       help='Use LSTM Q-network instead of standard feedforward network')
    return parser.parse_args()


# ============================================================================
# Data Loading with Component-wise Rewards
# ============================================================================

def load_component_rewards(eval_set='test', random_seed=42,
                           combined_or_train_data_path=None, eval_data_path=None):
    """
    Load test/val data and compute component-wise rewards for each transition.

    Args:
        eval_set: Which data split to evaluate on ('test' or 'val')
        random_seed: Random seed for reproducibility
        combined_or_train_data_path: Path to training dataset. If eval_data_path is also
            provided, all patients are used for training. Otherwise split into train/val/test.
        eval_data_path: Path to evaluation dataset. If provided, enables dual-dataset mode
            where this dataset is split 50/50 into val/test.

    Returns:
        components_by_transition: Dict[component_name -> np.array of shape (n_transitions,)]
        eval_data: The evaluation data dict from pipeline
        train_data: The training data dict (for behavior policy training)
    """
    print("Loading data and computing component-wise rewards...")

    # Load data using pipeline (for states, actions, patient_ids)
    pipeline = IntegratedDataPipelineV3(
        model_type='dual',
        reward_source='manual',
        random_seed=random_seed,
        combined_or_train_data_path=combined_or_train_data_path,
        eval_data_path=eval_data_path
    )
    train_data, val_data, test_data = pipeline.prepare_data()

    eval_data = test_data if eval_set == 'test' else val_data

    # Load raw data for component computation
    # Use eval_data_path for component computation if in dual-dataset mode
    data_path_for_components = eval_data_path if eval_data_path else combined_or_train_data_path
    loader = DataLoader(verbose=False, random_seed=random_seed, data_path=data_path_for_components)
    splitter = DataSplitter(random_seed=random_seed)

    full_data = loader.load_data()
    patient_ids = loader.get_patient_ids()

    # In dual-dataset mode, eval patients come from eval_data_path (split 50/50)
    if eval_data_path:
        # Split eval dataset 50/50 into val/test
        n_eval_patients = len(patient_ids)
        n_val = n_eval_patients // 2
        val_patients = patient_ids[:n_val]
        test_patients = patient_ids[n_val:]
        train_patients = []  # Not used for component computation
    else:
        train_patients, val_patients, test_patients = splitter.split_patients(patient_ids)

    loader.encode_categorical_features()
    state_features = config.DUAL_STATE_FEATURES

    eval_patients = test_patients if eval_set == 'test' else val_patients

    # Initialize component arrays
    # We need to match the order of transitions in eval_data
    components_by_transition = {comp: [] for comp in REWARD_COMPONENTS}

    # Build mapping from (patient_id, timestep) to transition index
    # eval_data has patient_groups: {patient_id: (start_idx, end_idx)}

    for pid in eval_patients:
        patient_data = loader.get_patient_data(pid)

        if len(patient_data) < 2:
            continue

        # Extract states and actions
        states = patient_data[state_features].values
        vp1 = patient_data['action_vaso'].values.astype(float)
        vp2 = patient_data['norepinephrine'].values.astype(float)
        actions = np.column_stack([vp1, vp2])

        mortality = int(patient_data[config.DEATH_COL].iloc[-1])

        # Compute components for each transition
        for t in range(len(states) - 1):
            is_terminal = (t == len(states) - 2)

            comp_values = compute_outcome_reward_components(
                states, actions, t, is_terminal, mortality, state_features
            )

            for comp in REWARD_COMPONENTS:
                components_by_transition[comp].append(comp_values[comp])

    # Convert to numpy arrays
    for comp in REWARD_COMPONENTS:
        components_by_transition[comp] = np.array(components_by_transition[comp])

    print(f"  Loaded {len(components_by_transition['total'])} transitions with component rewards")
    print(f"  Components: {list(REWARD_COMPONENTS)}")

    # Verify alignment with eval_data
    n_eval_transitions = len(eval_data['rewards'])
    n_component_transitions = len(components_by_transition['total'])

    if n_eval_transitions != n_component_transitions:
        print(f"  WARNING: Transition count mismatch!")
        print(f"    eval_data: {n_eval_transitions}")
        print(f"    components: {n_component_transitions}")

    return components_by_transition, eval_data, train_data


# ============================================================================
# Model Loading and Action Selection
# ============================================================================

def load_q_networks(model_path, state_dim, n_bins, device, use_lstm=False):
    """Load trained Q-networks from checkpoint.

    Args:
        model_path: Path to model checkpoint
        state_dim: State dimension
        n_bins: Number of VP2 bins
        device: Torch device
        use_lstm: Whether to load LSTM Q-networks (default: False)

    Returns:
        q1_network, q2_network: Loaded Q-networks
    """
    n_actions = 2 * n_bins  # VP1 (2) x VP2 (n_bins)

    if use_lstm:
        from lstm_block_discrete_cql_network import LSTMDiscreteQNetwork

        q1_network = LSTMDiscreteQNetwork(state_dim=state_dim, num_actions=n_actions).to(device)
        q2_network = LSTMDiscreteQNetwork(state_dim=state_dim, num_actions=n_actions).to(device)
    else:
        from run_block_discrete_cql_allalphas import DualBlockDiscreteQNetwork

        q1_network = DualBlockDiscreteQNetwork(state_dim=state_dim, vp2_bins=n_bins).to(device)
        q2_network = DualBlockDiscreteQNetwork(state_dim=state_dim, vp2_bins=n_bins).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    q1_network.load_state_dict(checkpoint['q1_state_dict'])
    q2_network.load_state_dict(checkpoint['q2_state_dict'])
    q1_network.eval()
    q2_network.eval()

    return q1_network, q2_network


def select_action_batch_discrete(states, q1_net, q2_net, vp2_bins, state_dim, device):
    """Select best actions for a batch of states using min(Q1, Q2)."""
    with torch.no_grad():
        if states.ndim == 1:
            states = states.reshape(1, -1)

        batch_size = states.shape[0]
        state_tensor = torch.FloatTensor(states).to(device)

        total_actions = 2 * vp2_bins
        all_actions = torch.arange(total_actions).to(device)
        all_actions = all_actions.unsqueeze(0).expand(batch_size, -1)

        state_expanded = state_tensor.unsqueeze(1).expand(-1, total_actions, -1)
        state_expanded = state_expanded.reshape(-1, state_dim)
        actions_flat = all_actions.reshape(-1)

        q1_values = q1_net(state_expanded, actions_flat).reshape(batch_size, total_actions)
        q2_values = q2_net(state_expanded, actions_flat).reshape(batch_size, total_actions)
        q_values = torch.min(q1_values, q2_values)

        best_action_indices = q_values.argmax(dim=1).cpu().numpy()

        return best_action_indices


def continuous_to_discrete_action(actions, vp2_edges, vp2_bins):
    """Convert continuous actions [vp1, vp2] to discrete action indices (0-9)."""
    vp1 = actions[:, 0].astype(int)
    vp2 = actions[:, 1].clip(0, 0.5)

    vp2_bins_idx = np.digitize(vp2, vp2_edges) - 1
    vp2_bins_idx = np.clip(vp2_bins_idx, 0, len(vp2_edges) - 2)

    action_indices = vp1 * vp2_bins + vp2_bins_idx
    return action_indices


# ============================================================================
# Behavior Policy Training
# ============================================================================

def get_full_proba(clf, X, n_classes):
    """Get probability predictions ensuring output has n_classes columns."""
    proba = clf.predict_proba(X)
    if proba.shape[1] == n_classes:
        return proba

    full_proba = np.zeros((X.shape[0], n_classes))
    for i, c in enumerate(clf.classes_):
        full_proba[:, int(c)] = proba[:, i]

    missing_classes = set(range(n_classes)) - set(clf.classes_.astype(int))
    if missing_classes:
        for c in missing_classes:
            full_proba[:, c] = 1e-10
        full_proba = full_proba / full_proba.sum(axis=1, keepdims=True)

    return full_proba


def train_behavior_policies(train_states, train_model_actions, train_clinician_actions, n_actions):
    """Train behavior policy classifiers for model and clinician."""
    print("\nTraining behavior policy classifiers...")

    # Model policy classifier
    clf_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42)
    clf_model.fit(train_states, train_model_actions)

    # Clinician policy classifier
    clf_clinician = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42)
    clf_clinician.fit(train_states, train_clinician_actions)

    return clf_model, clf_clinician


def compute_is_weights(eval_states, eval_clinician_actions, clf_model, clf_clinician, n_actions):
    """Compute importance sampling weights."""
    eps = 1e-10

    # Get probabilities
    eval_probs_model = get_full_proba(clf_model, eval_states, n_actions)
    eval_prob_model = eval_probs_model[np.arange(len(eval_clinician_actions)), eval_clinician_actions.astype(int)]

    eval_probs_clinician = get_full_proba(clf_clinician, eval_states, n_actions)
    eval_prob_clinician = eval_probs_clinician[np.arange(len(eval_clinician_actions)), eval_clinician_actions.astype(int)]

    # Compute IS weights
    is_weight = eval_prob_model / (eval_prob_clinician + eps)

    # Clip weights
    isw_ci_lower = np.percentile(is_weight, 0.5) #2.5
    isw_ci_upper = np.percentile(is_weight, 99.5) #97.5
    is_weight = np.clip(is_weight, a_min=isw_ci_lower, a_max=isw_ci_upper)

    return is_weight, isw_ci_lower, isw_ci_upper


# ============================================================================
# Component-wise WIS Computation
# ============================================================================

def compute_wis_transition_level(is_weight, rewards):
    """Compute WIS at transition level: sum(w*r) / sum(w)."""
    sum_weighted_rewards = (is_weight * rewards).sum()
    sum_weights = is_weight.sum()
    wis = sum_weighted_rewards / sum_weights if sum_weights > 0 else 0.0
    clinician_mean = rewards.mean()
    return wis, clinician_mean


def compute_wis_trajectory_level(is_weight, rewards, patient_ids, isw_ci_diff_lower, isw_ci_diff_upper, n_bootstrap=1000):
    """
    Compute WIS at trajectory level with bootstrapped 95% CI.

    Returns:
        wis: WIS estimate
        clinician_mean: Mean clinician reward
        ci_lower: 95% CI lower bound for difference
        ci_upper: 95% CI upper bound for difference
    """
    unique_patients = np.unique(patient_ids)

    weights_per_traj = []
    total_rewards_per_traj = []
    weighted_rewards_per_traj = []

    for patient_id in unique_patients:
        mask = patient_ids == patient_id
        patient_weights = is_weight[mask]
        patient_rewards = rewards[mask]
        weights_per_traj.append(patient_weights.mean())

        patient_weights = np.cumprod(patient_weights)
        patient_weights = np.clip(patient_weights, a_min = isw_ci_diff_lower, a_max = isw_ci_diff_upper)

        # WIS per trajectory
        if patient_weights.sum() > 0:
            est_total = (patient_weights * patient_rewards).sum() / patient_weights.sum() * len(patient_weights)
        else:
            est_total = 0.0

        
        total_rewards_per_traj.append(patient_rewards.sum())
        weighted_rewards_per_traj.append(est_total)

    weights_per_traj = np.array(weights_per_traj)
    total_rewards_per_traj = np.array(total_rewards_per_traj)
    weighted_rewards_per_traj = np.array(weighted_rewards_per_traj)

    # WIS trajectory level
    if weights_per_traj.sum() > 0:
        wis = (weights_per_traj * weighted_rewards_per_traj).sum() / weights_per_traj.sum()
    else:
        wis = 0.0

    clinician_mean = total_rewards_per_traj.mean()

    # Bootstrap for 95% CI
    np.random.seed(42)
    bootstrap_diffs = []

    for _ in range(n_bootstrap):
        idx = np.random.choice(len(total_rewards_per_traj), size=len(total_rewards_per_traj), replace=True)

        boot_weights = weights_per_traj[idx]
        boot_rewards = total_rewards_per_traj[idx]
        boot_weighted = weighted_rewards_per_traj[idx]

        if boot_weights.sum() > 0:
            boot_wis = (boot_weights * boot_weighted).sum() / boot_weights.sum()
        else:
            boot_wis = 0.0

        boot_clinician = boot_rewards.mean()
        bootstrap_diffs.append(boot_wis - boot_clinician)

    ci_lower = np.percentile(bootstrap_diffs, 2.5)
    ci_upper = np.percentile(bootstrap_diffs, 97.5)

    return wis, clinician_mean, ci_lower, ci_upper


def compute_wis_trajectory_level_sparse(is_weight, rewards, patient_ids, n_bootstrap=1000):
    """
    Compute WIS at trajectory level for SPARSE rewards (like mortality).

    Unlike regular trajectory WIS, this does NOT apply per-transition weights
    within each trajectory. Instead:
    - Sum the sparse reward per trajectory (raw, unweighted)
    - Weight each trajectory's total by the trajectory-level IS weight

    This is appropriate for sparse terminal rewards like mortality where
    the reward only appears once per trajectory.

    Returns:
        wis: WIS estimate
        clinician_mean: Mean clinician reward per trajectory
        ci_lower: 95% CI lower bound for difference
        ci_upper: 95% CI upper bound for difference
    """
    unique_patients = np.unique(patient_ids)

    weights_per_traj = []
    total_rewards_per_traj = []

    for patient_id in unique_patients:
        mask = patient_ids == patient_id
        patient_weights = is_weight[mask]
        patient_rewards = rewards[mask]

        # Trajectory weight = mean of transition weights (same as before)
        weights_per_traj.append(patient_weights.mean())
        # Raw sum of sparse reward (no per-transition weighting)
        total_rewards_per_traj.append(patient_rewards.sum())

    weights_per_traj = np.array(weights_per_traj)
    total_rewards_per_traj = np.array(total_rewards_per_traj)

    # WIS: weight each trajectory's total reward by trajectory weight
    if weights_per_traj.sum() > 0:
        wis = (weights_per_traj * total_rewards_per_traj).sum() / weights_per_traj.sum()
    else:
        wis = 0.0

    clinician_mean = total_rewards_per_traj.mean()

    # Bootstrap for 95% CI
    np.random.seed(42)
    bootstrap_diffs = []

    for _ in range(n_bootstrap):
        idx = np.random.choice(len(total_rewards_per_traj), size=len(total_rewards_per_traj), replace=True)

        boot_weights = weights_per_traj[idx]
        boot_rewards = total_rewards_per_traj[idx]

        if boot_weights.sum() > 0:
            boot_wis = (boot_weights * boot_rewards).sum() / boot_weights.sum()
        else:
            boot_wis = 0.0

        boot_clinician = boot_rewards.mean()
        bootstrap_diffs.append(boot_wis - boot_clinician)

    bootstrap_diffs = np.array(bootstrap_diffs)
    ci_lower = np.percentile(bootstrap_diffs, 2.5)
    ci_upper = np.percentile(bootstrap_diffs, 97.5)

    return wis, clinician_mean, ci_lower, ci_upper


# ============================================================================
# Table Generation
# ============================================================================

def generate_component_wis_table(results, level='transition'):
    """Generate ASCII table for component-wise WIS results."""
    header = (
        f"{'Component':<24} | "
        f"{'Clinician':>12} {'Model (WIS)':>12} {'Difference':>12}"
    )
    if level == 'trajectory':
        header += f" {'95% CI':>20}"

    separator = "-" * len(header)

    lines = [
        "=" * len(header),
        f"COMPONENT-WISE WIS ({level.upper()} LEVEL)",
        "=" * len(header),
        header,
        separator,
    ]

    for comp in REWARD_COMPONENTS:
        display_name = COMPONENT_DISPLAY_NAMES.get(comp, comp)
        r = results.get(comp, {})

        if 'error' in r:
            lines.append(f"{display_name:<24} | {'N/A':>12} {'N/A':>12} {'N/A':>12}")
        else:
            clinician = r.get('clinician', 0)
            wis = r.get('wis', 0)

            # Handle NaN for sparse rewards (e.g., mortality at transition level)
            if np.isnan(wis) if isinstance(wis, float) else False:
                row = f"{display_name:<24} | {clinician:>12.4f} {'N/A':>12} {'N/A':>12}"
            else:
                diff = wis - clinician
                row = f"{display_name:<24} | {clinician:>12.4f} {wis:>12.4f} {diff:>12.4f}"

            if level == 'trajectory' and 'ci_lower' in r:
                ci_str = f"[{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]"
                row += f" {ci_str:>20}"

            lines.append(row)

    lines.append(separator)

    return "\n".join(lines)


def generate_component_wis_latex(results_trans, results_traj, output_path):
    """Generate LaTeX table for component-wise WIS results."""
    latex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Component-wise Weighted Importance Sampling (WIS) Results}",
        r"\label{tab:component_wis}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{l|ccc|cccc}",
        r"\hline",
        r"\multirow{2}{*}{\textbf{Component}} & \multicolumn{3}{c|}{\textbf{Transition-Level}} & \multicolumn{4}{c}{\textbf{Trajectory-Level}} \\",
        r" & Clinician & Model & Diff & Clinician & Model & Diff & 95\% CI \\",
        r"\hline",
    ]

    for comp in REWARD_COMPONENTS:
        display_name = COMPONENT_DISPLAY_NAMES.get(comp, comp)

        rt = results_trans.get(comp, {})
        rj = results_traj.get(comp, {})

        if 'error' in rt:
            trans_vals = ['--', '--', '--']
        else:
            wis_trans = rt.get('wis', 0)
            # Handle NaN for sparse rewards (e.g., mortality)
            if np.isnan(wis_trans) if isinstance(wis_trans, float) else False:
                trans_vals = [
                    f"{rt.get('clinician', 0):.3f}",
                    'N/A',
                    'N/A',
                ]
            else:
                trans_vals = [
                    f"{rt.get('clinician', 0):.3f}",
                    f"{wis_trans:.3f}",
                    f"{wis_trans - rt.get('clinician', 0):.3f}",
                ]

        if 'error' in rj:
            traj_vals = ['--', '--', '--', '--']
        else:
            diff = rj.get('wis', 0) - rj.get('clinician', 0)
            ci_str = f"[{rj.get('ci_lower', 0):.2f}, {rj.get('ci_upper', 0):.2f}]"
            traj_vals = [
                f"{rj.get('clinician', 0):.3f}",
                f"{rj.get('wis', 0):.3f}",
                f"{diff:.3f}",
                ci_str,
            ]

        row = f"{display_name} & {' & '.join(trans_vals)} & {' & '.join(traj_vals)} \\\\"
        latex_lines.append(row)

    latex_lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"}",
        r"\vspace{2mm}",
        r"",
        r"\footnotesize",
        r"\textit{Diff = Model (WIS) - Clinician. Positive values indicate model improvement.}",
        r"\end{table}",
    ])

    with open(output_path, 'w') as f:
        f.write("\n".join(latex_lines))

    print(f"LaTeX table saved to: {output_path}")


# ============================================================================
# Main Function
# ============================================================================

def run_component_wis_analysis(args):
    """Run component-wise WIS analysis."""

    print("=" * 70)
    print("COMPONENT-WISE WIS ANALYSIS")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print(f"Model: {args.model_path}")
    print(f"VP2 bins: {args.vp2_bins}")
    print(f"Eval set: {args.eval_set}")
    if args.eval_data_path:
        print(f"Dataset mode: DUAL-DATASET")
        print(f"  Train data: {args.combined_or_train_data_path or 'default'}")
        print(f"  Eval data:  {args.eval_data_path}")
    else:
        print(f"Dataset mode: SINGLE-DATASET")
        print(f"  Data path: {args.combined_or_train_data_path or 'default'}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    n_bins = args.vp2_bins
    n_actions = 2 * n_bins

    # =========================================================================
    # Load data with component-wise rewards
    # =========================================================================
    components_by_transition, eval_data, train_data = load_component_rewards(
        eval_set=args.eval_set,
        random_seed=RANDOM_SEED,
        combined_or_train_data_path=args.combined_or_train_data_path,
        eval_data_path=args.eval_data_path
    )

    # =========================================================================
    # Load Q-networks
    # =========================================================================
    print("\nLoading Q-networks...")
    q1_network, q2_network = load_q_networks(args.model_path, STATE_DIM, n_bins, device)

    # =========================================================================
    # Generate actions
    # =========================================================================
    print("\nGenerating model actions...")
    vp2_bin_edges = np.linspace(0, 0.5, n_bins + 1)

    train_model_actions = select_action_batch_discrete(
        train_data['states'], q1_network, q2_network, n_bins, STATE_DIM, device
    )
    eval_model_actions = select_action_batch_discrete(
        eval_data['states'], q1_network, q2_network, n_bins, STATE_DIM, device
    )

    train_clinician_actions = continuous_to_discrete_action(train_data['actions'], vp2_bin_edges, n_bins)
    eval_clinician_actions = continuous_to_discrete_action(eval_data['actions'], vp2_bin_edges, n_bins)

    # =========================================================================
    # Train behavior policies and compute IS weights
    # =========================================================================
    clf_model, clf_clinician = train_behavior_policies(
        train_data['states'], train_model_actions, train_clinician_actions, n_actions
    )

    is_weight, isw_ci_lower, isw_ci_upper = compute_is_weights(
        eval_data['states'], eval_clinician_actions, clf_model, clf_clinician, n_actions
    )

    print(f"\nIS weight statistics:")
    print(f"  Mean: {is_weight.mean():.4f}, Std: {is_weight.std():.4f}")

    # =========================================================================
    # Compute component-wise WIS
    # =========================================================================
    print("\n" + "=" * 70)
    print("COMPUTING COMPONENT-WISE WIS")
    print("=" * 70)

    eval_patient_ids = eval_data['patient_ids']

    results_transition = {}
    results_trajectory = {}

    for comp in REWARD_COMPONENTS:
        print(f"\nProcessing: {COMPONENT_DISPLAY_NAMES[comp]}...")

        comp_rewards = components_by_transition[comp]

        # Handle potential length mismatch
        min_len = min(len(comp_rewards), len(is_weight))
        comp_rewards_aligned = comp_rewards[:min_len]
        is_weight_aligned = is_weight[:min_len]
        patient_ids_aligned = eval_patient_ids[:min_len]
        
        """
        # Special handling for mortality (sparse terminal reward)
        if comp == 'mortality':
            # Transition-level: Not applicable for sparse rewards
            # Report raw clinician mean only (no WIS)
            results_transition[comp] = {
                'wis': np.nan,  # Not applicable
                'clinician': comp_rewards_aligned.mean(),
                'note': 'Transition-level WIS not applicable for sparse terminal reward'
            }

            # Trajectory-level: Use sparse WIS (no per-transition weighting)
            comp_rewards_offset = (comp_rewards_aligned + 10) / 5.0
            wis_traj, clin_traj, ci_lower, ci_upper = compute_wis_trajectory_level_sparse(
                is_weight_aligned, comp_rewards_offset, patient_ids_aligned
            )
            results_trajectory[comp] = {
                'wis': wis_traj,
                'clinician': clin_traj,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
            }

            print(f"  Transition: N/A (sparse reward)")
            print(f"  Trajectory (sparse): Clinician={clin_traj:.4f}, WIS={wis_traj:.4f}, Diff={wis_traj-clin_traj:.4f}")
        else:
        """
        
        if comp == 'mortality':
            # Per-transition credit assignment for mortality
            # Assign K/L to each transition where K = +10 (survived) or -10 (died)
            comp_rewards_credit = np.zeros_like(comp_rewards_aligned)

            unique_patients = np.unique(patient_ids_aligned)
            for pid in unique_patients:
                mask = patient_ids_aligned == pid
                traj_len = mask.sum()  # L = trajectory length

                # Determine mortality outcome from raw component values
                # Raw mortality component: -10 at terminal state if died, 0 otherwise
                # So sum < 0 means patient died
                patient_mortality_sum = comp_rewards_aligned[mask].sum()
                if patient_mortality_sum < 0:  # patient died
                    K = -10
                else:  # patient survived
                    K = 10

                # Assign K/L to each transition
                comp_rewards_credit[mask] = K / traj_len

            comp_rewards_aligned = comp_rewards_credit

        # Standard handling for dense rewards
        # Transition-level WIS
        wis_trans, clin_trans = compute_wis_transition_level(is_weight_aligned, comp_rewards_aligned)
        results_transition[comp] = {
            'wis': wis_trans,
            'clinician': clin_trans,
        }

        # Trajectory-level WIS
        wis_traj, clin_traj, ci_lower, ci_upper = compute_wis_trajectory_level(
            is_weight_aligned, comp_rewards_aligned, patient_ids_aligned, isw_ci_lower, isw_ci_upper,
        )
        results_trajectory[comp] = {
            'wis': wis_traj,
            'clinician': clin_traj,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
        }

        print(f"  Transition: Clinician={clin_trans:.4f}, WIS={wis_trans:.4f}, Diff={wis_trans-clin_trans:.4f}")
        print(f"  Trajectory: Clinician={clin_traj:.4f}, WIS={wis_traj:.4f}, Diff={wis_traj-clin_traj:.4f}")

    # =========================================================================
    # Print tables
    # =========================================================================
    print("\n")
    print(generate_component_wis_table(results_transition, level='transition'))
    print("\n")
    print(generate_component_wis_table(results_trajectory, level='trajectory'))

    # =========================================================================
    # Save results
    # =========================================================================
    os.makedirs(args.output_dir, exist_ok=True)

    # Save LaTeX table
    latex_path = os.path.join(args.output_dir, 'component_wis_results.tex')
    generate_component_wis_latex(results_transition, results_trajectory, latex_path)

    # Save pickle
    import pickle
    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'model_path': args.model_path,
            'vp2_bins': args.vp2_bins,
            'eval_set': args.eval_set,
            'combined_or_train_data_path': args.combined_or_train_data_path,
            'eval_data_path': args.eval_data_path,
        },
        'transition_level': results_transition,
        'trajectory_level': results_trajectory,
    }

    pickle_path = os.path.join(args.output_dir, 'component_wis_results.pkl')
    with open(pickle_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to: {pickle_path}")

    return results


# ============================================================================
# Multi-Model Table Generation
# ============================================================================

def generate_all_models_component_table(all_results, level='trajectory'):
    """
    Generate ASCII table for all models and all components.

    Args:
        all_results: Dict[model_name -> {'transition_level': {...}, 'trajectory_level': {...}}]
        level: 'transition' or 'trajectory'
    """
    # Build header
    model_headers = "  ".join([f"{MODEL_DISPLAY_NAMES.get(m, m):>14}" for m in MODEL_ORDER])
    header = f"{'Component':<24} | {model_headers}"
    separator = "-" * len(header)

    lines = [
        "=" * len(header),
        f"COMPONENT-WISE WIS DIFFERENCE ({level.upper()} LEVEL)",
        "(Model - Clinician; positive = model improvement)",
        "=" * len(header),
        header,
        separator,
    ]

    level_key = f'{level}_level'

    for comp in REWARD_COMPONENTS:
        comp_display = COMPONENT_DISPLAY_NAMES.get(comp, comp)

        values = []
        for model_name in MODEL_ORDER:
            if model_name not in all_results:
                values.append(f"{'N/A':>14}")
            else:
                result = all_results[model_name].get(level_key, {}).get(comp, {})
                if 'error' in result or not result:
                    values.append(f"{'N/A':>14}")
                else:
                    wis = result.get('wis', 0)
                    # Handle NaN for sparse rewards (e.g., mortality at transition level)
                    if np.isnan(wis) if isinstance(wis, float) else False:
                        values.append(f"{'N/A':>14}")
                    else:
                        diff = wis - result.get('clinician', 0)
                        values.append(f"{diff:>14.4f}")

        lines.append(f"{comp_display:<24} | {'  '.join(values)}")

    lines.append(separator)

    return "\n".join(lines)


def generate_all_models_component_table_with_ci(all_results):
    """
    Generate ASCII table for trajectory-level with 95% CI.
    Shows: Difference [CI_lower, CI_upper]
    """
    # Build header - need more space for CI
    model_headers = "  ".join([f"{MODEL_DISPLAY_NAMES.get(m, m):>24}" for m in MODEL_ORDER])
    header = f"{'Component':<24} | {model_headers}"
    separator = "-" * len(header)

    lines = [
        "=" * len(header),
        "COMPONENT-WISE WIS (TRAJECTORY LEVEL) WITH 95% CI",
        "(Difference [CI_lower, CI_upper])",
        "=" * len(header),
        header,
        separator,
    ]

    for comp in REWARD_COMPONENTS:
        comp_display = COMPONENT_DISPLAY_NAMES.get(comp, comp)

        values = []
        for model_name in MODEL_ORDER:
            if model_name not in all_results:
                values.append(f"{'N/A':>24}")
            else:
                result = all_results[model_name].get('trajectory_level', {}).get(comp, {})
                if 'error' in result or not result:
                    values.append(f"{'N/A':>24}")
                else:
                    diff = result.get('wis', 0) - result.get('clinician', 0)
                    ci_l = result.get('ci_lower', 0)
                    ci_u = result.get('ci_upper', 0)
                    val_str = f"{diff:.2f} [{ci_l:.2f},{ci_u:.2f}]"
                    values.append(f"{val_str:>24}")

        lines.append(f"{comp_display:<24} | {'  '.join(values)}")

    lines.append(separator)

    return "\n".join(lines)


def generate_all_models_latex_table(all_results, output_path):
    """Generate LaTeX table for all models component-wise WIS."""
    n_models = len(MODEL_ORDER)

    latex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Component-wise WIS Improvement (Model - Clinician) Across IRL Methods}",
        r"\label{tab:component_wis_all_models}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{l|" + "c" * n_models + "}",
        r"\hline",
    ]

    # Header row: model names
    model_headers = " & ".join([MODEL_DISPLAY_NAMES.get(m, m) for m in MODEL_ORDER])
    latex_lines.append(f"\\textbf{{Component}} & {model_headers} \\\\")
    latex_lines.append(r"\hline")

    # Data rows (trajectory-level differences with CI)
    for comp in REWARD_COMPONENTS:
        comp_display = COMPONENT_DISPLAY_NAMES.get(comp, comp)

        values = []
        for model_name in MODEL_ORDER:
            if model_name not in all_results:
                values.append('--')
            else:
                result = all_results[model_name].get('trajectory_level', {}).get(comp, {})
                if 'error' in result or not result:
                    values.append('--')
                else:
                    diff = result.get('wis', 0) - result.get('clinician', 0)
                    ci_l = result.get('ci_lower', 0)
                    ci_u = result.get('ci_upper', 0)
                    # Bold if CI doesn't cross zero (significant)
                    if ci_l > 0 or ci_u < 0:
                        values.append(f"\\textbf{{{diff:.2f}}} [{ci_l:.2f},{ci_u:.2f}]")
                    else:
                        values.append(f"{diff:.2f} [{ci_l:.2f},{ci_u:.2f}]")

        latex_lines.append(f"{comp_display} & {' & '.join(values)} \\\\")

    latex_lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"}",
        r"\vspace{2mm}",
        r"",
        r"\footnotesize",
        r"\textit{Values show trajectory-level WIS difference (Model - Clinician) with 95\% CI.}\\",
        r"\textit{Bold values indicate statistically significant improvement (CI excludes zero).}",
        r"\end{table}",
    ])

    with open(output_path, 'w') as f:
        f.write("\n".join(latex_lines))

    print(f"All-models LaTeX table saved to: {output_path}")


# ============================================================================
# Single Model Evaluation (Reusable)
# ============================================================================

def evaluate_single_model(model_path, model_name, components_by_transition, eval_data, train_data,
                          n_bins, device, verbose=True):
    """
    Evaluate a single Q-learning model and return component-wise WIS results.

    Returns:
        Dict with 'transition_level' and 'trajectory_level' results
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"Evaluating: {MODEL_DISPLAY_NAMES.get(model_name, model_name)}")
        print(f"Model path: {model_path}")
        print(f"{'='*70}")

    n_actions = 2 * n_bins
    vp2_bin_edges = np.linspace(0, 0.5, n_bins + 1)

    # Load Q-networks
    try:
        q1_network, q2_network = load_q_networks(model_path, STATE_DIM, n_bins, device)
    except Exception as e:
        print(f"  ERROR loading model: {e}")
        return {'error': str(e)}

    # Generate model actions
    train_model_actions = select_action_batch_discrete(
        train_data['states'], q1_network, q2_network, n_bins, STATE_DIM, device
    )
    eval_model_actions = select_action_batch_discrete(
        eval_data['states'], q1_network, q2_network, n_bins, STATE_DIM, device
    )

    # Convert clinician actions to discrete
    train_clinician_actions = continuous_to_discrete_action(train_data['actions'], vp2_bin_edges, n_bins)
    eval_clinician_actions = continuous_to_discrete_action(eval_data['actions'], vp2_bin_edges, n_bins)

    # Train behavior policies
    clf_model, clf_clinician = train_behavior_policies(
        train_data['states'], train_model_actions, train_clinician_actions, n_actions
    )

    # Compute IS weights
    is_weight, isw_ci_lower, isw_ci_upper = compute_is_weights(
        eval_data['states'], eval_clinician_actions, clf_model, clf_clinician, n_actions
    )

    if verbose:
        print(f"  IS weight stats: mean={is_weight.mean():.4f}, std={is_weight.std():.4f}")

    # Compute component-wise WIS
    eval_patient_ids = eval_data['patient_ids']
    results_transition = {}
    results_trajectory = {}

    for comp in REWARD_COMPONENTS:
        comp_rewards = components_by_transition[comp]

        # Align lengths
        min_len = min(len(comp_rewards), len(is_weight))
        comp_rewards_aligned = comp_rewards[:min_len]
        is_weight_aligned = is_weight[:min_len]
        patient_ids_aligned = eval_patient_ids[:min_len]

        # Transform mortality rewards for per-transition credit assignment
        if comp == 'mortality':
            # Per-transition credit assignment for mortality
            # Assign K/L to each transition where K = +10 (survived) or -10 (died)
            comp_rewards_credit = np.zeros_like(comp_rewards_aligned)

            unique_patients = np.unique(patient_ids_aligned)
            for pid in unique_patients:
                mask = patient_ids_aligned == pid
                traj_len = mask.sum()  # L = trajectory length

                # Determine mortality outcome from raw component values
                # Raw mortality component: -10 at terminal state if died, 0 otherwise
                # So sum < 0 means patient died
                patient_mortality_sum = comp_rewards_aligned[mask].sum()
                if patient_mortality_sum < 0:  # patient died
                    K = -10
                else:  # patient survived
                    K = +10

                # Assign K/L to each transition
                comp_rewards_credit[mask] = K / traj_len

            comp_rewards_aligned = comp_rewards_credit

        # Standard handling for all components
        # Transition-level WIS
        wis_trans, clin_trans = compute_wis_transition_level(is_weight_aligned, comp_rewards_aligned)
        results_transition[comp] = {
            'wis': wis_trans,
            'clinician': clin_trans,
        }

        # Trajectory-level WIS
        wis_traj, clin_traj, ci_lower, ci_upper = compute_wis_trajectory_level(
            is_weight_aligned, comp_rewards_aligned, patient_ids_aligned, isw_ci_lower, isw_ci_upper, 
        )
        results_trajectory[comp] = {
            'wis': wis_traj,
            'clinician': clin_traj,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
        }

        if verbose:
            diff_trans = wis_trans - clin_trans
            diff_traj = wis_traj - clin_traj
            print(f"  {COMPONENT_DISPLAY_NAMES[comp]:<24}: Trans Diff={diff_trans:+.4f}, Traj Diff={diff_traj:+.4f}")

    return {
        'transition_level': results_transition,
        'trajectory_level': results_trajectory,
    }


# ============================================================================
# Run All Models Analysis
# ============================================================================

def run_all_models_analysis(eval_set='test', vp2_bins=5, output_dir='irl_analysis',
                            combined_or_train_data_path=None, eval_data_path=None):
    """
    Run component-wise WIS analysis for all IRL models.

    Evaluates each Q-learning model (trained with different IRL rewards)
    and generates combined tables.

    Args:
        eval_set: Which data split to evaluate on ('test' or 'val')
        vp2_bins: Number of bins for VP2 discretization
        output_dir: Directory to save results
        combined_or_train_data_path: Path to training dataset
        eval_data_path: Path to evaluation dataset (enables dual-dataset mode)
    """
    import pickle

    print("=" * 80)
    print("COMPONENT-WISE WIS ANALYSIS - ALL MODELS")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print(f"VP2 bins: {vp2_bins}")
    print(f"Eval set: {eval_set}")
    if eval_data_path:
        print(f"Dataset mode: DUAL-DATASET")
        print(f"  Train data: {combined_or_train_data_path or 'default'}")
        print(f"  Eval data:  {eval_data_path}")
    else:
        print(f"Dataset mode: SINGLE-DATASET")
        print(f"  Data path: {combined_or_train_data_path or 'default'}")
    print(f"Models to evaluate: {len(MODEL_ORDER)}")
    for m in MODEL_ORDER:
        print(f"  - {MODEL_DISPLAY_NAMES[m]}: {QL_MODEL_PATHS.get(m, 'N/A')}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # =========================================================================
    # Load data with component-wise rewards (only once)
    # =========================================================================
    print("\n" + "=" * 70)
    print("LOADING DATA AND COMPONENT REWARDS")
    print("=" * 70)

    components_by_transition, eval_data, train_data = load_component_rewards(
        eval_set=eval_set,
        random_seed=RANDOM_SEED,
        combined_or_train_data_path=combined_or_train_data_path,
        eval_data_path=eval_data_path
    )

    # =========================================================================
    # Evaluate each model
    # =========================================================================
    all_results = {}

    for model_name in MODEL_ORDER:
        model_path = QL_MODEL_PATHS.get(model_name)

        if model_path is None:
            print(f"\n  {model_name}: No Q-learning model path defined, skipping")
            continue

        if not os.path.exists(model_path):
            print(f"\n  {model_name}: Model file not found at {model_path}, skipping")
            continue

        result = evaluate_single_model(
            model_path=model_path,
            model_name=model_name,
            components_by_transition=components_by_transition,
            eval_data=eval_data,
            train_data=train_data,
            n_bins=vp2_bins,
            device=device,
            verbose=True
        )

        all_results[model_name] = result

    # =========================================================================
    # Generate combined tables
    # =========================================================================
    print("\n" + "=" * 80)
    print("COMBINED RESULTS - ALL MODELS")
    print("=" * 80)

    # Transition-level table
    print("\n")
    print(generate_all_models_component_table(all_results, level='transition'))

    # Trajectory-level table (simple)
    print("\n")
    print(generate_all_models_component_table(all_results, level='trajectory'))

    # Trajectory-level table with CI
    print("\n")
    print(generate_all_models_component_table_with_ci(all_results))

    # =========================================================================
    # Save results
    # =========================================================================
    os.makedirs(output_dir, exist_ok=True)

    # Save combined LaTeX table
    latex_path = os.path.join(output_dir, 'component_wis_all_models.tex')
    generate_all_models_latex_table(all_results, latex_path)

    # Save pickle
    results_data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'vp2_bins': vp2_bins,
            'eval_set': eval_set,
            'model_paths': QL_MODEL_PATHS,
            'irl_model_paths': IRL_MODEL_PATHS,
            'combined_or_train_data_path': combined_or_train_data_path,
            'eval_data_path': eval_data_path,
        },
        'all_results': all_results,
    }

    pickle_path = os.path.join(output_dir, 'component_wis_all_models.pkl')
    with open(pickle_path, 'wb') as f:
        pickle.dump(results_data, f)
    print(f"\nAll results saved to: {pickle_path}")

    return all_results


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    import sys

    # Check if running in "all models" mode or single model mode
    if len(sys.argv) == 1 or '--all' in sys.argv:
        # Run all models analysis
        eval_set = 'test'
        vp2_bins = DEFAULT_VP2_BINS
        output_dir = 'irl_analysis'
        combined_or_train_data_path = None
        eval_data_path = None
        use_lstm = False

        # Parse optional args for all-models mode
        for i, arg in enumerate(sys.argv):
            if arg == '--eval_set' and i + 1 < len(sys.argv):
                eval_set = sys.argv[i + 1]
            elif arg == '--vp2_bins' and i + 1 < len(sys.argv):
                vp2_bins = int(sys.argv[i + 1])
            elif arg == '--output_dir' and i + 1 < len(sys.argv):
                output_dir = sys.argv[i + 1]
            elif arg == '--combined_or_train_data_path' and i + 1 < len(sys.argv):
                combined_or_train_data_path = sys.argv[i + 1]
            elif arg == '--eval_data_path' and i + 1 < len(sys.argv):
                eval_data_path = sys.argv[i + 1]
            elif arg == '--use_lstm':
                use_lstm = True

        run_all_models_analysis(
            eval_set=eval_set,
            vp2_bins=vp2_bins,
            output_dir=output_dir,
            combined_or_train_data_path=combined_or_train_data_path,
            eval_data_path=eval_data_path,
            use_lstm=use_lstm
        )
    else:
        # Single model mode (original behavior)
        args = parse_args()
        run_component_wis_analysis(args)
