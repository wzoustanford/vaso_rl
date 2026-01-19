"""
Spearman Rank Correlation Analysis
Computes correlations between clinician-designed rewards and IRL rewards.

Two levels of analysis:
1. Transition-level: Rank all transitions, compute Spearman correlation
2. Trajectory-level: Sum rewards per trajectory, rank trajectories, compute correlation

Usage:
    python irl_analysis/spearman_rank_corr.py
"""

import os
import pickle
import numpy as np
from scipy.stats import spearmanr, pearsonr
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integrated_data_pipeline_v3 import IntegratedDataPipelineV3, compute_outcome_reward_components
from extract_irl_rewards import MODEL_PATHS
from data_loader import DataLoader, DataSplitter
import data_config as config


# Configuration
RANDOM_SEED = 42
REWARDS_PICKLE_PATH = 'irl_analysis/all_irl_rewards_test.pkl'
OUTPUT_DIR = 'irl_analysis'

# IRL model names to analyze (order for display)
IRL_MODELS = ['unet_tanh', 'semi_sup_unet', 'maxent_irl', 'gcl', 'iq_learn']

# Display names for table
MODEL_DISPLAY_NAMES = {
    'unet_tanh': 'U-Net (Tanh)',
    'semi_sup_unet': 'Semi-Sup U-Net',
    'maxent_irl': 'MaxEnt IRL',
    'gcl': 'GCL',
    'iq_learn': 'IQ-Learn',
}

# Clinician reward component names
REWARD_COMPONENTS = ['base', 'lactate', 'mbp', 'sofa', 'norepinephrine', 'mortality', 'total']

COMPONENT_DISPLAY_NAMES = {
    'base': 'Base (Survival)',
    'lactate': 'Lactate',
    'mbp': 'Blood Pressure',
    'sofa': 'SOFA Score',
    'norepinephrine': 'Norepinephrine',
    'mortality': 'Mortality',
    'total': 'Total',
}


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_rewards_from_pickle(pickle_path: str) -> Optional[Dict]:
    """Load rewards from pickle file."""
    if not os.path.exists(pickle_path):
        print(f"  WARNING: Pickle file not found: {pickle_path}")
        return None

    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)

    return data


def load_clinician_rewards_from_pipeline() -> Tuple[Dict[int, List[float]], Dict]:
    """
    Load clinician rewards directly from IntegratedDataPipelineV3.

    Returns:
        rewards_dict: Dict mapping patient_id -> list of rewards
        test_data: Full test data dict
    """
    print("Loading clinician rewards from IntegratedDataPipelineV3...")

    pipeline = IntegratedDataPipelineV3(
        model_type='dual',
        reward_source='manual',
        random_seed=RANDOM_SEED
    )
    train_data, val_data, test_data = pipeline.prepare_data()

    # Extract rewards by trajectory
    rewards_dict = {}
    for pid, (start_idx, end_idx) in test_data['patient_groups'].items():
        rewards_dict[pid] = test_data['rewards'][start_idx:end_idx].tolist()

    print(f"  Loaded {len(rewards_dict)} trajectories from TEST set")

    return rewards_dict, test_data


def load_clinician_reward_components() -> Dict[str, Dict[int, List[float]]]:
    """
    Load clinician reward components separately for each trajectory.

    Returns:
        Dict mapping component_name -> {patient_id -> list of component values}
        Components: 'base', 'lactate', 'mbp', 'sofa', 'norepinephrine', 'mortality', 'total'
    """
    print("Loading clinician reward components...")

    # Load raw data
    loader = DataLoader(verbose=False, random_seed=RANDOM_SEED)
    splitter = DataSplitter(random_seed=RANDOM_SEED)

    full_data = loader.load_data()
    patient_ids = loader.get_patient_ids()
    train_patients, val_patients, test_patients = splitter.split_patients(patient_ids)

    loader.encode_categorical_features()
    state_features = config.DUAL_STATE_FEATURES

    # Initialize component dictionaries
    components_dict = {comp: {} for comp in REWARD_COMPONENTS}

    for pid in test_patients:
        patient_data = loader.get_patient_data(pid)

        if len(patient_data) < 2:
            continue

        # Extract states and actions
        states = patient_data[state_features].values
        vp1 = patient_data['action_vaso'].values.astype(float)
        vp2 = patient_data['norepinephrine'].values.astype(float)
        actions = np.column_stack([vp1, vp2])

        mortality = int(patient_data[config.DEATH_COL].iloc[-1])

        # Initialize component lists for this patient
        for comp in REWARD_COMPONENTS:
            components_dict[comp][pid] = []

        # Compute components for each transition
        for t in range(len(states) - 1):
            is_terminal = (t == len(states) - 2)

            comp_values = compute_outcome_reward_components(
                states, actions, t, is_terminal, mortality, state_features
            )

            for comp in REWARD_COMPONENTS:
                components_dict[comp][pid].append(comp_values[comp])

    print(f"  Loaded components for {len(components_dict['total'])} trajectories")

    return components_dict


# ============================================================================
# Transition-Level Correlation (Per-Trajectory, then Averaged)
# ============================================================================

def compute_transition_level_correlation(
    clinician_rewards: Dict[int, List[float]],
    irl_rewards: Dict[int, List[float]],
    min_trajectory_length: int = 5
) -> Dict:
    """
    Compute transition-level Spearman and Pearson correlations.

    For each trajectory, compute the correlation between clinician and IRL
    rewards across transitions within that trajectory. Then average the
    correlations across all trajectories.

    Args:
        clinician_rewards: Dict mapping patient_id -> list of rewards
        irl_rewards: Dict mapping patient_id -> list of rewards
        min_trajectory_length: Minimum trajectory length to compute correlation

    Returns:
        Dict with correlation results (mean and std across trajectories)
    """
    # Get common trajectory IDs
    common_pids = set(clinician_rewards.keys()) & set(irl_rewards.keys())

    if len(common_pids) == 0:
        return {'error': 'No common trajectories found'}

    # Compute correlation for each trajectory
    spearman_corrs = []
    pearson_corrs = []
    valid_trajectories = 0
    total_transitions = 0
    skipped_short = 0
    skipped_constant = 0

    for pid in common_pids:
        clin_r = np.array(clinician_rewards[pid])
        irl_r = np.array(irl_rewards[pid])

        # Ensure same length
        min_len = min(len(clin_r), len(irl_r))
        clin_r = clin_r[:min_len]
        irl_r = irl_r[:min_len]

        total_transitions += min_len

        # Skip short trajectories
        if min_len < min_trajectory_length:
            skipped_short += 1
            continue

        # Skip if either has zero variance (constant rewards)
        if np.std(clin_r) < 1e-10 or np.std(irl_r) < 1e-10:
            skipped_constant += 1
            continue

        # Compute correlations for this trajectory
        sp_corr, _ = spearmanr(clin_r, irl_r)
        pe_corr, _ = pearsonr(clin_r, irl_r)

        # Skip if NaN (can happen with ties)
        if not np.isnan(sp_corr):
            spearman_corrs.append(sp_corr)
        if not np.isnan(pe_corr):
            pearson_corrs.append(pe_corr)

        valid_trajectories += 1

    spearman_corrs = np.array(spearman_corrs)
    pearson_corrs = np.array(pearson_corrs)

    if len(spearman_corrs) == 0:
        return {'error': 'No valid trajectories for correlation'}

    return {
        'n_transitions': total_transitions,
        'n_trajectories': len(common_pids),
        'n_valid_trajectories': valid_trajectories,
        'n_skipped_short': skipped_short,
        'n_skipped_constant': skipped_constant,
        # Spearman: mean and std across trajectories
        'spearman_corr': spearman_corrs.mean(),
        'spearman_std': spearman_corrs.std(),
        'spearman_median': np.median(spearman_corrs),
        'spearman_min': spearman_corrs.min(),
        'spearman_max': spearman_corrs.max(),
        # Pearson: mean and std across trajectories
        'pearson_corr': pearson_corrs.mean(),
        'pearson_std': pearson_corrs.std(),
        'pearson_median': np.median(pearson_corrs),
        # For p-value, use t-test on correlations (H0: mean = 0)
        'spearman_pval': None,  # Will compute below
        'pearson_pval': None,
    }


def compute_transition_level_correlation_with_pval(
    clinician_rewards: Dict[int, List[float]],
    irl_rewards: Dict[int, List[float]],
    min_trajectory_length: int = 5
) -> Dict:
    """
    Wrapper that also computes p-value for mean correlation being different from 0.
    Uses one-sample t-test on the per-trajectory correlations.
    """
    from scipy.stats import ttest_1samp

    result = compute_transition_level_correlation(
        clinician_rewards, irl_rewards, min_trajectory_length
    )

    if 'error' in result:
        return result

    # Recompute to get the arrays for t-test
    common_pids = set(clinician_rewards.keys()) & set(irl_rewards.keys())

    spearman_corrs = []
    pearson_corrs = []

    for pid in common_pids:
        clin_r = np.array(clinician_rewards[pid])
        irl_r = np.array(irl_rewards[pid])
        min_len = min(len(clin_r), len(irl_r))
        clin_r = clin_r[:min_len]
        irl_r = irl_r[:min_len]

        if min_len < min_trajectory_length:
            continue
        if np.std(clin_r) < 1e-10 or np.std(irl_r) < 1e-10:
            continue

        sp_corr, _ = spearmanr(clin_r, irl_r)
        pe_corr, _ = pearsonr(clin_r, irl_r)

        if not np.isnan(sp_corr):
            spearman_corrs.append(sp_corr)
        if not np.isnan(pe_corr):
            pearson_corrs.append(pe_corr)

    # One-sample t-test: is the mean correlation significantly different from 0?
    if len(spearman_corrs) > 1:
        _, sp_pval = ttest_1samp(spearman_corrs, 0)
        result['spearman_pval'] = sp_pval
    if len(pearson_corrs) > 1:
        _, pe_pval = ttest_1samp(pearson_corrs, 0)
        result['pearson_pval'] = pe_pval

    return result


# ============================================================================
# Trajectory-Level Correlation (Normalized by Trajectory Length)
# ============================================================================

def compute_trajectory_level_correlation(
    clinician_rewards: Dict[int, List[float]],
    irl_rewards: Dict[int, List[float]]
) -> Dict:
    """
    Compute trajectory-level Spearman rank correlation.

    Computes mean reward per trajectory (sum / length), then computes
    Spearman correlation between clinician and IRL trajectory-level means.

    This normalization accounts for different trajectory lengths.

    Args:
        clinician_rewards: Dict mapping patient_id -> list of rewards
        irl_rewards: Dict mapping patient_id -> list of rewards

    Returns:
        Dict with correlation results
    """
    # Get common trajectory IDs
    common_pids = set(clinician_rewards.keys()) & set(irl_rewards.keys())

    if len(common_pids) == 0:
        return {'error': 'No common trajectories found'}

    # Compute mean reward per trajectory (normalized by length)
    clinician_means = []
    irl_means = []
    trajectory_lengths = []

    for pid in common_pids:
        clin_r = clinician_rewards[pid]
        irl_r = irl_rewards[pid]

        # Normalize by trajectory length
        clin_len = len(clin_r)
        irl_len = len(irl_r)

        clinician_means.append(sum(clin_r) / clin_len if clin_len > 0 else 0.0)
        irl_means.append(sum(irl_r) / irl_len if irl_len > 0 else 0.0)
        trajectory_lengths.append(clin_len)

    clinician_means = np.array(clinician_means)
    irl_means = np.array(irl_means)

    # Compute Spearman correlation on trajectory means
    spearman_corr, spearman_pval = spearmanr(clinician_means, irl_means)

    # Also compute Pearson for reference
    pearson_corr, pearson_pval = pearsonr(clinician_means, irl_means)

    return {
        'n_trajectories': len(common_pids),
        'spearman_corr': spearman_corr,
        'spearman_pval': spearman_pval,
        'pearson_corr': pearson_corr,
        'pearson_pval': pearson_pval,
        'clinician_mean_mean': clinician_means.mean(),
        'clinician_mean_std': clinician_means.std(),
        'irl_mean_mean': irl_means.mean(),
        'irl_mean_std': irl_means.std(),
        'avg_trajectory_length': np.mean(trajectory_lengths),
    }


# ============================================================================
# Table Generation Functions
# ============================================================================

def generate_combined_table(results: Dict) -> str:
    """
    Generate a combined ASCII table with trajectory-level (left) and
    transition-level (right) correlations side by side.

    Trajectory-level: Spearman ρ and p-value (single correlation across trajectory sums)
    Transition-level: Mean Spearman ρ and Mean Pearson r (averaged across per-trajectory correlations)
    """
    # Table header
    header = (
        f"{'Model':<18} | "
        f"{'Traj-Level ρ':>12} {'p-value':>12} | "
        f"{'Trans Spearman':>14} {'Trans Pearson':>14}"
    )
    separator = "-" * len(header)

    lines = [
        "=" * len(header),
        "CORRELATION: Clinician vs IRL Rewards",
        "=" * len(header),
        header,
        separator,
    ]

    for model_name in IRL_MODELS:
        display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)

        traj = results['trajectory_level'].get(model_name, {})
        trans = results['transition_level'].get(model_name, {})

        if 'error' in traj or not traj:
            traj_r, traj_p = 'N/A', 'N/A'
        else:
            traj_r = f"{traj['spearman_corr']:.4f}"
            traj_p = f"{traj['spearman_pval']:.2e}"

        if 'error' in trans or not trans:
            trans_spearman, trans_pearson = 'N/A', 'N/A'
        else:
            trans_spearman = f"{trans['spearman_corr']:.4f}"
            trans_pearson = f"{trans['pearson_corr']:.4f}"

        lines.append(
            f"{display_name:<18} | "
            f"{traj_r:>12} {traj_p:>12} | "
            f"{trans_spearman:>14} {trans_pearson:>14}"
        )

    lines.append(separator)
    lines.append("")
    lines.append("Traj-Level: Spearman ρ on mean reward per trajectory (sum/length)")
    lines.append("Trans-Level: Mean correlation across per-trajectory correlations")

    return "\n".join(lines)


def generate_latex_table(results: Dict, output_path: str):
    """
    Generate a LaTeX table file with trajectory-level and transition-level
    correlations side by side.

    Trajectory-level: Spearman ρ and p-value
    Transition-level: Mean Spearman ρ and Mean Pearson r (averaged across trajectories)
    """
    latex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Correlation between Clinician-Designed and IRL-Learned Rewards}",
        r"\label{tab:reward_correlation}",
        r"\begin{tabular}{l|cc|cc}",
        r"\hline",
        r"\multirow{2}{*}{\textbf{IRL Method}} & \multicolumn{2}{c|}{\textbf{Trajectory-Level}} & \multicolumn{2}{c}{\textbf{Transition-Level}} \\",
        r" & Spearman $\rho$ & $p$-value & Spearman $\rho$ & Pearson $r$ \\",
        r"\hline",
    ]

    for model_name in IRL_MODELS:
        display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        # Escape underscores for LaTeX
        display_name_latex = display_name.replace('_', r'\_')

        traj = results['trajectory_level'].get(model_name, {})
        trans = results['transition_level'].get(model_name, {})

        if 'error' in traj or not traj:
            traj_r, traj_p = '--', '--'
        else:
            traj_r = f"{traj['spearman_corr']:.4f}"
            traj_p = f"{traj['spearman_pval']:.2e}"

        if 'error' in trans or not trans:
            trans_spearman, trans_pearson = '--', '--'
        else:
            trans_spearman = f"{trans['spearman_corr']:.4f}"
            trans_pearson = f"{trans['pearson_corr']:.4f}"

        latex_lines.append(
            f"{display_name_latex} & {traj_r} & {traj_p} & {trans_spearman} & {trans_pearson} \\\\"
        )

    latex_lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\vspace{2mm}",
        r"",
        r"\footnotesize",
        r"\textit{Trajectory-level: Spearman $\rho$ computed on mean rewards per trajectory (sum/length).}\\",
        r"\textit{Transition-level: Mean of per-trajectory correlations.}",
        r"\end{table}",
    ])

    with open(output_path, 'w') as f:
        f.write("\n".join(latex_lines))

    print(f"LaTeX table saved to: {output_path}")


# ============================================================================
# Component-wise Analysis Functions
# ============================================================================

def compute_component_correlations(
    components_dict: Dict[str, Dict[int, List[float]]],
    irl_rewards: Dict[int, List[float]],
    min_trajectory_length: int = 5
) -> Dict[str, Dict]:
    """
    Compute correlations between each clinician reward component and IRL rewards.

    Args:
        components_dict: Dict mapping component_name -> {patient_id -> list of values}
        irl_rewards: Dict mapping patient_id -> list of IRL rewards
        min_trajectory_length: Minimum trajectory length for correlation

    Returns:
        Dict mapping component_name -> correlation results
    """
    results = {}

    for comp_name in REWARD_COMPONENTS:
        comp_rewards = components_dict[comp_name]

        # Use transition-level correlation (per-trajectory, then averaged)
        corr_result = compute_transition_level_correlation_with_pval(
            comp_rewards, irl_rewards, min_trajectory_length
        )
        results[comp_name] = corr_result

    return results


def generate_component_table(
    component_results: Dict[str, Dict[str, Dict]],
    irl_model_name: str
) -> str:
    """
    Generate ASCII table for component-wise correlations for a single IRL model.

    Args:
        component_results: Dict mapping component_name -> correlation results
        irl_model_name: Name of the IRL model
    """
    header = (
        f"{'Component':<20} | "
        f"{'Spearman ρ':>12} {'Pearson r':>12}"
    )
    separator = "-" * len(header)

    display_name = MODEL_DISPLAY_NAMES.get(irl_model_name, irl_model_name)

    lines = [
        "=" * len(header),
        f"COMPONENT-WISE CORRELATION: Clinician vs {display_name}",
        "=" * len(header),
        header,
        separator,
    ]

    for comp_name in REWARD_COMPONENTS:
        comp_display = COMPONENT_DISPLAY_NAMES.get(comp_name, comp_name)
        result = component_results.get(comp_name, {})

        if 'error' in result or not result:
            spearman, pearson = 'N/A', 'N/A'
        else:
            spearman = f"{result['spearman_corr']:.4f}"
            pearson = f"{result['pearson_corr']:.4f}"

        lines.append(
            f"{comp_display:<20} | "
            f"{spearman:>12} {pearson:>12}"
        )

    lines.append(separator)

    return "\n".join(lines)


def generate_all_components_table(
    all_component_results: Dict[str, Dict[str, Dict]]
) -> str:
    """
    Generate ASCII table for all IRL models and all components.

    Args:
        all_component_results: Dict mapping irl_model -> component_name -> correlation results
    """
    # Header with all IRL models
    model_headers = "  ".join([f"{MODEL_DISPLAY_NAMES.get(m, m):>14}" for m in IRL_MODELS])
    header = f"{'Component':<18} | {model_headers}"
    separator = "-" * len(header)

    lines = [
        "=" * len(header),
        "COMPONENT-WISE SPEARMAN CORRELATION (Mean per-trajectory)",
        "=" * len(header),
        header,
        separator,
    ]

    for comp_name in REWARD_COMPONENTS:
        comp_display = COMPONENT_DISPLAY_NAMES.get(comp_name, comp_name)

        values = []
        for model_name in IRL_MODELS:
            if model_name not in all_component_results:
                values.append(f"{'N/A':>14}")
            else:
                result = all_component_results[model_name].get(comp_name, {})
                if 'error' in result or not result:
                    values.append(f"{'N/A':>14}")
                else:
                    values.append(f"{result['spearman_corr']:>14.4f}")

        lines.append(f"{comp_display:<18} | {'  '.join(values)}")

    lines.append(separator)

    # Add Pearson section
    lines.append("")
    lines.append("=" * len(header))
    lines.append("COMPONENT-WISE PEARSON CORRELATION (Mean per-trajectory)")
    lines.append("=" * len(header))
    lines.append(header)
    lines.append(separator)

    for comp_name in REWARD_COMPONENTS:
        comp_display = COMPONENT_DISPLAY_NAMES.get(comp_name, comp_name)

        values = []
        for model_name in IRL_MODELS:
            if model_name not in all_component_results:
                values.append(f"{'N/A':>14}")
            else:
                result = all_component_results[model_name].get(comp_name, {})
                if 'error' in result or not result:
                    values.append(f"{'N/A':>14}")
                else:
                    values.append(f"{result['pearson_corr']:>14.4f}")

        lines.append(f"{comp_display:<18} | {'  '.join(values)}")

    lines.append(separator)

    return "\n".join(lines)


def generate_component_latex_table(
    all_component_results: Dict[str, Dict[str, Dict]],
    output_path: str
):
    """
    Generate LaTeX table for component-wise correlations.
    """
    n_models = len(IRL_MODELS)

    latex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Component-wise Correlation between Clinician Reward Components and IRL Rewards}",
        r"\label{tab:component_correlation}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{l|" + "cc|" * n_models + "}",
        r"\hline",
    ]

    # Header row 1: IRL model names
    model_headers = " & ".join([
        f"\\multicolumn{{2}}{{c|}}{{{MODEL_DISPLAY_NAMES.get(m, m)}}}"
        for m in IRL_MODELS
    ])
    latex_lines.append(f"\\textbf{{Component}} & {model_headers} \\\\")

    # Header row 2: Spearman/Pearson for each model
    corr_headers = " & ".join(["$\\rho$ & $r$"] * n_models)
    latex_lines.append(f" & {corr_headers} \\\\")
    latex_lines.append(r"\hline")

    # Data rows
    for comp_name in REWARD_COMPONENTS:
        comp_display = COMPONENT_DISPLAY_NAMES.get(comp_name, comp_name)

        values = []
        for model_name in IRL_MODELS:
            if model_name not in all_component_results:
                values.extend(['--', '--'])
            else:
                result = all_component_results[model_name].get(comp_name, {})
                if 'error' in result or not result:
                    values.extend(['--', '--'])
                else:
                    values.append(f"{result['spearman_corr']:.3f}")
                    values.append(f"{result['pearson_corr']:.3f}")

        latex_lines.append(f"{comp_display} & {' & '.join(values)} \\\\")

    latex_lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"}",
        r"\vspace{2mm}",
        r"",
        r"\footnotesize",
        r"\textit{$\rho$: Spearman correlation, $r$: Pearson correlation.}\\",
        r"\textit{Values are mean correlations across per-trajectory computations.}",
        r"\end{table}",
    ])

    with open(output_path, 'w') as f:
        f.write("\n".join(latex_lines))

    print(f"Component LaTeX table saved to: {output_path}")


# ============================================================================
# Component-wise Trajectory-Level Analysis (Mean per Trajectory, then Rank)
# ============================================================================

def compute_component_trajectory_level_correlations(
    components_dict: Dict[str, Dict[int, List[float]]],
    irl_rewards: Dict[int, List[float]]
) -> Dict[str, Dict]:
    """
    Compute trajectory-level correlations between each clinician reward component and IRL rewards.

    For each component:
    1. Compute mean component value per trajectory (sum / length)
    2. Compute mean IRL reward per trajectory (sum / length)
    3. Compute single Spearman/Pearson correlation on those trajectory-level means

    This normalization accounts for different trajectory lengths.

    Args:
        components_dict: Dict mapping component_name -> {patient_id -> list of values}
        irl_rewards: Dict mapping patient_id -> list of IRL rewards

    Returns:
        Dict mapping component_name -> correlation results
    """
    results = {}

    # Get common patient IDs across IRL rewards and components
    irl_pids = set(irl_rewards.keys())

    for comp_name in REWARD_COMPONENTS:
        comp_rewards = components_dict[comp_name]
        common_pids = irl_pids & set(comp_rewards.keys())

        if len(common_pids) < 3:
            results[comp_name] = {'error': 'Too few common trajectories'}
            continue

        # Compute mean per trajectory (normalized by length)
        comp_means = []
        irl_means = []

        for pid in common_pids:
            comp_r = comp_rewards[pid]
            irl_r = irl_rewards[pid]

            comp_len = len(comp_r)
            irl_len = len(irl_r)

            comp_means.append(sum(comp_r) / comp_len if comp_len > 0 else 0.0)
            irl_means.append(sum(irl_r) / irl_len if irl_len > 0 else 0.0)

        comp_means = np.array(comp_means)
        irl_means = np.array(irl_means)

        # Check for zero variance (e.g., base survival is constant per-transition mean)
        if np.std(comp_means) < 1e-10:
            results[comp_name] = {'error': 'Zero variance in component means'}
            continue

        # Compute correlations
        spearman_corr, spearman_pval = spearmanr(comp_means, irl_means)
        pearson_corr, pearson_pval = pearsonr(comp_means, irl_means)

        results[comp_name] = {
            'n_trajectories': len(common_pids),
            'spearman_corr': spearman_corr,
            'spearman_pval': spearman_pval,
            'pearson_corr': pearson_corr,
            'pearson_pval': pearson_pval,
            'comp_mean_mean': comp_means.mean(),
            'comp_mean_std': comp_means.std(),
            'irl_mean_mean': irl_means.mean(),
            'irl_mean_std': irl_means.std(),
        }

    return results


def generate_all_components_trajectory_table(
    all_component_results: Dict[str, Dict[str, Dict]]
) -> str:
    """
    Generate ASCII table for trajectory-level component correlations.

    Args:
        all_component_results: Dict mapping irl_model -> component_name -> correlation results
    """
    # Header with all IRL models
    model_headers = "  ".join([f"{MODEL_DISPLAY_NAMES.get(m, m):>14}" for m in IRL_MODELS])
    header = f"{'Component':<18} | {model_headers}"
    separator = "-" * len(header)

    lines = [
        "=" * len(header),
        "COMPONENT-WISE SPEARMAN CORRELATION (Trajectory-Level Mean)",
        "=" * len(header),
        header,
        separator,
    ]

    for comp_name in REWARD_COMPONENTS:
        comp_display = COMPONENT_DISPLAY_NAMES.get(comp_name, comp_name)

        values = []
        for model_name in IRL_MODELS:
            if model_name not in all_component_results:
                values.append(f"{'N/A':>14}")
            else:
                result = all_component_results[model_name].get(comp_name, {})
                if 'error' in result or not result:
                    values.append(f"{'N/A':>14}")
                else:
                    values.append(f"{result['spearman_corr']:>14.4f}")

        lines.append(f"{comp_display:<18} | {'  '.join(values)}")

    lines.append(separator)

    # Add Pearson section
    lines.append("")
    lines.append("=" * len(header))
    lines.append("COMPONENT-WISE PEARSON CORRELATION (Trajectory-Level Mean)")
    lines.append("=" * len(header))
    lines.append(header)
    lines.append(separator)

    for comp_name in REWARD_COMPONENTS:
        comp_display = COMPONENT_DISPLAY_NAMES.get(comp_name, comp_name)

        values = []
        for model_name in IRL_MODELS:
            if model_name not in all_component_results:
                values.append(f"{'N/A':>14}")
            else:
                result = all_component_results[model_name].get(comp_name, {})
                if 'error' in result or not result:
                    values.append(f"{'N/A':>14}")
                else:
                    values.append(f"{result['pearson_corr']:>14.4f}")

        lines.append(f"{comp_display:<18} | {'  '.join(values)}")

    lines.append(separator)

    return "\n".join(lines)


def generate_component_trajectory_latex_table(
    all_component_results: Dict[str, Dict[str, Dict]],
    output_path: str
):
    """
    Generate LaTeX table for trajectory-level component correlations.
    """
    n_models = len(IRL_MODELS)

    latex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Component-wise Correlation (Trajectory-Level Mean)}",
        r"\label{tab:component_correlation_trajectory}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{l|" + "cc|" * n_models + "}",
        r"\hline",
    ]

    # Header row 1: IRL model names
    model_headers = " & ".join([
        f"\\multicolumn{{2}}{{c|}}{{{MODEL_DISPLAY_NAMES.get(m, m)}}}"
        for m in IRL_MODELS
    ])
    latex_lines.append(f"\\textbf{{Component}} & {model_headers} \\\\")

    # Header row 2: Spearman/Pearson for each model
    corr_headers = " & ".join(["$\\rho$ & $r$"] * n_models)
    latex_lines.append(f" & {corr_headers} \\\\")
    latex_lines.append(r"\hline")

    # Data rows
    for comp_name in REWARD_COMPONENTS:
        comp_display = COMPONENT_DISPLAY_NAMES.get(comp_name, comp_name)

        values = []
        for model_name in IRL_MODELS:
            if model_name not in all_component_results:
                values.extend(['--', '--'])
            else:
                result = all_component_results[model_name].get(comp_name, {})
                if 'error' in result or not result:
                    values.extend(['--', '--'])
                else:
                    values.append(f"{result['spearman_corr']:.3f}")
                    values.append(f"{result['pearson_corr']:.3f}")

        latex_lines.append(f"{comp_display} & {' & '.join(values)} \\\\")

    latex_lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"}",
        r"\vspace{2mm}",
        r"",
        r"\footnotesize",
        r"\textit{$\rho$: Spearman correlation, $r$: Pearson correlation.}\\",
        r"\textit{Values computed on trajectory-level means (sum/length per trajectory).}",
        r"\end{table}",
    ])

    with open(output_path, 'w') as f:
        f.write("\n".join(latex_lines))

    print(f"Component trajectory-level LaTeX table saved to: {output_path}")


# ============================================================================
# Main Analysis Function
# ============================================================================

def run_correlation_analysis():
    """Run full Spearman correlation analysis."""

    print("="*70)
    print("SPEARMAN RANK CORRELATION ANALYSIS")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    # =========================================================================
    # Load clinician rewards from pipeline
    # =========================================================================
    clinician_rewards, test_data = load_clinician_rewards_from_pipeline()

    # =========================================================================
    # Load IRL rewards from pickle
    # =========================================================================
    print(f"\nLoading IRL rewards from: {REWARDS_PICKLE_PATH}")
    rewards_data = load_rewards_from_pickle(REWARDS_PICKLE_PATH)

    if rewards_data is None:
        print("ERROR: Could not load rewards pickle file.")
        print("Please run extract_irl_rewards.py first.")
        return None

    # =========================================================================
    # Compute correlations for each IRL model
    # =========================================================================
    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'random_seed': RANDOM_SEED,
            'eval_set': 'test',
            'model_paths': MODEL_PATHS,
        },
        'transition_level': {},
        'trajectory_level': {},
    }

    print("\nComputing correlations for all IRL models...")

    for model_name in IRL_MODELS:
        if model_name not in rewards_data:
            print(f"  {model_name}: NOT FOUND in pickle")
            continue

        irl_rewards = rewards_data[model_name]

        # Trajectory-level correlation
        traj_result = compute_trajectory_level_correlation(
            clinician_rewards, irl_rewards
        )
        results['trajectory_level'][model_name] = traj_result

        # Transition-level correlation (per-trajectory, then averaged)
        trans_result = compute_transition_level_correlation_with_pval(
            clinician_rewards, irl_rewards
        )
        results['transition_level'][model_name] = trans_result

        print(f"  {model_name}: done")

    # =========================================================================
    # Print combined table
    # =========================================================================
    print("\n")
    table_str = generate_combined_table(results)
    print(table_str)

    # =========================================================================
    # Save results
    # =========================================================================
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save pickle
    pickle_path = os.path.join(OUTPUT_DIR, 'spearman_correlation_results.pkl')
    with open(pickle_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to: {pickle_path}")

    # Save LaTeX table
    latex_path = os.path.join(OUTPUT_DIR, 'spearman_correlation_table.tex')
    generate_latex_table(results, latex_path)

    # =========================================================================
    # Print summary statistics
    # =========================================================================
    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS")
    print(f"{'='*70}")

    # Get sample sizes from first available model
    for model_name in IRL_MODELS:
        if model_name in results['trajectory_level']:
            traj = results['trajectory_level'][model_name]
            trans = results['transition_level'][model_name]
            if 'error' not in traj and 'error' not in trans:
                print(f"N trajectories: {traj['n_trajectories']}")
                print(f"N transitions:  {trans['n_transitions']}")
                break

    # =========================================================================
    # Component-wise Analysis
    # =========================================================================
    print(f"\n{'='*70}")
    print("COMPONENT-WISE CORRELATION ANALYSIS")
    print(f"{'='*70}")

    # Load clinician reward components
    components_dict = load_clinician_reward_components()

    # Compute correlations for each IRL model and each component
    all_component_results = {}

    for model_name in IRL_MODELS:
        if model_name not in rewards_data:
            print(f"  {model_name}: NOT FOUND in pickle")
            continue

        irl_rewards = rewards_data[model_name]

        # Compute component-wise correlations
        comp_results = compute_component_correlations(
            components_dict, irl_rewards
        )
        all_component_results[model_name] = comp_results

        print(f"  {model_name}: component analysis done")

    # Add component results to main results
    results['component_wise'] = all_component_results

    # Print component-wise tables
    print("\n")
    component_table_str = generate_all_components_table(all_component_results)
    print(component_table_str)

    # Save component-wise LaTeX table (transition-level)
    component_latex_path = os.path.join(OUTPUT_DIR, 'component_correlation_transition_table.tex')
    generate_component_latex_table(all_component_results, component_latex_path)

    # =========================================================================
    # Component-wise Trajectory-Level Analysis (Sum per Trajectory)
    # =========================================================================
    print(f"\n{'='*70}")
    print("COMPONENT-WISE TRAJECTORY-LEVEL ANALYSIS")
    print(f"{'='*70}")

    # Compute trajectory-level component correlations for each IRL model
    all_component_traj_results = {}

    for model_name in IRL_MODELS:
        if model_name not in rewards_data:
            print(f"  {model_name}: NOT FOUND in pickle")
            continue

        irl_rewards = rewards_data[model_name]

        # Compute trajectory-level component correlations
        comp_traj_results = compute_component_trajectory_level_correlations(
            components_dict, irl_rewards
        )
        all_component_traj_results[model_name] = comp_traj_results

        print(f"  {model_name}: trajectory-level component analysis done")

    # Add trajectory-level component results to main results
    results['component_wise_trajectory'] = all_component_traj_results

    # Print trajectory-level component tables
    print("\n")
    component_traj_table_str = generate_all_components_trajectory_table(all_component_traj_results)
    print(component_traj_table_str)

    # Save trajectory-level component LaTeX table
    component_traj_latex_path = os.path.join(OUTPUT_DIR, 'component_correlation_trajectory_table.tex')
    generate_component_trajectory_latex_table(all_component_traj_results, component_traj_latex_path)

    # Update pickle with all results
    with open(pickle_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nUpdated results saved to: {pickle_path}")

    return results


if __name__ == "__main__":
    run_correlation_analysis()
