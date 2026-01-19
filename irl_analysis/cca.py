"""
Canonical Correlation Analysis (CCA)
Computes canonical correlations between IRL rewards and clinician reward components.

X: IRL reward (1-dimensional, trajectory-level mean)
Y: Clinician reward components (6-dimensional, trajectory-level means)
   - Base (Survival), Lactate, Blood Pressure, SOFA, Norepinephrine, Mortality

For each IRL algorithm, we find the first canonical correlation (rho) between
X and Y, which measures how well the IRL reward captures the clinician's
multi-dimensional reward structure.

Usage:
    python irl_analysis/cca.py
"""

import os
import pickle
import numpy as np
from sklearn.cross_decomposition import CCA
from scipy.stats import pearsonr
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integrated_data_pipeline_v3 import IntegratedDataPipelineV3, compute_outcome_reward_components
from extract_irl_rewards import MODEL_PATHS
from data_loader import DataLoader, DataSplitter
import data_config as config


# ============================================================================
# Configuration
# ============================================================================

RANDOM_SEED = 42

# Use combined validation + test data for more robust CCA analysis
USE_VAL_AND_TEST = True

# Pickle paths for IRL rewards
REWARDS_PICKLE_PATH_COMBINED = 'irl_analysis/all_irl_rewards_val_test.pkl'
REWARDS_PICKLE_PATH_TEST = 'irl_analysis/all_irl_rewards_test.pkl'

OUTPUT_DIR = 'irl_analysis/cca'

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

# Clinician reward component names (6 components, excluding 'total')
REWARD_COMPONENTS = ['base', 'lactate', 'mbp', 'sofa', 'norepinephrine', 'mortality']

COMPONENT_DISPLAY_NAMES = {
    'base': 'Base (Survival)',
    'lactate': 'Lactate',
    'mbp': 'Blood Pressure',
    'sofa': 'SOFA Score',
    'norepinephrine': 'Norepinephrine',
    'mortality': 'Mortality',
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


def load_clinician_reward_components(include_val: bool = False) -> Dict[str, Dict[int, List[float]]]:
    """
    Load clinician reward components separately for each trajectory.

    Args:
        include_val: If True, include both validation and test patients.
                     If False, only include test patients.

    Returns:
        Dict mapping component_name -> {patient_id -> list of component values}
        Components: 'base', 'lactate', 'mbp', 'sofa', 'norepinephrine', 'mortality'
    """
    dataset_name = "validation + test" if include_val else "test"
    print(f"Loading clinician reward components ({dataset_name})...")

    # Load raw data
    loader = DataLoader(verbose=False, random_seed=RANDOM_SEED)
    splitter = DataSplitter(random_seed=RANDOM_SEED)

    full_data = loader.load_data()
    patient_ids = loader.get_patient_ids()
    train_patients, val_patients, test_patients = splitter.split_patients(patient_ids)

    loader.encode_categorical_features()
    state_features = config.DUAL_STATE_FEATURES

    # Determine which patients to include
    if include_val:
        target_patients = list(val_patients) + list(test_patients)
        print(f"  Validation patients: {len(val_patients)}")
        print(f"  Test patients: {len(test_patients)}")
        print(f"  Total patients: {len(target_patients)}")
    else:
        target_patients = list(test_patients)
        print(f"  Test patients: {len(test_patients)}")

    # Initialize component dictionaries
    components_dict = {comp: {} for comp in REWARD_COMPONENTS}

    for pid in target_patients:
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

    print(f"  Loaded components for {len(components_dict['base'])} trajectories")

    return components_dict


# ============================================================================
# CCA Analysis Functions
# ============================================================================

def compute_trajectory_level_means(
    rewards_dict: Dict[int, List[float]]
) -> Tuple[np.ndarray, List[int]]:
    """
    Compute trajectory-level means (sum / length) for rewards.

    Args:
        rewards_dict: Dict mapping patient_id -> list of rewards

    Returns:
        means: np.array of shape (n_trajectories,)
        pids: List of patient IDs in order
    """
    means = []
    pids = []

    for pid, rewards in rewards_dict.items():
        if len(rewards) > 0:
            means.append(sum(rewards) / len(rewards))
            pids.append(pid)

    return np.array(means), pids


def prepare_cca_data(
    irl_rewards: Dict[int, List[float]],
    components_dict: Dict[str, Dict[int, List[float]]]
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Prepare data matrices for CCA.

    Args:
        irl_rewards: Dict mapping patient_id -> list of IRL rewards
        components_dict: Dict mapping component_name -> {patient_id -> list of values}

    Returns:
        X: np.array of shape (n_trajectories, 1) - IRL reward means
        Y: np.array of shape (n_trajectories, 6) - Component means
        n_trajectories: Number of common trajectories
    """
    # Get common patient IDs across IRL rewards and all components
    common_pids = set(irl_rewards.keys())
    for comp_name in REWARD_COMPONENTS:
        common_pids &= set(components_dict[comp_name].keys())

    common_pids = sorted(list(common_pids))  # Sort for reproducibility
    n_trajectories = len(common_pids)

    if n_trajectories == 0:
        return None, None, 0

    # Build X: IRL reward means (n_trajectories, 1)
    X = np.zeros((n_trajectories, 1))
    for i, pid in enumerate(common_pids):
        irl_r = irl_rewards[pid]
        X[i, 0] = sum(irl_r) / len(irl_r) if len(irl_r) > 0 else 0.0

    # Build Y: Component means (n_trajectories, 6)
    Y = np.zeros((n_trajectories, len(REWARD_COMPONENTS)))
    for j, comp_name in enumerate(REWARD_COMPONENTS):
        for i, pid in enumerate(common_pids):
            comp_r = components_dict[comp_name][pid]
            Y[i, j] = sum(comp_r) / len(comp_r) if len(comp_r) > 0 else 0.0

    return X, Y, n_trajectories


def compute_cca(X: np.ndarray, Y: np.ndarray, n_components: int = 1) -> Dict:
    """
    Compute Canonical Correlation Analysis between X and Y.

    Args:
        X: np.array of shape (n_samples, 1) - IRL reward means
        Y: np.array of shape (n_samples, 6) - Component means
        n_components: Number of canonical components (default 1)

    Returns:
        Dict with CCA results including:
        - canonical_corr: First canonical correlation (rho)
        - x_loadings: Loadings for X
        - y_loadings: Loadings for Y (weights for each component)
        - x_scores: Canonical scores for X
        - y_scores: Canonical scores for Y
    """
    # Check for sufficient samples
    n_samples = X.shape[0]
    if n_samples < 10:
        return {'error': f'Too few samples ({n_samples})'}

    # Check for zero variance in X
    if np.std(X) < 1e-10:
        return {'error': 'Zero variance in IRL rewards'}

    # Standardize the data
    X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
    Y_std = (Y - Y.mean(axis=0)) / (Y.std(axis=0) + 1e-10)

    # Handle constant columns in Y (e.g., base is always 1.0)
    non_constant_cols = np.std(Y, axis=0) > 1e-10
    Y_filtered = Y_std[:, non_constant_cols]

    if Y_filtered.shape[1] == 0:
        return {'error': 'All Y components have zero variance'}

    # Fit CCA
    # n_components is min(X.shape[1], Y_filtered.shape[1])
    actual_n_components = min(n_components, X_std.shape[1], Y_filtered.shape[1])

    try:
        cca = CCA(n_components=actual_n_components, max_iter=1000)
        X_c, Y_c = cca.fit_transform(X_std, Y_filtered)

        # Compute canonical correlation as Pearson correlation between scores
        canonical_corr, _ = pearsonr(X_c[:, 0], Y_c[:, 0])

        # Get loadings (weights)
        x_loadings = cca.x_weights_
        y_loadings_filtered = cca.y_weights_

        # Reconstruct full y_loadings with zeros for constant columns
        y_loadings = np.zeros((len(REWARD_COMPONENTS), actual_n_components))
        y_loadings[non_constant_cols, :] = y_loadings_filtered

        return {
            'canonical_corr': canonical_corr,
            'x_loadings': x_loadings,
            'y_loadings': y_loadings,
            'x_scores': X_c,
            'y_scores': Y_c,
            'n_samples': n_samples,
            'n_components_used': actual_n_components,
            'non_constant_components': [REWARD_COMPONENTS[i] for i in range(len(REWARD_COMPONENTS)) if non_constant_cols[i]],
        }

    except Exception as e:
        return {'error': str(e)}


def run_cca_for_model(
    model_name: str,
    irl_rewards: Dict[int, List[float]],
    components_dict: Dict[str, Dict[int, List[float]]]
) -> Dict:
    """
    Run CCA for a single IRL model.

    Args:
        model_name: Name of the IRL model
        irl_rewards: IRL rewards dict
        components_dict: Clinician components dict

    Returns:
        Dict with CCA results
    """
    # Prepare data
    X, Y, n_trajectories = prepare_cca_data(irl_rewards, components_dict)

    if X is None or n_trajectories == 0:
        return {'error': 'No common trajectories'}

    # Run CCA
    result = compute_cca(X, Y)
    result['n_trajectories'] = n_trajectories

    return result


# ============================================================================
# Table Generation Functions
# ============================================================================

def generate_cca_comparison_table(results: Dict[str, Dict]) -> str:
    """
    Generate ASCII table comparing canonical correlations across IRL models.

    Args:
        results: Dict mapping model_name -> CCA results
    """
    header = (
        f"{'IRL Model':<18} | "
        f"{'Canon. Corr (ρ)':>15} {'N Trajectories':>15}"
    )
    separator = "-" * len(header)

    lines = [
        "=" * len(header),
        "CANONICAL CORRELATION ANALYSIS",
        "X: IRL Reward (1D)  |  Y: Clinician Components (6D)",
        "=" * len(header),
        header,
        separator,
    ]

    for model_name in IRL_MODELS:
        display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        r = results.get(model_name, {})

        if 'error' in r:
            lines.append(f"{display_name:<18} | {'N/A':>15} {r.get('error', 'Error'):>15}")
        else:
            rho = r.get('canonical_corr', 0)
            n_traj = r.get('n_trajectories', 0)
            lines.append(f"{display_name:<18} | {rho:>15.4f} {n_traj:>15}")

    lines.append(separator)

    return "\n".join(lines)


def generate_component_loadings_table(results: Dict[str, Dict]) -> str:
    """
    Generate ASCII table showing Y loadings (component weights) for each IRL model.

    Args:
        results: Dict mapping model_name -> CCA results
    """
    # Header with all IRL models
    model_headers = "  ".join([f"{MODEL_DISPLAY_NAMES.get(m, m):>14}" for m in IRL_MODELS])
    header = f"{'Component (λ)':^18} | {model_headers}"
    separator = "-" * len(header)

    lines = [
        "=" * len(header),
        "CCA COMPONENT COEFFICIENTS (λ): X* = λ₁Y₁ + λ₂Y₂ + ... + λ₆Y₆",
        "=" * len(header),
        header,
        separator,
    ]

    for i, comp_name in enumerate(REWARD_COMPONENTS):
        comp_display = COMPONENT_DISPLAY_NAMES.get(comp_name, comp_name)

        values = []
        for model_name in IRL_MODELS:
            r = results.get(model_name, {})
            if 'error' in r or 'y_loadings' not in r:
                values.append(f"{'N/A':>14}")
            else:
                loading = r['y_loadings'][i, 0] if r['y_loadings'].shape[0] > i else 0
                values.append(f"{loading:>14.4f}")

        lines.append(f"λ_{i+1} ({comp_display:<12}) | {'  '.join(values)}")

    lines.append(separator)
    lines.append("")
    lines.append("Note: X* is the canonical variate of IRL reward X that maximally correlates")
    lines.append("      with the linear combination of clinician components Y.")

    return "\n".join(lines)


def generate_canonical_pairs_detail(results: Dict[str, Dict]) -> str:
    """
    Generate detailed output for each IRL model showing the canonical pair
    with ρ and the equation X* = λ₁Y₁ + λ₂Y₂ + ...

    Args:
        results: Dict mapping model_name -> CCA results
    """
    lines = [
        "=" * 80,
        "CANONICAL PAIRS DETAIL",
        "First Canonical Pair: X* = λ₁·Y₁ + λ₂·Y₂ + λ₃·Y₃ + λ₄·Y₄ + λ₅·Y₅ + λ₆·Y₆",
        "(Note: Since X is 1-dimensional, only 1 canonical pair exists)",
        "=" * 80,
    ]

    # Component short names for equation
    comp_short = ['Base', 'Lact', 'MBP', 'SOFA', 'Norepi', 'Mort']

    for model_name in IRL_MODELS:
        display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        r = results.get(model_name, {})

        lines.append("")
        lines.append(f"{'─' * 80}")
        lines.append(f"IRL Model: {display_name}")
        lines.append(f"{'─' * 80}")

        if 'error' in r:
            lines.append(f"  ERROR: {r['error']}")
            continue

        rho = r.get('canonical_corr', 0)
        n_traj = r.get('n_trajectories', 0)
        y_loadings = r.get('y_loadings', None)

        lines.append(f"  Canonical Correlation (ρ): {rho:.4f}")
        lines.append(f"  N Trajectories: {n_traj}")
        lines.append("")

        if y_loadings is not None:
            # Build the equation string
            lines.append("  Canonical Pair Equation:")

            equation_parts = []
            for i, comp_name in enumerate(REWARD_COMPONENTS):
                lam = y_loadings[i, 0] if y_loadings.shape[0] > i else 0
                comp_short_name = comp_short[i] if i < len(comp_short) else comp_name[:4]

                if lam >= 0:
                    sign = " + " if i > 0 else ""
                else:
                    sign = " - " if i > 0 else "-"
                    lam = abs(lam)

                equation_parts.append(f"{sign}{lam:.4f}·{comp_short_name}")

            # Print equation in a readable format
            eq_line = "  X* =" + "".join(equation_parts)
            lines.append(eq_line)

            lines.append("")
            lines.append("  Coefficients (λ):")
            for i, comp_name in enumerate(REWARD_COMPONENTS):
                comp_display = COMPONENT_DISPLAY_NAMES.get(comp_name, comp_name)
                lam = y_loadings[i, 0] if y_loadings.shape[0] > i else 0
                lines.append(f"    λ_{i+1} ({comp_display:<16}): {lam:>8.4f}")

    lines.append("")
    lines.append("=" * 80)

    return "\n".join(lines)


def generate_cca_latex_table(results: Dict[str, Dict], output_path: str, dataset_name: str = "test"):
    """
    Generate LaTeX table for CCA results.
    """
    n_models = len(IRL_MODELS)

    latex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Canonical Correlation Analysis: IRL Rewards vs Clinician Components}",
        r"\label{tab:cca_results}",
        r"\begin{tabular}{l|cc}",
        r"\hline",
        r"\textbf{IRL Method} & \textbf{Canonical Corr. ($\rho$)} & \textbf{N Trajectories} \\",
        r"\hline",
    ]

    # Find the best model (highest rho)
    best_rho = -1
    best_model = None
    for model_name in IRL_MODELS:
        r = results.get(model_name, {})
        if 'error' not in r:
            rho = r.get('canonical_corr', 0)
            if rho > best_rho:
                best_rho = rho
                best_model = model_name

    for model_name in IRL_MODELS:
        display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        r = results.get(model_name, {})

        if 'error' in r:
            latex_lines.append(f"{display_name} & -- & -- \\\\")
        else:
            rho = r.get('canonical_corr', 0)
            n_traj = r.get('n_trajectories', 0)

            # Bold the best result
            if model_name == best_model:
                latex_lines.append(f"\\textbf{{{display_name}}} & \\textbf{{{rho:.4f}}} & {n_traj} \\\\")
            else:
                latex_lines.append(f"{display_name} & {rho:.4f} & {n_traj} \\\\")

    latex_lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\vspace{2mm}",
        r"",
        r"\footnotesize",
        f"\\textit{{Dataset: {dataset_name}. X: IRL reward (1D, trajectory-level mean). Y: 6 clinician components.}}\\\\",
        r"\textit{Canonical correlation measures alignment between IRL reward and optimal linear combination of clinician components.}\\",
        r"\textit{Bold indicates highest canonical correlation.}",
        r"\end{table}",
    ])

    with open(output_path, 'w') as f:
        f.write("\n".join(latex_lines))

    print(f"CCA LaTeX table saved to: {output_path}")


def generate_loadings_latex_table(results: Dict[str, Dict], output_path: str, dataset_name: str = "test"):
    """
    Generate LaTeX table for component loadings.
    """
    n_models = len(IRL_MODELS)

    latex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{CCA Component Loadings (Y weights for first canonical variate)}",
        r"\label{tab:cca_loadings}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{l|" + "c" * n_models + "}",
        r"\hline",
    ]

    # Header row
    model_headers = " & ".join([MODEL_DISPLAY_NAMES.get(m, m) for m in IRL_MODELS])
    latex_lines.append(f"\\textbf{{Component}} & {model_headers} \\\\")
    latex_lines.append(r"\hline")

    # Data rows
    for i, comp_name in enumerate(REWARD_COMPONENTS):
        comp_display = COMPONENT_DISPLAY_NAMES.get(comp_name, comp_name)

        values = []
        for model_name in IRL_MODELS:
            r = results.get(model_name, {})
            if 'error' in r or 'y_loadings' not in r:
                values.append('--')
            else:
                loading = r['y_loadings'][i, 0] if r['y_loadings'].shape[0] > i else 0
                values.append(f"{loading:.3f}")

        latex_lines.append(f"{comp_display} & {' & '.join(values)} \\\\")

    latex_lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"}",
        r"\vspace{2mm}",
        r"",
        r"\footnotesize",
        f"\\textit{{Dataset: {dataset_name}. Loadings show contribution of each clinician component to the canonical variate.}}",
        r"\end{table}",
    ])

    with open(output_path, 'w') as f:
        f.write("\n".join(latex_lines))

    print(f"CCA loadings LaTeX table saved to: {output_path}")


def generate_canonical_pairs_latex_table(results: Dict[str, Dict], output_path: str, dataset_name: str = "test"):
    """
    Generate LaTeX table showing canonical pairs with ρ and equation coefficients.
    """
    # Component symbols for equation
    comp_symbols = [r'Y_{\text{base}}', r'Y_{\text{lact}}', r'Y_{\text{mbp}}',
                    r'Y_{\text{sofa}}', r'Y_{\text{norepi}}', r'Y_{\text{mort}}']

    latex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Canonical Correlation Analysis: First Canonical Pair}",
        r"\label{tab:cca_canonical_pairs}",
        r"\begin{tabular}{l|c|l}",
        r"\hline",
        r"\textbf{IRL Method} & \textbf{$\rho$} & \textbf{Canonical Equation: $X^* = \sum_i \lambda_i Y_i$} \\",
        r"\hline",
    ]

    # Find best model
    best_rho = -1
    best_model = None
    for model_name in IRL_MODELS:
        r = results.get(model_name, {})
        if 'error' not in r:
            rho = r.get('canonical_corr', 0)
            if rho > best_rho:
                best_rho = rho
                best_model = model_name

    for model_name in IRL_MODELS:
        display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        r = results.get(model_name, {})

        if 'error' in r:
            latex_lines.append(f"{display_name} & -- & -- \\\\")
            continue

        rho = r.get('canonical_corr', 0)
        y_loadings = r.get('y_loadings', None)

        # Build equation
        if y_loadings is not None:
            eq_parts = []
            for i in range(len(REWARD_COMPONENTS)):
                lam = y_loadings[i, 0] if y_loadings.shape[0] > i else 0
                if abs(lam) < 0.001:
                    continue  # Skip near-zero coefficients

                if lam >= 0:
                    sign = "+" if eq_parts else ""
                else:
                    sign = "-"
                    lam = abs(lam)

                eq_parts.append(f"{sign}{lam:.2f}{comp_symbols[i]}")

            equation = "$" + "".join(eq_parts) + "$" if eq_parts else "--"
        else:
            equation = "--"

        # Bold the best
        if model_name == best_model:
            latex_lines.append(f"\\textbf{{{display_name}}} & \\textbf{{{rho:.4f}}} & {equation} \\\\")
        else:
            latex_lines.append(f"{display_name} & {rho:.4f} & {equation} \\\\")

    latex_lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\vspace{2mm}",
        r"",
        r"\footnotesize",
        f"\\textit{{Dataset: {dataset_name}. $\\rho$: First canonical correlation. $X^*$: Canonical variate of IRL reward.}}\\\\",
        r"\textit{$Y_i$: Clinician reward components (Base, Lactate, MBP, SOFA, Norepinephrine, Mortality).}\\",
        r"\textit{Bold indicates highest canonical correlation.}",
        r"\end{table}",
    ])

    with open(output_path, 'w') as f:
        f.write("\n".join(latex_lines))

    print(f"CCA canonical pairs LaTeX table saved to: {output_path}")


# ============================================================================
# Main Analysis Function
# ============================================================================

def run_cca_analysis():
    """Run full CCA analysis."""

    print("=" * 70)
    print("CANONICAL CORRELATION ANALYSIS (CCA)")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print("X: IRL Reward (1D, trajectory-level mean)")
    print("Y: Clinician Components (6D, trajectory-level means)")
    print("   - Base, Lactate, Blood Pressure, SOFA, Norepinephrine, Mortality")

    # Determine which dataset to use
    if USE_VAL_AND_TEST:
        print("\n** Using COMBINED validation + test data for more robust analysis **")
        rewards_pickle_path = REWARDS_PICKLE_PATH_COMBINED
    else:
        print("\n** Using test data only **")
        rewards_pickle_path = REWARDS_PICKLE_PATH_TEST

    print("=" * 70)

    # =========================================================================
    # Load IRL rewards from pickle
    # =========================================================================
    print(f"\nLoading IRL rewards from: {rewards_pickle_path}")
    rewards_data = load_rewards_from_pickle(rewards_pickle_path)

    # Fallback to test-only if combined doesn't exist
    if rewards_data is None and USE_VAL_AND_TEST:
        print(f"  Combined pickle not found, falling back to test-only...")
        rewards_pickle_path = REWARDS_PICKLE_PATH_TEST
        rewards_data = load_rewards_from_pickle(rewards_pickle_path)

    if rewards_data is None:
        print("ERROR: Could not load rewards pickle file.")
        print("Please run: python extract_irl_rewards.py --include-val")
        return None

    # =========================================================================
    # Load clinician reward components
    # =========================================================================
    components_dict = load_clinician_reward_components(include_val=USE_VAL_AND_TEST)

    # =========================================================================
    # Run CCA for each IRL model
    # =========================================================================
    print("\nRunning CCA for each IRL model...")

    results = {}

    for model_name in IRL_MODELS:
        if model_name not in rewards_data:
            print(f"  {model_name}: NOT FOUND in pickle")
            results[model_name] = {'error': 'Not found in pickle'}
            continue

        irl_rewards = rewards_data[model_name]

        cca_result = run_cca_for_model(model_name, irl_rewards, components_dict)
        results[model_name] = cca_result

        if 'error' in cca_result:
            print(f"  {model_name}: ERROR - {cca_result['error']}")
        else:
            rho = cca_result['canonical_corr']
            n_traj = cca_result['n_trajectories']
            print(f"  {model_name}: ρ = {rho:.4f} (n = {n_traj})")

    # =========================================================================
    # Print tables
    # =========================================================================
    print("\n")
    print(generate_cca_comparison_table(results))
    print("\n")
    print(generate_component_loadings_table(results))
    print("\n")
    print(generate_canonical_pairs_detail(results))

    # =========================================================================
    # Save results
    # =========================================================================
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save pickle
    pickle_path = os.path.join(OUTPUT_DIR, 'cca_results.pkl')
    eval_set = 'val+test' if USE_VAL_AND_TEST else 'test'
    results_data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'random_seed': RANDOM_SEED,
            'eval_set': eval_set,
            'model_paths': MODEL_PATHS,
            'components': REWARD_COMPONENTS,
        },
        'results': results,
    }
    with open(pickle_path, 'wb') as f:
        pickle.dump(results_data, f)
    print(f"\nResults saved to: {pickle_path}")

    # Save LaTeX tables with dataset name
    dataset_label = "Validation + Test" if USE_VAL_AND_TEST else "Test"

    cca_latex_path = os.path.join(OUTPUT_DIR, 'cca_results.tex')
    generate_cca_latex_table(results, cca_latex_path, dataset_label)

    loadings_latex_path = os.path.join(OUTPUT_DIR, 'cca_loadings.tex')
    generate_loadings_latex_table(results, loadings_latex_path, dataset_label)

    pairs_latex_path = os.path.join(OUTPUT_DIR, 'cca_canonical_pairs.tex')
    generate_canonical_pairs_latex_table(results, pairs_latex_path, dataset_label)

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    # Find best model
    best_rho = -1
    best_model = None
    for model_name in IRL_MODELS:
        r = results.get(model_name, {})
        if 'error' not in r:
            rho = r.get('canonical_corr', 0)
            if rho > best_rho:
                best_rho = rho
                best_model = model_name

    if best_model:
        print(f"Best IRL model: {MODEL_DISPLAY_NAMES.get(best_model, best_model)}")
        print(f"Canonical correlation: {best_rho:.4f}")
        print("\nThis model's reward best captures the linear combination of")
        print("clinician reward components.")

    return results


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    run_cca_analysis()
