### import relevant libraries
import torch, pdb
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss
import os
import argparse
import warnings
warnings.filterwarnings('ignore')

# Import project modules
from integrated_data_pipeline_v3 import IntegratedDataPipelineV3

# Parse command line arguments
parser = argparse.ArgumentParser(description='Importance Sampling evaluation for Block Discrete SQIL')
parser.add_argument('--model_path', type=str, required=True,
                   help='Path to trained SQIL model checkpoint')
parser.add_argument('--vp2_bins', type=int, default=5,
                   help='Number of bins for VP2 discretization (default: 5)')
parser.add_argument('--eval_set', type=str, default='test', choices=['val', 'test'],
                   help='Which data split to evaluate on (default: test)')
parser.add_argument('--reward_type', type=str, default='manual', choices=['manual', 'irl'],
                   help='Reward type: manual (clinician-defined) or irl (learned from IRL model)')
parser.add_argument('--irl_model_path', type=str, default=None,
                   help='Path to IRL model for reward computation (required if reward_type=irl)')
parser.add_argument('--combined_or_train_data_path', type=str, default=None,
                   help='Path to training dataset. If eval_data_path is also provided, '
                        'all patients from this dataset are used for training. '
                        'If eval_data_path is None, this dataset is split into train/val/test.')
parser.add_argument('--eval_data_path', type=str, default=None,
                   help='Path to evaluation dataset (for val/test). If provided, enables '
                        'dual-dataset mode where this dataset is split 50/50 into val/test.')
parser.add_argument('--disable_is_clipping', action='store_true',
                   help='Disable percentile-based clipping for IS weights and trajectory multiplied weights')
parser.add_argument('--is_clip_lower_pct', type=float, default=0.5,
                   help='Lower percentile for IS weight clipping (default: 0.5)')
parser.add_argument('--is_clip_upper_pct', type=float, default=99.5,
                   help='Upper percentile for IS weight clipping (default: 99.5)')
args = parser.parse_args()

# Validate arguments
if args.reward_type == 'irl' and args.irl_model_path is None:
    parser.error("--irl_model_path is required when --reward_type=irl")
if not (0.0 <= args.is_clip_lower_pct < args.is_clip_upper_pct <= 100.0):
    parser.error("--is_clip_lower_pct and --is_clip_upper_pct must satisfy 0 <= lower < upper <= 100")

n_bins = args.vp2_bins
model_path = args.model_path
eval_set = args.eval_set
reward_type = args.reward_type
irl_model_path = args.irl_model_path
use_is_clipping = not args.disable_is_clipping

# Define constants
STATE_DIM = 17  # 17 state features for dual model

### define the block discrete model
# - load the trained SQIL model (same Q-network architecture as CQL)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Model parameters
n_actions = 2 * n_bins  # VP1 (2 options: 0,1) x VP2 (n_bins) = total actions
state_dim = STATE_DIM  # 17 features

# Import the Q-network class from the SQIL training script
from run_block_discrete_sqil import DualBlockDiscreteQNetwork

# Initialize Q-networks (we use both Q1 and Q2, then take min)
q1_network = DualBlockDiscreteQNetwork(state_dim=state_dim, vp2_bins=n_bins).to(device)
q2_network = DualBlockDiscreteQNetwork(state_dim=state_dim, vp2_bins=n_bins).to(device)

# Load the trained model checkpoint
checkpoint = torch.load(model_path, map_location=device)
q1_network.load_state_dict(checkpoint['q1_state_dict'])
q2_network.load_state_dict(checkpoint['q2_state_dict'])
q1_network.eval()
q2_network.eval()

print(f"Loaded SQIL model from: {model_path}")
print(f"State dimension: {state_dim}")
print(f"Number of actions: {n_actions} (VP1: 2 binary × VP2: {n_bins} bins)")
print("Using min(Q1, Q2) for action selection (double Q-learning)")
print("Model type: SQIL Feedforward Q-Network")

###
# prepare and load the data with random seed 42

print("\n" + "="*70)
print("LOADING DATA")
print("="*70)
print(f"Evaluation set: {eval_set}")
print(f"Reward type: {reward_type}")
if irl_model_path:
    print(f"IRL model: {irl_model_path}")

# Initialize data pipeline based on reward type
if reward_type == 'irl':
    # Use V3 pipeline with learned rewards
    pipeline = IntegratedDataPipelineV3(
        model_type='dual', reward_source='learned', random_seed=42,
        combined_or_train_data_path=args.combined_or_train_data_path,
        eval_data_path=args.eval_data_path
    )

    # Detect IRL model type from path and load accordingly
    if 'iq_learn' in irl_model_path.lower():
        pipeline.load_iq_learn_reward_model(irl_model_path)
    elif 'gcl' in irl_model_path.lower():
        pipeline.load_gcl_reward_model(irl_model_path)
    elif 'maxent' in irl_model_path.lower():
        pipeline.load_maxent_reward_model(irl_model_path)
    elif 'semi_supervised_unet' in irl_model_path.lower():
        pipeline.load_semi_supervised_unet_reward_model(irl_model_path, vp1_bins=2, vp2_bins=n_bins)
    elif 'unet' in irl_model_path.lower():
        pipeline.load_unet_reward_model(irl_model_path, vp1_bins=2, vp2_bins=n_bins)
    else:
        # Try to infer from file content or default to iq_learn
        print(f"Warning: Could not detect IRL model type from path, trying IQ-Learn...")
        pipeline.load_iq_learn_reward_model(irl_model_path)

    train_data, val_data, test_data = pipeline.prepare_data()
else:
    # Use V3 pipeline with manual rewards (supports dual-dataset mode)
    pipeline = IntegratedDataPipelineV3(
        model_type='dual', reward_source='manual', random_seed=42,
        combined_or_train_data_path=args.combined_or_train_data_path,
        eval_data_path=args.eval_data_path
    )
    train_data, val_data, test_data = pipeline.prepare_data()

# Select evaluation data based on eval_set argument
if eval_set == 'val':
    eval_data = val_data
    eval_set_name = "Validation"
else:
    eval_data = test_data
    eval_set_name = "Test"

# Print data statistics
print(f"\nData splits:")
print(f"  Training:   {len(train_data['states'])} transitions")
print(f"  Validation: {len(val_data['states'])} transitions")
print(f"  Test:       {len(test_data['states'])} transitions")
print(f"  Evaluating: {eval_set_name} ({len(eval_data['states'])} transitions)")

# Check unique patients in each split
train_patients = len(np.unique(train_data['patient_ids']))
val_patients = len(np.unique(val_data['patient_ids']))
test_patients = len(np.unique(test_data['patient_ids']))
eval_patients = len(np.unique(eval_data['patient_ids']))

print(f"\nPatient counts:")
print(f"  Training:   {train_patients} patients")
print(f"  Validation: {val_patients} patients")
print(f"  Test:       {test_patients} patients")
print(f"  Evaluating: {eval_set_name} ({eval_patients} patients)")

# Verify state and action dimensions
print(f"\nData shapes:")
print(f"  States:  {train_data['states'].shape}")
print(f"  Actions: {train_data['actions'].shape}")
print(f"  Rewards: {train_data['rewards'].shape}")
print(f"  Reward type: {reward_type}")

## use the trained model to predict the optimal actions predicted by the model
## for example to get model actions: model_actions = model.select_action(states)
## for example to get clinician actions: clinician_actions = data.actions
## note the resulting action should have both vp1 action and vp2 action

print("\n" + "="*70)
print("GENERATING MODEL ACTIONS")
print("="*70)

# Define VP2 bin edges (exactly as in training script)
vp2_bin_edges = np.linspace(0, 0.5, n_bins + 1)
print(f"VP2 bin edges: {vp2_bin_edges}")

# Helper function to select actions and return DISCRETE indices - Feedforward version
def select_action_batch_discrete(states, q1_net, q2_net, vp2_bins, device):
    """
    Select best actions for a batch of states using min(Q1, Q2)
    Returns: numpy array of discrete action indices [0 to n_actions-1]
    """
    with torch.no_grad():
        if states.ndim == 1:
            states = states.reshape(1, -1)

        batch_size = states.shape[0]
        state_tensor = torch.FloatTensor(states).to(device)

        # Create all possible discrete actions for each state in batch
        total_actions = 2 * vp2_bins
        all_actions = torch.arange(total_actions).to(device)
        all_actions = all_actions.unsqueeze(0).expand(batch_size, -1)

        # Expand states to match actions
        state_expanded = state_tensor.unsqueeze(1).expand(-1, total_actions, -1)
        state_expanded = state_expanded.reshape(-1, state_dim)
        actions_flat = all_actions.reshape(-1)

        # Compute Q-values for all actions
        q1_values = q1_net(state_expanded, actions_flat).reshape(batch_size, total_actions)
        q2_values = q2_net(state_expanded, actions_flat).reshape(batch_size, total_actions)
        q_values = torch.min(q1_values, q2_values)

        # Get best action indices
        best_action_indices = q_values.argmax(dim=1).cpu().numpy()

        return best_action_indices

# Helper function to convert continuous actions to discrete indices
def continuous_to_discrete_action(actions, vp2_edges, vp2_bins):
    """
    Convert continuous actions [vp1, vp2] to discrete action indices
    action_idx = vp1 * n_bins + vp2_bin
    """
    vp1 = actions[:, 0].astype(int)  # Binary 0 or 1
    vp2 = actions[:, 1].clip(0, 0.5)

    # Convert VP2 continuous to bin index
    vp2_bins_idx = np.digitize(vp2, vp2_edges) - 1
    vp2_bins_idx = np.clip(vp2_bins_idx, 0, len(vp2_edges) - 2)

    # Combine: action_idx = vp1 * n_bins + vp2_bin
    action_indices = vp1 * vp2_bins + vp2_bins_idx
    return action_indices

# Get model actions as discrete indices
print("Computing model actions (discrete) for training data...")
train_model_actions_discrete = select_action_batch_discrete(train_data['states'], q1_network, q2_network, n_bins, device)

print(f"Computing model actions (discrete) for {eval_set_name.lower()} data...")
eval_model_actions_discrete = select_action_batch_discrete(eval_data['states'], q1_network, q2_network, n_bins, device)

# Convert clinician continuous actions to discrete indices
train_clinician_actions_discrete = continuous_to_discrete_action(train_data['actions'], vp2_bin_edges, n_bins)
eval_clinician_actions_discrete = continuous_to_discrete_action(eval_data['actions'], vp2_bin_edges, n_bins)

print(f"\nDiscrete action indices generated:")
print(f"  Train model actions:     {train_model_actions_discrete.shape}")
print(f"  Train clinician actions: {train_clinician_actions_discrete.shape}")
print(f"  {eval_set_name} model actions:      {eval_model_actions_discrete.shape}")
print(f"  {eval_set_name} clinician actions:  {eval_clinician_actions_discrete.shape}")

# Display sample discrete actions
print(f"\nSample discrete action indices (first 10):")
print(f"  Model (train):     {train_model_actions_discrete[:10]}")
print(f"  Clinician (train): {train_clinician_actions_discrete[:10]}")

# Show action distribution
print(f"\nAction distribution (0-{n_actions-1}):")
for action_idx in range(n_actions):
    print(f"  Action {action_idx}: Model (train)={np.sum(train_model_actions_discrete==action_idx)}, "
          f"Clinician (train)={np.sum(train_clinician_actions_discrete==action_idx)}, "
          f"Model ({eval_set_name.lower()})={np.sum(eval_model_actions_discrete==action_idx)}, "
          f"Clinician ({eval_set_name.lower()})={np.sum(eval_clinician_actions_discrete==action_idx)}")

## train a logistic classifier (lg_vp1_ma) to predict the vp1 model actions (binary) using the states on the training data, then use it on the test data to get the probability of clinician vp1 actions
## train a logistic classifier (lg_vp1_ca) to predict the vp1 clinician actions (binary) using the states on the training data, then use it on the test data to get the probability of clinician vp1 actions
## train a softmax classifier (sm_vp2_ma) to predict the vp2 model actions (block discrete) using the states on the training data, then use it on the test data to get the probability of clinician vp2 actions
## train a softmax classifier (sm_vp2_ca) to predict the vp2 clinician (block discrete) using the states on the training data, then use it on the test data to get the probability of clinician vp2 actions
## save the test data probabilities, and display some of them (each of the four)

print("\n" + "="*70)
print("TRAINING BEHAVIOR POLICY CLASSIFIERS (JOINT ACTION SPACE)")
print("="*70)

# Helper function to get probabilities for all n_actions classes
# even if classifier didn't see all classes during training
def get_full_proba(clf, X, n_classes):
    """
    Get probability predictions ensuring output has n_classes columns.
    If classifier didn't see some classes, assign them probability 0.
    """
    proba = clf.predict_proba(X)
    if proba.shape[1] == n_classes:
        return proba

    # Map classifier's classes to full class set
    full_proba = np.zeros((X.shape[0], n_classes))
    for i, c in enumerate(clf.classes_):
        full_proba[:, int(c)] = proba[:, i]

    # Renormalize rows to sum to 1 (add small epsilon to missing classes)
    # This prevents division by zero in importance sampling
    missing_classes = set(range(n_classes)) - set(clf.classes_.astype(int))
    if missing_classes:
        print(f"   Warning: Classifier missing classes {missing_classes}, assigning small probability")
        for c in missing_classes:
            full_proba[:, c] = 1e-10
        full_proba = full_proba / full_proba.sum(axis=1, keepdims=True)

    return full_proba

# Train softmax classifier for model actions (0 to n_actions-1)
print(f"\n1. Training softmax classifier for model actions ({n_actions} classes)...")
clf_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42)
clf_model.fit(train_data['states'], train_model_actions_discrete)
print(f"   Classifier learned classes: {sorted(clf_model.classes_)}")
train_acc_model = accuracy_score(train_model_actions_discrete, clf_model.predict(train_data['states']))
eval_acc_model = accuracy_score(eval_model_actions_discrete, clf_model.predict(eval_data['states']))
print(f"   Train accuracy: {train_acc_model:.4f}")
print(f"   {eval_set_name} accuracy:  {eval_acc_model:.4f}")

# Train softmax classifier for clinician actions (0 to n_actions-1)
print(f"\n2. Training softmax classifier for clinician actions ({n_actions} classes)...")
clf_clinician = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42)
clf_clinician.fit(train_data['states'], train_clinician_actions_discrete)
print(f"   Classifier learned classes: {sorted(clf_clinician.classes_)}")
train_acc_clinician = accuracy_score(train_clinician_actions_discrete, clf_clinician.predict(train_data['states']))
eval_acc_clinician = accuracy_score(eval_clinician_actions_discrete, clf_clinician.predict(eval_data['states']))
print(f"   Train accuracy: {train_acc_clinician:.4f}")
print(f"   {eval_set_name} accuracy:  {eval_acc_clinician:.4f}")

# Get probabilities on eval data (ensuring all n_actions classes are present)
print("\n" + "="*70)
print("COMPUTING BEHAVIOR POLICY PROBABILITIES")
print("="*70)

# Model policy: π_model(a_clinician | s)
eval_probs_model = get_full_proba(clf_model, eval_data['states'], n_actions)
eval_prob_model = eval_probs_model[np.arange(len(eval_clinician_actions_discrete)), eval_clinician_actions_discrete.astype(int)]

# Clinician policy: π_clinician(a_clinician | s)
eval_probs_clinician = get_full_proba(clf_clinician, eval_data['states'], n_actions)
eval_prob_clinician = eval_probs_clinician[np.arange(len(eval_clinician_actions_discrete)), eval_clinician_actions_discrete.astype(int)]

print(f"\n{eval_set_name} probability statistics:")
print(f"  Model policy π(a_clinician|s)     - Mean: {eval_prob_model.mean():.4f}, Std: {eval_prob_model.std():.4f}, Min: {eval_prob_model.min():.4f}, Max: {eval_prob_model.max():.4f}")
print(f"  Clinician policy π(a_clinician|s) - Mean: {eval_prob_clinician.mean():.4f}, Std: {eval_prob_clinician.std():.4f}, Min: {eval_prob_clinician.min():.4f}, Max: {eval_prob_clinician.max():.4f}")

print(f"\nSample probabilities (first 10 {eval_set_name.lower()} transitions):")
print(f"  Index | Model π | Clinician π | Ratio")
print(f"  " + "-"*50)
for i in range(min(10, len(eval_prob_model))):
    ratio = eval_prob_model[i] / (eval_prob_clinician[i] + 1e-10)
    print(f"  {i:5d} | {eval_prob_model[i]:.4f} | {eval_prob_clinician[i]:.4f} | {ratio:.4f}")


# Finally for each of the transitions (indices) in the test data, compute pi_lg_vp1_ma(s) / (eps + pi_lg_vp1_ca(s)) * pi_sm_vp2_ma(s) / (esp + pi_sm_vp2_ca(s)) * R where R is the reward on that transition index or state, this is the expected reward using the model actions on that state
# then compute the per transition average of the expected reward (using the model recommended actions), as well as the per patient average of the expected reward (using the model recommended actions)
# compare the two values with the average reward (the raw reward) per transition, and the average reward per patient
# print out the results and save it in latex table in a separate latex file

print("\n" + "="*70)
print("COMPUTING IMPORTANCE SAMPLING WEIGHTS AND REWARDS")
print("="*70)

# Set epsilon for numerical stability
eps = 1e-10

# Compute importance sampling weights (single ratio, no chain rule!)
# IS_weight = π_model(a_clinician|s) / π_clinician(a_clinician|s)
is_weight = eval_prob_model / (eval_prob_clinician + eps)

if use_is_clipping:
    isw_ci_diff_lower = np.percentile(is_weight, args.is_clip_lower_pct)
    isw_ci_diff_upper = np.percentile(is_weight, args.is_clip_upper_pct)
    is_weight = np.clip(is_weight, a_min=isw_ci_diff_lower, a_max=isw_ci_diff_upper)
    print(f"  IS clipping enabled with percentiles [{args.is_clip_lower_pct}, {args.is_clip_upper_pct}]")
    print(f"  IS clipping bounds: [{isw_ci_diff_lower:.6f}, {isw_ci_diff_upper:.6f}]")
else:
    isw_ci_diff_lower = None
    isw_ci_diff_upper = None
    print("  IS clipping disabled")

#is_weight = np.cumprod(is_weight)

print(f"\nIS weight statistics:")
print(f"  Mean: {is_weight.mean():.4f}, Std: {is_weight.std():.4f}, Min: {is_weight.min():.4f}, Max: {is_weight.max():.4f}")

# Get eval rewards
eval_rewards = eval_data['rewards']

# Compute IS-weighted rewards
is_weighted_rewards = is_weight * eval_rewards

print(f"\nReward statistics ({reward_type}):")
print(f"  Raw rewards (clinician)     - Mean: {eval_rewards.mean():.4f}, Std: {eval_rewards.std():.4f}, Sum: {eval_rewards.sum():.2f}")
print(f"  IS-weighted rewards (model) - Mean: {is_weighted_rewards.mean():.4f}, Std: {is_weighted_rewards.std():.4f}, Sum: {is_weighted_rewards.sum():.2f}")

# Per-transition average
avg_raw_reward_per_transition = eval_rewards.mean()
avg_is_reward_per_transition = is_weighted_rewards.mean()

print(f"\nPer-transition averages:")
print(f"  Clinician policy (raw):     {avg_raw_reward_per_transition:.4f}")
print(f"  Model policy (IS-weighted): {avg_is_reward_per_transition:.4f}")
print(f"  Difference:                 {avg_is_reward_per_transition - avg_raw_reward_per_transition:.4f}")

# Per-patient average (efficient computation)
eval_patient_ids = eval_data['patient_ids']
unique_patients = np.unique(eval_patient_ids)
n_patients = len(unique_patients)

avg_raw_reward_per_patient = eval_rewards.sum() / n_patients
avg_is_reward_per_patient = is_weighted_rewards.sum() / n_patients

print(f"\nPer-patient averages (efficient computation):")
print(f"  Clinician policy (raw):     {avg_raw_reward_per_patient:.4f}")
print(f"  Model policy (IS-weighted): {avg_is_reward_per_patient:.4f}")
print(f"  Difference:                 {avg_is_reward_per_patient - avg_raw_reward_per_patient:.4f}")

# Compute per-patient statistics (std, min, max) - requires loop
print(f"\nComputing per-patient statistics...")
patient_raw_rewards = []
patient_raw_rewards_ave = []

patient_is_rewards = []

for patient_id in unique_patients:
    patient_mask = eval_patient_ids == patient_id
    patient_raw_reward = eval_rewards[patient_mask].sum()
    patient_is_reward = is_weighted_rewards[patient_mask].sum()
    patient_raw_rewards.append(patient_raw_reward)
    patient_raw_rewards_ave.append(patient_raw_reward/len(eval_rewards[patient_mask]))
    patient_is_rewards.append(patient_is_reward)

patient_raw_rewards = np.array(patient_raw_rewards)
patient_raw_rewards_ave = np.array(patient_raw_rewards_ave)
patient_is_rewards = np.array(patient_is_rewards)

print(f"\nPer-patient reward statistics:")
print(f"  Raw (clinician):    Mean: {patient_raw_rewards.mean():.4f}, Std: {patient_raw_rewards.std():.4f}, Min: {patient_raw_rewards.min():.2f}, Max: {patient_raw_rewards.max():.2f}")
print(f"  IS (model):         Mean: {patient_is_rewards.mean():.4f}, Std: {patient_is_rewards.std():.4f}, Min: {patient_is_rewards.min():.2f}, Max: {patient_is_rewards.max():.2f}")

# Create latex directory
os.makedirs('latex', exist_ok=True)

# Save results to LaTeX table
latex_output = r"""\begin{table}[h]
\centering
\caption{Importance Sampling Off-Policy Evaluation Results (Block Discrete SQIL)}
\label{tab:is_ope_results_sqil}
\begin{tabular}{lcc}
\hline
\textbf{Metric} & \textbf{Clinician Policy} & \textbf{Model Policy (IS)} \\
\hline
Per-Transition Avg Reward & %.4f & %.4f \\
Per-Patient Avg Reward & %.4f & %.4f \\
\hline
Per-Patient Reward Std & %.4f & %.4f \\
Per-Patient Reward Min & %.2f & %.2f \\
Per-Patient Reward Max & %.2f & %.2f \\
\hline
Number of Patients & %d & %d \\
Number of Transitions & %d & %d \\
\hline
\multicolumn{3}{l}{\textbf{Improvement (Model - Clinician)}} \\
\hline
Per-Transition & \multicolumn{2}{c}{%.4f} \\
Per-Patient & \multicolumn{2}{c}{%.4f} \\
\hline
\end{tabular}
\end{table}
""" % (
    avg_raw_reward_per_transition, avg_is_reward_per_transition,
    avg_raw_reward_per_patient, avg_is_reward_per_patient,
    patient_raw_rewards.std(), patient_is_rewards.std(),
    patient_raw_rewards.min(), patient_is_rewards.min(),
    patient_raw_rewards.max(), patient_is_rewards.max(),
    n_patients, n_patients,
    len(eval_rewards), len(eval_rewards),
    avg_is_reward_per_transition - avg_raw_reward_per_transition,
    avg_is_reward_per_patient - avg_raw_reward_per_patient
)

latex_file = 'latex/is_sqil_ope_results.tex'
with open(latex_file, 'w') as f:
    f.write(latex_output)

print(f"\n" + "="*70)
print(f"Results saved to: {latex_file}")
print("="*70)

# Weighted Importance Sampling (WIS) - Per-Transition Level
print("\n" + "="*70)
print("WEIGHTED IMPORTANCE SAMPLING (WIS) - PER-TRANSITION")
print("="*70)

# V_WIS = Σᵢ [wᵢ · rᵢ] / Σᵢ wᵢ
# where wᵢ is the IS weight for transition i, rᵢ is the reward for transition i

# Sum of weighted rewards
sum_weighted_rewards = (is_weight * eval_rewards).sum()

# Sum of weights
sum_weights = is_weight.sum()

# WIS estimate
wis_per_transition = sum_weighted_rewards / sum_weights if sum_weights > 0 else 0.0

# Standard IS (for comparison)
is_per_transition = (is_weight * eval_rewards).mean()

# Clinician average (raw)
clinician_per_transition = eval_rewards.mean()

print(f"\nPer-transition evaluation:")
print(f"  Clinician policy (raw):        {clinician_per_transition:.4f}")
print(f"  Model policy (standard IS):    {is_per_transition:.4f}")
print(f"  Model policy (weighted IS):    {wis_per_transition:.4f}")

print(f"\nImprovement over clinician:")
print(f"  Standard IS:  {is_per_transition - clinician_per_transition:.4f}")
print(f"  Weighted IS:  {wis_per_transition - clinician_per_transition:.4f}")

print(f"\nWeight normalization:")
print(f"  Sum of weights: {sum_weights:.4f}")
print(f"  Number of transitions: {len(eval_rewards)}")
print(f"  Average weight: {sum_weights / len(eval_rewards):.4f}")

# (4) Weighted Importance Sampling (WIS) - Per-Trajectory Level
print("\n" + "="*70)
print("WEIGHTED IMPORTANCE SAMPLING (WIS) - PER-TRAJECTORY")
print("="*70)

# For each patient trajectory: WIS_τ = Σᵢ(wᵢ·rᵢ) / Σᵢ(wᵢ) for transitions in that trajectory
# Then average across all patients

# METHOD 1
weights_per_trajectory_list = []
total_rewards_per_trajectory_list = []
weighted_rewards_per_trajectory_list = []

for patient_id in unique_patients:
    patient_mask = eval_patient_ids == patient_id

    # Get weights and rewards for this trajectory
    patient_weights = is_weight[patient_mask]
    patient_rewards = eval_rewards[patient_mask]
    weights_per_trajectory_list.append(patient_weights.mean())

    patient_weights = np.cumprod(patient_weights)
    if use_is_clipping:
        patient_weights = np.clip(patient_weights, a_min=isw_ci_diff_lower, a_max=isw_ci_diff_upper)

    est_total_reward_per_traj = (patient_weights *  patient_rewards).sum() / patient_weights.sum() * len(patient_weights)

    total_rewards_per_trajectory_list.append(patient_rewards.sum())
    weighted_rewards_per_trajectory_list.append(est_total_reward_per_traj)

# Compute mean WIS across all trajectories
weights_per_trajectory_list = np.array(weights_per_trajectory_list)
total_rewards_per_trajectory_list = np.array(total_rewards_per_trajectory_list)
weighted_rewards_per_trajectory_list = np.array(weighted_rewards_per_trajectory_list)

wis_trajectory_level = (weights_per_trajectory_list * weighted_rewards_per_trajectory_list).sum() / weights_per_trajectory_list.sum()

# Also compute raw clinician per-trajectory for comparison
clinician_per_trajectory_mean = total_rewards_per_trajectory_list.mean()

# Compute 95% confidence interval using bootstrapping for the DIFFERENCE
print(f"\nComputing 95% CI for difference using bootstrapping (1000 iterations)...")

# First, compute the difference for each trajectory
# For clinician: just the raw reward sum per patient
# For model: WIS per trajectory (already computed)
# Difference: s = WIS_model - raw_clinician for each patient

# Bootstrap from the bag of differences
n_bootstrap = 1000
bootstrap_differences = []

np.random.seed(42)  # For reproducibility
for _ in range(n_bootstrap):
    # Resample differences with replacement
    bootstrap_indices = np.random.choice(len(total_rewards_per_trajectory_list), size=len(total_rewards_per_trajectory_list), replace=True)

    bootstrap_weights = weights_per_trajectory_list[bootstrap_indices]
    bootstrap_rewards = total_rewards_per_trajectory_list[bootstrap_indices]
    weighted_bootstrap_rewards = weighted_rewards_per_trajectory_list[bootstrap_indices]

    if np.sum(bootstrap_weights) > 0:
        bootstrap_wis = np.sum(bootstrap_weights * weighted_bootstrap_rewards) / np.sum(bootstrap_weights)
    else:
        bootstrap_wis = 0

    bootstrap_behavior = np.mean(bootstrap_rewards)
    bootstrap_diff = bootstrap_wis - bootstrap_behavior
    bootstrap_differences.append(bootstrap_diff)

bootstrap_differences = np.array(bootstrap_differences)

# Compute 95% CI using percentile method
ci_diff_lower = np.percentile(bootstrap_differences, 2.5)
ci_diff_upper = np.percentile(bootstrap_differences, 97.5)

# Mean difference
improvement_mean = wis_trajectory_level - clinician_per_trajectory_mean

print(f"\nPer-trajectory WIS evaluation with 95% CI:")
print(f"  Clinician policy (raw):        {clinician_per_trajectory_mean:.4f}")
print(f"  Model policy (WIS):            {wis_trajectory_level:.4f}")
print(f"  Difference (Model - Clinician): {improvement_mean:.4f} (95% CI: [{ci_diff_lower:.4f}, {ci_diff_upper:.4f}])")

# =============================================================================
# IMPORTANCE SAMPLING DIAGNOSTICS
# (i)   Effective Sample Size (ESS) as fraction of total N
# (ii)  Histograms of normalized importance weight distributions
# (iii) Maximum weight ratio  max_j w_j / sum_j w_j
# References: Thomas et al. (2015); Precup et al. (2000)
# =============================================================================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Extract model name from checkpoint path for output file naming
model_name = os.path.splitext(os.path.basename(model_path))[0]

print("\n" + "="*70)
print(f"IMPORTANCE SAMPLING DIAGNOSTICS (model: {model_name})")
print("="*70)

# --- (i) Effective Sample Size ---
# Kish's ESS = (sum w_i)^2 / sum(w_i^2)
ess_transition = is_weight.sum()**2 / (is_weight**2).sum()
ess_transition_frac = ess_transition / len(is_weight)

ess_trajectory = weights_per_trajectory_list.sum()**2 / (weights_per_trajectory_list**2).sum()
ess_trajectory_frac = ess_trajectory / len(weights_per_trajectory_list)

print(f"\n(i) Effective Sample Size (ESS):")
print(f"  Per-transition level:")
print(f"    ESS:            {ess_transition:.2f}")
print(f"    Total N:        {len(is_weight)}")
print(f"    ESS / N:        {ess_transition_frac:.4f} ({ess_transition_frac*100:.2f}%)")
print(f"  Per-trajectory level:")
print(f"    ESS:            {ess_trajectory:.2f}")
print(f"    Total N:        {len(weights_per_trajectory_list)}")
print(f"    ESS / N:        {ess_trajectory_frac:.4f} ({ess_trajectory_frac*100:.2f}%)")

# --- (iii) Maximum Weight Ratio ---
max_weight_ratio_transition = is_weight.max() / is_weight.sum()
max_weight_ratio_trajectory = weights_per_trajectory_list.max() / weights_per_trajectory_list.sum()

print(f"\n(iii) Maximum Weight Ratio (single-sample dominance):")
print(f"  Per-transition:   max(w)/sum(w) = {max_weight_ratio_transition:.6f} ({max_weight_ratio_transition*100:.4f}%)")
print(f"  Per-trajectory:   max(w)/sum(w) = {max_weight_ratio_trajectory:.6f} ({max_weight_ratio_trajectory*100:.4f}%)")

# --- (ii) Histograms of Normalized Importance Weight Distributions ---
norm_weights_transition = is_weight / is_weight.sum()
norm_weights_trajectory = weights_per_trajectory_list / weights_per_trajectory_list.sum()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(norm_weights_transition, bins=50, edgecolor='black', alpha=0.7)
axes[0].axvline(1.0 / len(norm_weights_transition), color='red', linestyle='--',
                label=f'Uniform = {1.0/len(norm_weights_transition):.2e}')
axes[0].set_xlabel('Normalized IS Weight')
axes[0].set_ylabel('Count')
axes[0].set_title(f'Per-Transition Normalized Weights\n'
                  f'ESS/N={ess_transition_frac:.4f}, '
                  f'max(w)/sum(w)={max_weight_ratio_transition:.4e}')
axes[0].legend()
axes[0].set_yscale('log')

axes[1].hist(norm_weights_trajectory, bins=50, edgecolor='black', alpha=0.7, color='orange')
axes[1].axvline(1.0 / len(norm_weights_trajectory), color='red', linestyle='--',
                label=f'Uniform = {1.0/len(norm_weights_trajectory):.2e}')
axes[1].set_xlabel('Normalized IS Weight')
axes[1].set_ylabel('Count')
axes[1].set_title(f'Per-Trajectory Normalized Weights\n'
                  f'ESS/N={ess_trajectory_frac:.4f}, '
                  f'max(w)/sum(w)={max_weight_ratio_trajectory:.4e}')
axes[1].legend()
axes[1].set_yscale('log')

plt.tight_layout()
hist_path = f'latex/is_weight_histograms_{model_name}.png'
plt.savefig(hist_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n(ii) Weight distribution histograms saved to: {hist_path}")

# --- Summary diagnostics table (LaTeX) ---
diagnostics_latex = r"""\begin{table}[h]
\centering
\caption{Importance Sampling Diagnostics - SQIL (%s)}
\label{tab:is_diagnostics_sqil_%s}
\begin{tabular}{lcc}
\hline
\textbf{Diagnostic} & \textbf{Per-Transition} & \textbf{Per-Trajectory} \\
\hline
Effective Sample Size (ESS) & %.2f & %.2f \\
Total $N$ & %d & %d \\
ESS / $N$ & %.4f & %.4f \\
Max weight ratio $\max_j w_j / \sum_j w_j$ & %.6f & %.6f \\
\hline
\end{tabular}
\end{table}
""" % (
    model_name.replace('_', r'\_'), model_name,
    ess_transition, ess_trajectory,
    len(is_weight), len(weights_per_trajectory_list),
    ess_transition_frac, ess_trajectory_frac,
    max_weight_ratio_transition, max_weight_ratio_trajectory
)

diagnostics_latex_file = f'latex/is_diagnostics_{model_name}.tex'
with open(diagnostics_latex_file, 'w') as f:
    f.write(diagnostics_latex)
print(f"Diagnostics LaTeX table saved to: {diagnostics_latex_file}")
