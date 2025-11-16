### import relevant libraries
import torch
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
from integrated_data_pipeline_v2 import IntegratedDataPipelineV2

# ============================================================================
# TASK 1.1: Command-line argument parsing for batch evaluation
# ============================================================================
# Parse command-line arguments to enable running different model configurations
# This allows us to evaluate multiple alpha values and bin sizes without 
# changing the code manually each time
parser = argparse.ArgumentParser(
    description='WIS/OPE Evaluation for Block Discrete CQL',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

# Alpha parameter: CQL conservatism level (0.0 = no penalty, higher = more conservative)
parser.add_argument(
    '--alpha', 
    type=float, 
    default=0.0, 
    help='CQL alpha parameter for conservative Q-learning (0.0, 0.001, 0.01)'
)

# Bins parameter: VP2 discretization granularity (3, 5, or 10 bins)
parser.add_argument(
    '--bins', 
    type=int, 
    default=5, 
    choices=[3, 5, 10], 
    help='Number of VP2 discretization bins'
)

# Output directory for LaTeX tables
parser.add_argument(
    '--output-dir', 
    type=str, 
    default='latex', 
    help='Output directory for LaTeX results'
)

# Parse arguments
args = parser.parse_args()

# Create output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

# Print configuration
print("\n" + "="*70)
print("BLOCK DISCRETE CQL - WIS/OPE EVALUATION")
print("="*70)
print(f"Configuration:")
print(f"  Alpha (CQL conservatism): {args.alpha}")
print(f"  Bins (VP2 discretization): {args.bins}")
print(f"  Output directory: {args.output_dir}")
print("="*70 + "\n")

# Define constants
STATE_DIM = 17  # 17 state features for dual model

# Define bin edges based on number of bins
# This creates evenly spaced bins from 0 to 0.5 mcg/kg/min
if args.bins == 3:
    # 3 bins: [0, 0.167), [0.167, 0.333), [0.333, 0.5]
    ACTION_BINS = [0, 0.167, 0.333, 0.5]
elif args.bins == 5:
    # 5 bins: [0, 0.1), [0.1, 0.2), [0.2, 0.3), [0.3, 0.4), [0.4, 0.5]
    ACTION_BINS = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
elif args.bins == 10:
    # 10 bins: [0, 0.05), [0.05, 0.1), ..., [0.45, 0.5]
    ACTION_BINS = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
else:
    raise ValueError(f"Unsupported bins value: {args.bins}") 

### define the block discrete model
# - load the trained block discrete model based on command-line arguments
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Import the network class from the training script
from run_block_discrete_cql_allalphas import DualBlockDiscreteQNetwork

# Model parameters from command-line arguments
n_bins = args.bins  # Number of VP2 bins (3, 5, or 10)
action_bins = ACTION_BINS  # VP2 discretization bins from above
n_actions = 2 * n_bins  # VP1 (2 options: 0,1) x VP2 (n_bins) = total actions
state_dim = STATE_DIM  # 17 features

# Initialize Q-networks (we use both Q1 and Q2, then take min for double Q-learning)
q1_network = DualBlockDiscreteQNetwork(state_dim=state_dim, vp2_bins=n_bins).to(device)
q2_network = DualBlockDiscreteQNetwork(state_dim=state_dim, vp2_bins=n_bins).to(device)

# ============================================================================
# TASK 1.1: Dynamic model path construction based on alpha and bins
# ============================================================================
# Construct model path from command-line arguments
# Format: block_discrete_cql_alpha{alpha:.4f}_bins{bins}_best.pt
# Example: block_discrete_cql_alpha0.0000_bins5_best.pt
model_path = f'./experiment/block_discrete_cql_alpha{args.alpha:.4f}_bins{args.bins}_best.pt'

# Check if model file exists
if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"Model file not found: {model_path}\n"
        f"Please ensure the model has been trained for alpha={args.alpha} and bins={args.bins}"
    )

# Load the trained model checkpoint
print(f"\nLoading model from: {model_path}")
checkpoint = torch.load(model_path, map_location=device)
q1_network.load_state_dict(checkpoint['q1_state_dict'])
q2_network.load_state_dict(checkpoint['q2_state_dict'])
q1_network.eval()
q2_network.eval()

print(f"✓ Model loaded successfully")
print(f"  State dimension: {state_dim}")
print(f"  Number of actions: {n_actions} (VP1: 2 binary × VP2: {n_bins} bins)")
print(f"  Action bins for VP2: {action_bins}")
print("Using min(Q1, Q2) for action selection (double Q-learning)") 



###
# prepare and load the data with random seed 42 with all the data in a sequence, for example:
# pipeline = IntegratedDataPipelineV2(model_type='dual', random_seed=42)
# train_data, val_data, test_data = pipeline.prepare_data()

print("\n" + "="*70)
print("LOADING DATA")
print("="*70)

# Initialize data pipeline with random seed 42
pipeline = IntegratedDataPipelineV2(model_type='dual', random_seed=42)
train_data, val_data, test_data = pipeline.prepare_data()

# Print data statistics
print(f"\nData splits:")
print(f"  Training:   {len(train_data['states'])} transitions")
print(f"  Validation: {len(val_data['states'])} transitions")
print(f"  Test:       {len(test_data['states'])} transitions")

# Check unique patients in each split
train_patients = len(np.unique(train_data['patient_ids']))
val_patients = len(np.unique(val_data['patient_ids']))
test_patients = len(np.unique(test_data['patient_ids']))

print(f"\nPatient counts:")
print(f"  Training:   {train_patients} patients")
print(f"  Validation: {val_patients} patients")
print(f"  Test:       {test_patients} patients")

# Verify state and action dimensions
print(f"\nData shapes:")
print(f"  States:  {train_data['states'].shape}")
print(f"  Actions: {train_data['actions'].shape}")
print(f"  Rewards: {train_data['rewards'].shape}")


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

# Helper function to select actions using the loaded Q-networks
def select_action_batch(states, q1_net, q2_net, vp2_bins, vp2_edges, device):
    """
    Select best actions for a batch of states using min(Q1, Q2)
    Exactly mirrors the select_action method in DualBlockDiscreteCQL
    Returns: numpy array of shape [batch_size, 2] with [vp1, vp2] actions
    """
    with torch.no_grad():
        if states.ndim == 1:
            states = states.reshape(1, -1)

        batch_size = states.shape[0]
        state_tensor = torch.FloatTensor(states).to(device)

        # Create all possible discrete actions for each state in batch
        # Shape: [batch_size, total_actions]
        total_actions = 2 * vp2_bins
        all_actions = torch.arange(total_actions).to(device)
        all_actions = all_actions.unsqueeze(0).expand(batch_size, -1)

        # Expand states to match actions
        # Shape: [batch_size * total_actions, state_dim]
        state_expanded = state_tensor.unsqueeze(1).expand(-1, total_actions, -1)
        state_expanded = state_expanded.reshape(-1, state_dim)

        # Flatten actions for network input
        # Shape: [batch_size * total_actions]
        actions_flat = all_actions.reshape(-1)

        # Compute Q-values for all actions
        q1_values = q1_net(state_expanded, actions_flat).reshape(batch_size, total_actions)
        q2_values = q2_net(state_expanded, actions_flat).reshape(batch_size, total_actions)
        q_values = torch.min(q1_values, q2_values)

        # Get best action for each batch element
        best_action_indices = q_values.argmax(dim=1).cpu().numpy()

        # Vectorized conversion from discrete indices to continuous actions
        # Extract VP1 (binary) and VP2 bin indices
        vp1_actions = (best_action_indices // vp2_bins).astype(float)
        vp2_bin_indices = best_action_indices % vp2_bins

        # Convert VP2 bin indices to continuous values using bin centers
        vp2_bin_centers = (vp2_edges[:-1] + vp2_edges[1:]) / 2
        vp2_actions = vp2_bin_centers[vp2_bin_indices]

        # Stack VP1 and VP2 into action array
        actions = np.stack([vp1_actions, vp2_actions], axis=1)

        return actions if batch_size > 1 else actions[0]

# Helper function to select actions and return DISCRETE indices (0-9)
def select_action_batch_discrete(states, q1_net, q2_net, vp2_bins, device):
    """
    Select best actions for a batch of states using min(Q1, Q2)
    Returns: numpy array of discrete action indices [0-9]
    """
    with torch.no_grad():
        if states.ndim == 1:
            states = states.reshape(1, -1)

        batch_size = states.shape[0]
        state_tensor = torch.FloatTensor(states).to(device)

        # Create all possible discrete actions for each state in batch
        total_actions = 2 * vp2_bins  # 10 actions
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

        # Get best action indices (0-9)
        best_action_indices = q_values.argmax(dim=1).cpu().numpy()

        return best_action_indices

# Helper function to convert continuous actions to discrete indices
def continuous_to_discrete_action(actions, vp2_edges, vp2_bins):
    """
    Convert continuous actions [vp1, vp2] to discrete action indices (0-9)
    action_idx = vp1 * n_bins + vp2_bin
    """
    vp1 = actions[:, 0].astype(int)  # Binary 0 or 1
    vp2 = actions[:, 1]

    # Convert VP2 continuous to bin index
    vp2_bins_idx = np.digitize(vp2, vp2_edges) - 1
    vp2_bins_idx = np.clip(vp2_bins_idx, 0, len(vp2_edges) - 2)

    # Combine: action_idx = vp1 * n_bins + vp2_bin
    action_indices = vp1 * vp2_bins + vp2_bins_idx
    return action_indices

# Get model actions as discrete indices (0-9)
print("Computing model actions (discrete) for training data...")
train_model_actions_discrete = select_action_batch_discrete(train_data['states'], q1_network, q2_network, n_bins, device)

print("Computing model actions (discrete) for test data...")
test_model_actions_discrete = select_action_batch_discrete(test_data['states'], q1_network, q2_network, n_bins, device)

# Convert clinician continuous actions to discrete indices (0-9)
train_clinician_actions_discrete = continuous_to_discrete_action(train_data['actions'], vp2_bin_edges, n_bins)
test_clinician_actions_discrete = continuous_to_discrete_action(test_data['actions'], vp2_bin_edges, n_bins)

print(f"\nDiscrete action indices generated:")
print(f"  Train model actions:     {train_model_actions_discrete.shape}")
print(f"  Train clinician actions: {train_clinician_actions_discrete.shape}")
print(f"  Test model actions:      {test_model_actions_discrete.shape}")
print(f"  Test clinician actions:  {test_clinician_actions_discrete.shape}")

# Display sample discrete actions
print(f"\nSample discrete action in_actionsndices (first 10):")
print(f"  Model (train):     {train_model_actions_discrete[:10]}")
print(f"  Clinician (train): {train_clinician_actions_discrete[:10]}")

# Show action distribution (0-9)
print(f"\nAction distribution (0-9):")
for action_idx in range(n_actions):
    print(f"  Action {action_idx}: Model (train)={np.sum(train_model_actions_discrete==action_idx)}, "
          f"Clinician (train)={np.sum(train_clinician_actions_discrete==action_idx)}, "
          f"Model (test)={np.sum(test_model_actions_discrete==action_idx)}, "
          f"Clinician (test)={np.sum(test_clinician_actions_discrete==action_idx)}") 

## train a logistic classifier (lg_vp1_ma) to predict the vp1 model actions (binary) using the states on the training data, then use it on the test data to get the probability of clinician vp1 actions
## train a logistic classifier (lg_vp1_ca) to predict the vp1 clinician actions (binary) using the states on the training data, then use it on the test data to get the probability of clinician vp1 actions
## train a softmax classifier (sm_vp2_ma) to predict the vp2 model actions (block discrete) using the states on the training data, then use it on the test data to get the probability of clinician vp2 actions
## train a softmax classifier (sm_vp2_ca) to predict the vp2 clinician (block discrete) using the states on the training data, then use it on the test data to get the probability of clinician vp2 actions
## save the test data probabilities, and display some of them (each of the four)

print("\n" + "="*70)
print("TRAINING BEHAVIOR POLICY CLASSIFIERS (JOINT ACTION SPACE)")
print("="*70)

# Train softmax classifier for model actions (0-9)
print("\n1. Training softmax classifier for model actions (10 classes)...")
clf_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42)
clf_model.fit(train_data['states'], train_model_actions_discrete)
train_acc_model = accuracy_score(train_model_actions_discrete, clf_model.predict(train_data['states']))
test_acc_model = accuracy_score(test_model_actions_discrete, clf_model.predict(test_data['states']))
print(f"   Train accuracy: {train_acc_model:.4f}")
print(f"   Test accuracy:  {test_acc_model:.4f}")

# Train softmax classifier for clinician actions (0-9)
print("\n2. Training softmax classifier for clinician actions (10 classes)...")
clf_clinician = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42)
clf_clinician.fit(train_data['states'], train_clinician_actions_discrete)
train_acc_clinician = accuracy_score(train_clinician_actions_discrete, clf_clinician.predict(train_data['states']))
test_acc_clinician = accuracy_score(test_clinician_actions_discrete, clf_clinician.predict(test_data['states']))
print(f"   Train accuracy: {train_acc_clinician:.4f}")
print(f"   Test accuracy:  {test_acc_clinician:.4f}")

# Get probabilities on test data
print("\n" + "="*70)
print("COMPUTING BEHAVIOR POLICY PROBABILITIES")
print("="*70)

# Model policy: π_model(a_clinician | s)
test_probs_model = clf_model.predict_proba(test_data['states'])
test_prob_model = test_probs_model[np.arange(len(test_clinician_actions_discrete)), test_clinician_actions_discrete.astype(int)]

# Clinician policy: π_clinician(a_clinician | s)
test_probs_clinician = clf_clinician.predict_proba(test_data['states'])
test_prob_clinician = test_probs_clinician[np.arange(len(test_clinician_actions_discrete)), test_clinician_actions_discrete.astype(int)]

print(f"\nTest probability statistiv:")
print(f"  Model policy π(a_clinician|s)     - Mean: {test_prob_model.mean():.4f}, Std: {test_prob_model.std():.4f}, Min: {test_prob_model.min():.4f}, Max: {test_prob_model.max():.4f}")
print(f"  Clinician policy π(a_clinician|s) - Mean: {test_prob_clinician.mean():.4f}, Std: {test_prob_clinician.std():.4f}, Min: {test_prob_clinician.min():.4f}, Max: {test_prob_clinician.max():.4f}")

print(f"\nSample probabilities (first 10 test transitions):")
print(f"  Index | Model π | Clinician π | Ratio")
print(f"  " + "-"*50)
for i in range(min(10, len(test_prob_model))):
    ratio = test_prob_model[i] / (test_prob_clinician[i] + 1e-10)
    print(f"  {i:5d} | {test_prob_model[i]:.4f} | {test_prob_clinician[i]:.4f} | {ratio:.4f}") 


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
is_weight = test_prob_model / (test_prob_clinician + eps)

print(f"\nIS weight statistics:")
print(f"  Mean: {is_weight.mean():.4f}, Std: {is_weight.std():.4f}, Min: {is_weight.min():.4f}, Max: {is_weight.max():.4f}")

# Get test rewards
test_rewards = test_data['rewards']

# Compute IS-weighted rewards
is_weighted_rewards = is_weight * test_rewards

print(f"\nReward statistics:")
print(f"  Raw rewards (clinician)     - Mean: {test_rewards.mean():.4f}, Std: {test_rewards.std():.4f}, Sum: {test_rewards.sum():.2f}")
print(f"  IS-weighted rewards (model) - Mean: {is_weighted_rewards.mean():.4f}, Std: {is_weighted_rewards.std():.4f}, Sum: {is_weighted_rewards.sum():.2f}")

# Per-transition average
avg_raw_reward_per_transition = test_rewards.mean()
avg_is_reward_per_transition = is_weighted_rewards.mean()

print(f"\nPer-transition averages:")
print(f"  Clinician policy (raw):     {avg_raw_reward_per_transition:.4f}")
print(f"  Model policy (IS-weighted): {avg_is_reward_per_transition:.4f}")
print(f"  Difference:                 {avg_is_reward_per_transition - avg_raw_reward_per_transition:.4f}")

# Per-patient average (efficient computation)
test_patient_ids = test_data['patient_ids']
unique_patients = np.unique(test_patient_ids)
n_patients = len(unique_patients)

avg_raw_reward_per_patient = test_rewards.sum() / n_patients
avg_is_reward_per_patient = is_weighted_rewards.sum() / n_patients

print(f"\nPer-patient averages (efficient computation):")
print(f"  Clinician policy (raw):     {avg_raw_reward_per_patient:.4f}")
print(f"  Model policy (IS-weighted): {avg_is_reward_per_patient:.4f}")
print(f"  Difference:                 {avg_is_reward_per_patient - avg_raw_reward_per_patient:.4f}")

# Compute per-patient statistics (std, min, max) - requires loop
print(f"\nComputing per-patient statistics...")
patient_raw_rewards = []
patient_is_rewards = []

for patient_id in unique_patients:
    patient_mask = test_patient_ids == patient_id
    patient_raw_reward = test_rewards[patient_mask].sum()
    patient_is_reward = is_weighted_rewards[patient_mask].sum()
    patient_raw_rewards.append(patient_raw_reward)
    patient_is_rewards.append(patient_is_reward)

patient_raw_rewards = np.array(patient_raw_rewards)
patient_is_rewards = np.array(patient_is_rewards)

print(f"\nPer-patient reward statistics:")
print(f"  Raw (clinician):    Mean: {patient_raw_rewards.mean():.4f}, Std: {patient_raw_rewards.std():.4f}, Min: {patient_raw_rewards.min():.2f}, Max: {patient_raw_rewards.max():.2f}")
print(f"  IS (model):         Mean: {patient_is_rewards.mean():.4f}, Std: {patient_is_rewards.std():.4f}, Min: {patient_is_rewards.min():.2f}, Max: {patient_is_rewards.max():.2f}")

# Create latex directory
os.makedirs('latex', exist_ok=True)

# Save results to LaTeX table
latex_output = r"""\begin{table}[h]
\centering
\caption{Importance Sampling Off-Policy Evaluation Results (Block Discrete CQL, $\alpha=0.0$)}
\label{tab:is_ope_results}
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
    len(test_rewards), len(test_rewards),
    avg_is_reward_per_transition - avg_raw_reward_per_transition,
    avg_is_reward_per_patient - avg_raw_reward_per_patient
)

latex_file = 'latex/is_ope_results.tex'
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
sum_weighted_rewards = (is_weight * test_rewards).sum()

# Sum of weights
sum_weights = is_weight.sum()

# WIS estimate
wis_per_transition = sum_weighted_rewards / sum_weights if sum_weights > 0 else 0.0

# Standard IS (for comparison)
is_per_transition = (is_weight * test_rewards).mean()

# Clinician average (raw)
clinician_per_transition = test_rewards.mean()

print(f"\nPer-transition evaluation:")
print(f"  Clinician policy (raw):        {clinician_per_transition:.4f}")
print(f"  Model policy (standard IS):    {is_per_transition:.4f}")
print(f"  Model policy (weighted IS):    {wis_per_transition:.4f}")

print(f"\nImprovement over clinician:")
print(f"  Standard IS:  {is_per_transition - clinician_per_transition:.4f}")
print(f"  Weighted IS:  {wis_per_transition - clinician_per_transition:.4f}")

print(f"\nWeight normalization:")
print(f"  Sum of weights: {sum_weights:.4f}")
print(f"  Number of transitions: {len(test_rewards)}")
print(f"  Average weight: {sum_weights / len(test_rewards):.4f}")

# (4) Weighted Importance Sampling (WIS) - Per-Trajectory Level
print("\n" + "="*70)
print("WEIGHTED IMPORTANCE SAMPLING (WIS) - PER-TRAJECTORY")
print("="*70)

# For each patient trajectory: WIS_τ = Σᵢ(wᵢ·rᵢ) / Σᵢ(wᵢ) for transitions in that trajectory
# Then average across all patients

wis_per_trajectory_list = []

for patient_id in unique_patients:
    patient_mask = test_patient_ids == patient_id

    # Get weights and rewards for this trajectory
    patient_weights = is_weight[patient_mask]
    patient_rewards = test_rewards[patient_mask]

    # Compute WIS for this trajectory
    sum_weighted_rewards_patient = (patient_weights * patient_rewards).sum()
    sum_weights_patient = patient_weights.sum()

    wis_trajectory = sum_weighted_rewards_patient / sum_weights_patient if sum_weights_patient > 0 else 0.0
    wis_per_trajectory_list.append(wis_trajectory)

wis_per_trajectory_list = np.array(wis_per_trajectory_list)

# Compute mean WIS across all trajectories
wis_per_trajectory_mean = wis_per_trajectory_list.mean()

# Also compute raw clinician per-trajectory for comparison
clinician_per_trajectory_mean = patient_raw_rewards.mean()

# Compute 95% confidence interval using bootstrapping for the DIFFERENCE
print(f"\nComputing 95% CI for difference using bootstrapping (1000 iterations)...")

# First, compute the difference for each trajectory
# For clinician: just the raw reward sum per patient
# For model: WIS per trajectory (already computed)
# Difference: s = WIS_model - raw_clinician for each patient

trajectory_differences = wis_per_trajectory_list - patient_raw_rewards

# Bootstrap from the bag of differences
n_bootstrap = 1000
bootstrap_diff_means = []

np.random.seed(42)  # For reproducibility
for _ in range(n_bootstrap):
    # Resample differences with replacement
    bootstrap_indices = np.random.choice(len(trajectory_differences), size=len(trajectory_differences), replace=True)
    bootstrap_sample = trajectory_differences[bootstrap_indices]
    bootstrap_diff_means.append(bootstrap_sample.mean())

bootstrap_diff_means = np.array(bootstrap_diff_means)

# Compute 95% CI using percentile method
ci_diff_lower = np.percentile(bootstrap_diff_means, 2.5)
ci_diff_upper = np.percentile(bootstrap_diff_means, 97.5)

# Mean difference
improvement_mean = trajectory_differences.mean()

print(f"\nPer-trajectory WIS evaluation with 95% CI:")
print(f"  Clinician policy (raw):        {clinician_per_trajectory_mean:.4f}")
print(f"  Model policy (WIS):            {wis_per_trajectory_mean:.4f}")
print(f"  Difference (Model - Clinician): {improvement_mean:.4f} (95% CI: [{ci_diff_lower:.4f}, {ci_diff_upper:.4f}])")

print(f"\nPer-trajectory WIS statistics:")
print(f"  Mean: {wis_per_trajectory_list.mean():.4f}")
print(f"  Std:  {wis_per_trajectory_list.std():.4f}")
print(f"  Min:  {wis_per_trajectory_list.min():.4f}")
print(f"  Max:  {wis_per_trajectory_list.max():.4f}")

# Create comprehensive LaTeX table with all results
print("\n" + "="*70)
print("GENERATING COMPREHENSIVE LATEX TABLE")
print("="*70)

# Create LaTeX table with dynamic caption and label based on alpha and bins
# Use f-string formatting for caption/label, but keep old-style % formatting for data values
latex_comprehensive = r"""\begin{table}[h]
\centering
\caption{Importance Sampling Off-Policy Evaluation Results (Block Discrete CQL, $\alpha=""" + f"{args.alpha:.4f}" + r"""$, """ + f"{args.bins} bins" + r""")}
\label{tab:is_ope_alpha""" + f"{args.alpha:.4f}".replace('.', '_') + f"_bins{args.bins}" + r"""}
\begin{tabular}{lcccc}
\hline
\textbf{Method} & \textbf{Clinician} & \textbf{Model (IS/WIS)} & \textbf{Difference} & \textbf{95\%% CI} \\
\hline
\multicolumn{5}{l}{\textit{Per-Transition Estimates}} \\
\hline
Standard IS & %.4f & %.4f & %.4f & -- \\
Weighted IS (WIS) & %.4f & %.4f & %.4f & -- \\
\hline
\multicolumn{5}{l}{\textit{Per-Trajectory (Per-Patient) Estimates}} \\
\hline
Standard IS & %.4f & %.4f & %.4f & -- \\
Weighted IS (WIS) & %.4f & %.4f & %.4f & [%.4f, %.4f] \\
\hline
\multicolumn{5}{l}{\textit{Data Summary}} \\
\hline
\multicolumn{2}{l}{Number of Patients} & \multicolumn{3}{c}{%d} \\
\multicolumn{2}{l}{Number of Transitions} & \multicolumn{3}{c}{%d} \\
\multicolumn{2}{l}{Average Trajectory Length} & \multicolumn{3}{c}{%.1f} \\
\hline
\end{tabular}
\end{table}
""" % (
    # Per-transition IS
    avg_raw_reward_per_transition, avg_is_reward_per_transition,
    avg_is_reward_per_transition - avg_raw_reward_per_transition,
    # Per-transition WIS
    clinician_per_transition, wis_per_transition,
    wis_per_transition - clinician_per_transition,
    # Per-trajectory IS
    clinician_per_trajectory_mean, patient_is_rewards.mean(),
    patient_is_rewards.mean() - clinician_per_trajectory_mean,
    # Per-trajectory WIS
    clinician_per_trajectory_mean, wis_per_trajectory_mean,
    improvement_mean, ci_diff_lower, ci_diff_upper,
    # Data summary
    n_patients, len(test_rewards), len(test_rewards) / n_patients
)

# Dynamic output filename: latex/is_ope_alpha{alpha}_bins{bins}.tex
# Example: latex/is_ope_alpha0.0000_bins5.tex
latex_file_comprehensive = f'{args.output_dir}/is_ope_alpha{args.alpha:.4f}_bins{args.bins}.tex'
with open(latex_file_comprehensive, 'w') as f:
    f.write(latex_comprehensive)

print(f"Comprehensive results saved to: {latex_file_comprehensive}") 

## questions and answers
#1. Which trained model should I load? Should I use a specific alpha value (0.0, 0.001, 0.01) or a specific checkpoint? Or should I train a new one?
# - first let's use /scratch/zouwil/code/ucsf_rl/experiment/block_discrete_cql_alpha0.0000_bins5_best.pt with 5 bins and alpha = 0 
#  2. Block discrete binning: Should I use the same 5 bins [0, 0.05, 0.1, 0.2, 0.5] for VP2 as defined in data_config.py?
# - with 5 bins [0, 0.05, 0.1, 0.2, 0.5] 
#  3. Epsilon for numerical stability: What value should I use for eps in the denominator to avoid division by zero?
# - let's try 1e-10 first, later we may have to use larger values or apply truncation 
#  4. Output location: Where should I save the LaTeX table output? 
# creat a folder vaso_rl/latex/
#  5. Test data only: Should I perform IS-OPE only on test data, or also report validation results?
# - only on test data for now I think there are no hyperparameters for the softmax or logistic classifier model for probability estimation 
