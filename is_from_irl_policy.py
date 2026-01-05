### import relevant libraries
import torch, pdb
import torch.nn as nn
import torch.nn.functional as F
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
from unet_reward_generator import UNetRewardGenerator, Config

# Parse command line arguments
parser = argparse.ArgumentParser(description='Importance Sampling evaluation for IRL Policy Models')
parser.add_argument('--model_path', type=str, required=True,
                   help='Path to trained IRL policy model checkpoint')
parser.add_argument('--model_type', type=str, default='unet', choices=['unet', 'gcl', 'iqlearn'],
                   help='Type of IRL policy model (default: unet)')
parser.add_argument('--vp2_bins', type=int, default=5,
                   help='Number of bins for VP2 discretization (default: 5)')
parser.add_argument('--eval_set', type=str, default='test', choices=['val', 'test'],
                   help='Which data split to evaluate on (default: test)')
parser.add_argument('--temperature', type=float, default=1.0,
                   help='Temperature for softmax policy (default: 1.0)')
parser.add_argument('--sequence_length', type=int, default=40,
                   help='Sequence length for U-Net processing (default: 40)')
args = parser.parse_args()

n_bins = args.vp2_bins
model_path = args.model_path
model_type = args.model_type
eval_set = args.eval_set
temperature = args.temperature
sequence_length = args.sequence_length

# Define constants
n_actions = 2 * n_bins  # VP1 (2 options) x VP2 (n_bins) = total actions

### Load the IRL policy model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"Model type: {model_type}")

# Load checkpoint
checkpoint = torch.load(model_path, map_location=device)

if model_type == 'unet':
    # ============================================================
    # U-Net Reward Generator Model
    # ============================================================
    # Get state_size, action_size, conv_h_dim from checkpoint (saved at top level)
    state_size = checkpoint.get('state_size')
    action_size = checkpoint.get('action_size', n_actions)
    conv_h_dim = checkpoint.get('conv_h_dim', 64)

    # Fallback: infer from model weights if not found
    if state_size is None:
        input_size = checkpoint['model_state_dict']['enc1.0.weight'].shape[1]
        state_size = input_size - action_size

    print(f"Loaded config - state_size: {state_size}, action_size: {action_size}, conv_h_dim: {conv_h_dim}")

    # Initialize U-Net model with matching architecture
    irl_model = UNetRewardGenerator(state_size=state_size, action_size=action_size, conv_h_dim=conv_h_dim).to(device)
    irl_model.load_state_dict(checkpoint['model_state_dict'])
    irl_model.eval()

    print(f"Loaded U-Net model from: {model_path}")

elif model_type == 'gcl':
    # ============================================================
    # GCL (Guided Cost Learning) Policy Model - PLACEHOLDER
    # ============================================================
    # TODO: Implement GCL model loading
    # from gcl_policy import GCLPolicy
    # irl_model = GCLPolicy(state_size=state_size, action_size=action_size).to(device)
    # irl_model.load_state_dict(checkpoint['model_state_dict'])
    # irl_model.eval()
    raise NotImplementedError("GCL policy model loading not yet implemented")

elif model_type == 'iqlearn':
    # ============================================================
    # IQ-Learn Policy Model - PLACEHOLDER
    # ============================================================
    # TODO: Implement IQ-Learn model loading
    # from iqlearn_policy import IQLearnPolicy
    # irl_model = IQLearnPolicy(state_size=state_size, action_size=action_size).to(device)
    # irl_model.load_state_dict(checkpoint['model_state_dict'])
    # irl_model.eval()
    raise NotImplementedError("IQ-Learn policy model loading not yet implemented")

print(f"State dimension: {state_size}")
print(f"Number of actions: {n_actions} (VP1: 2 binary × VP2: {n_bins} bins)")
print(f"Temperature for softmax policy: {temperature}") 

###
# prepare and load the data with random seed 42

print("\n" + "="*70)
print("LOADING DATA")
print("="*70)
print(f"Evaluation set: {eval_set}")

# Initialize data pipeline with manual rewards (IRL policy models generate their own rewards)
pipeline = IntegratedDataPipelineV3(model_type='dual', reward_source='manual', random_seed=42)
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

## use the trained model to predict the optimal actions predicted by the model

print("\n" + "="*70)
print("GENERATING MODEL ACTIONS")
print("="*70)

# Define VP2 bin edges (exactly as in training script)
vp2_bin_edges = np.linspace(0, 0.5, n_bins + 1)
print(f"VP2 bin edges: {vp2_bin_edges}")

# Helper function to convert continuous actions to discrete indices
def continuous_to_discrete_action(actions, vp2_edges, vp2_bins):
    """
    Convert continuous actions [vp1, vp2] to discrete action indices (0 to n_actions-1)
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


# ============================================================
# U-Net: Compute Q-values from reward predictions
# ============================================================
def compute_q_values_unet(model, states_seq, actions_seq, D, gamma, device):
    """
    Compute Q-values for all actions at each timestep using U-Net reward generator.

    NOTE: For timesteps ct >= seq_len - D, we use reduced horizon (fewer future steps).
    This could cause issues due to insufficient look-ahead for those timesteps.

    Args:
        model: UNetRewardGenerator model
        states_seq: [batch, seq_len, state_size] tensor
        actions_seq: [batch, seq_len] discrete action indices
        D: horizon for Q computation
        gamma: discount factor
        device: torch device

    Returns:
        q_values: [batch, seq_len, n_actions] Q-values for each action at each timestep
    """
    batch_size, seq_len, state_size = states_seq.shape
    action_size = model.action_size

    # Convert expert actions to one-hot
    expert_action_one_hot = F.one_hot(actions_seq, num_classes=action_size).float()

    # Expand for all action choices at current timestep
    states_expanded = states_seq.unsqueeze(2).expand(-1, -1, action_size, -1)
    expert_action_expanded = expert_action_one_hot.unsqueeze(2).expand(-1, -1, action_size, -1)

    # Eye matrix for replacing actions at timestep ct
    action_eye = torch.eye(action_size, device=device).unsqueeze(0).expand(batch_size, -1, -1)

    # Discount factors for horizon D
    discount = torch.tensor([gamma ** t for t in range(D)], device=device)

    # Initialize Q-values
    q_values_all = torch.zeros(batch_size, seq_len, action_size, device=device)

    # Compute Q-values for all timesteps
    # NOTE: For ct >= seq_len - D, we have insufficient look-ahead (reduced horizon)
    for ct in range(seq_len):
        # Replace expert action at ct with all possible actions
        cur_action_expanded = expert_action_expanded.clone()
        cur_action_expanded[:, ct, :, :] = action_eye
        state_action = torch.cat([states_expanded, cur_action_expanded], dim=-1)

        # Reshape and forward through U-Net
        state_action_flat = state_action.permute(0, 2, 1, 3).reshape(
            batch_size * action_size, seq_len, state_size + action_size
        )
        rewards_pred = model(state_action_flat)
        rewards_pred = rewards_pred.reshape(batch_size, action_size, seq_len, 1)
        rewards_pred = rewards_pred.squeeze(-1).permute(0, 2, 1)

        # Q = sum of discounted future rewards
        # Use reduced horizon if near end of sequence
        remaining = min(D, seq_len - ct)
        future_rewards = rewards_pred[:, ct:ct+remaining, :]
        discount_slice = discount[:remaining]
        Q_values = (future_rewards * discount_slice.unsqueeze(0).unsqueeze(-1)).sum(dim=1)
        q_values_all[:, ct, :] = Q_values

    return q_values_all


# ============================================================
# GCL: Compute Q-values - PLACEHOLDER
# ============================================================
def compute_q_values_gcl(model, states, device):
    """TODO: Implement GCL Q-value computation"""
    raise NotImplementedError("GCL Q-value computation not yet implemented")


# ============================================================
# IQ-Learn: Compute Q-values - PLACEHOLDER
# ============================================================
def compute_q_values_iqlearn(model, states, device):
    """TODO: Implement IQ-Learn Q-value computation"""
    raise NotImplementedError("IQ-Learn Q-value computation not yet implemented")


# ============================================================
# U-Net: Get model actions for all transitions
# ============================================================
MIN_SEQ_LEN = 7  # Minimum length for U-Net (due to 3 conv layers shrinking by 6)

def get_model_actions_unet(model, data, vp2_bin_edges, n_bins, D, gamma, device):
    """
    Process each patient trajectory at its actual length (no padding needed).
    U-Net handles variable-length inputs via Conv1d.

    Returns:
        model_actions: best action for each transition
        clinician_actions_discrete: clinician actions as discrete indices
    """
    states = data['states']
    actions = data['actions']
    patient_ids = data['patient_ids']

    n_transitions = len(states)

    # Convert clinician actions to discrete
    clinician_actions_discrete = continuous_to_discrete_action(actions, vp2_bin_edges, n_bins)

    # Initialize output
    model_actions = np.zeros(n_transitions, dtype=int)

    # Process by patient
    unique_patients = np.unique(patient_ids)
    print(f"Processing {len(unique_patients)} patients...")

    with torch.no_grad():
        for i, pid in enumerate(unique_patients):
            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{len(unique_patients)} patients")

            mask = patient_ids == pid
            patient_states = states[mask]
            patient_actions = clinician_actions_discrete[mask]
            patient_indices = np.where(mask)[0]
            seq_len = len(patient_states)

            # Skip very short trajectories (U-Net needs min length)
            if seq_len < MIN_SEQ_LEN:
                # Use uniform random or clinician action for short sequences
                model_actions[patient_indices] = patient_actions
                continue

            # Convert to tensors (batch_size=1, variable seq_len)
            states_tensor = torch.FloatTensor(patient_states).unsqueeze(0).to(device)
            actions_tensor = torch.LongTensor(patient_actions).unsqueeze(0).to(device)

            # Compute Q-values for this patient's trajectory
            q_values = compute_q_values_unet(model, states_tensor, actions_tensor, D, gamma, device)
            q_values = q_values.squeeze(0)  # [seq_len, action_size]

            # Get best actions (argmax)
            best_actions = q_values.argmax(dim=-1).cpu().numpy()

            # Assign to output
            model_actions[patient_indices] = best_actions

    return model_actions, clinician_actions_discrete


# ============================================================
# GCL: Get model actions - PLACEHOLDER
# ============================================================
def get_model_actions_gcl(model, data, vp2_bin_edges, n_bins, device):
    """TODO: Implement GCL action selection"""
    raise NotImplementedError("GCL action selection not yet implemented")


# ============================================================
# IQ-Learn: Get model actions - PLACEHOLDER
# ============================================================
def get_model_actions_iqlearn(model, data, vp2_bin_edges, n_bins, device):
    """TODO: Implement IQ-Learn action selection"""
    raise NotImplementedError("IQ-Learn action selection not yet implemented")


# Get Q-value computation parameters
config = Config()
D = config.D
gamma = config.gamma

print(f"\nQ-value computation parameters:")
print(f"  Horizon D: {D}")
print(f"  Discount gamma: {gamma}")

# Compute model actions based on model type
if model_type == 'unet':
    print("\nComputing model actions for training data...")
    train_model_actions_discrete, train_clinician_actions_discrete = get_model_actions_unet(
        irl_model, train_data, vp2_bin_edges, n_bins, D, gamma, device
    )
    print(f"\nComputing model actions for {eval_set_name.lower()} data...")
    eval_model_actions_discrete, eval_clinician_actions_discrete = get_model_actions_unet(
        irl_model, eval_data, vp2_bin_edges, n_bins, D, gamma, device
    )
elif model_type == 'gcl':
    train_model_actions_discrete, train_clinician_actions_discrete = get_model_actions_gcl(
        irl_model, train_data, vp2_bin_edges, n_bins, device
    )
    eval_model_actions_discrete, eval_clinician_actions_discrete = get_model_actions_gcl(
        irl_model, eval_data, vp2_bin_edges, n_bins, device
    )
elif model_type == 'iqlearn':
    train_model_actions_discrete, train_clinician_actions_discrete = get_model_actions_iqlearn(
        irl_model, train_data, vp2_bin_edges, n_bins, device
    )
    eval_model_actions_discrete, eval_clinician_actions_discrete = get_model_actions_iqlearn(
        irl_model, eval_data, vp2_bin_edges, n_bins, device
    )

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

isw_ci_diff_lower = np.percentile(is_weight, 2.5)
isw_ci_diff_upper = np.percentile(is_weight, 97.5)

#is_weight = np.clip(is_weight, a_min = CLIP_MIN, a_max = CLIP_MAX)
is_weight = np.clip(is_weight, a_min = isw_ci_diff_lower, a_max = isw_ci_diff_upper)

print(f"\nIS weight statistics:")
print(f"  Mean: {is_weight.mean():.4f}, Std: {is_weight.std():.4f}, Min: {is_weight.min():.4f}, Max: {is_weight.max():.4f}")

# Get eval rewards
eval_rewards = eval_data['rewards']

# Compute IS-weighted rewards
is_weighted_rewards = is_weight * eval_rewards

print(f"\nReward statistics:")
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
model_type_label = {'unet': 'U-Net Reward Generator', 'gcl': 'GCL', 'iqlearn': 'IQ-Learn'}
latex_output = r"""\begin{table}[h]
\centering
\caption{Importance Sampling Off-Policy Evaluation Results (%s)}
\label{tab:is_ope_results_%s}
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
    model_type_label.get(model_type, model_type), model_type,
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

latex_file = f'latex/is_ope_results_{model_type}.tex'
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

weights_per_trajectory_list = [] 
total_rewards_per_trajectory_list = []
weighted_rewards_per_trajectory_list = []

for patient_id in unique_patients:
    patient_mask = eval_patient_ids == patient_id
    """
    mean_model_prob = eval_prob_model[patient_mask].mean()
    mean_clinician_prob = eval_prob_clinician[patient_mask].mean()
    patient_rewards = eval_rewards[patient_mask]
    traj_is_weight = mean_model_prob /(1e-8 + mean_clinician_prob)

    if not traj_is_weight == 0:
        #weights_per_trajectory_list.append(patient_weights.prod())
        weights_per_trajectory_list.append(traj_is_weight)
        total_rewards_per_trajectory_list.append(patient_rewards.sum())
    else:
        print("zero encountered in trajectory level multiplied weights")
    """

    # Get weights and rewards for this trajectory
    patient_weights = is_weight[patient_mask]
    patient_rewards = eval_rewards[patient_mask]
    est_total_reward_per_traj = (patient_weights *  patient_rewards).sum() / patient_weights.sum() * len(patient_weights)

    weights_per_trajectory_list.append(patient_weights.mean())
    total_rewards_per_trajectory_list.append(patient_rewards.sum())
    weighted_rewards_per_trajectory_list.append(est_total_reward_per_traj)
    

#wisw_ci_diff_lower = np.percentile(weights_per_trajectory_list, 5)
#wisw_ci_diff_upper = np.percentile(weights_per_trajectory_list, 95)

#weights_per_trajectory_list = np.clip(weights_per_trajectory_list, a_min = wisw_ci_diff_lower, a_max = wisw_ci_diff_upper)

#weights_per_trajectory_list = np.clip(weights_per_trajectory_list, a_min = 0.0, a_max = 10)

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
