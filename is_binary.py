### import relevant libraries
import torch, pdb
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss
import os
import warnings
warnings.filterwarnings('ignore')

# Import project modules
from integrated_data_pipeline_v2 import IntegratedDataPipelineV2

### define the binary CQL model
# - load the trained binary CQL model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Import the BinaryCQL class from the training script
from train_binary_cql import BinaryCQL

# Model parameters
n_actions = 2  # Binary actions: 0 or 1 for VP1
state_dim = 17  # 18 state features for binary CQL (includes VP2 as a state feature)

# Initialize BinaryCQL agent
binary_agent = BinaryCQL(
    state_dim=state_dim,
    alpha=0.0,
    gamma=0.99,
    tau=0.08,
    lr=1e-3,
    device=device
)

# Load the trained model checkpoint
model_path = 'experiment/binary_vp1_only/binary_cql_unified_alpha00_final.pt'

checkpoint = torch.load(model_path, map_location=device)
binary_agent.q1.load_state_dict(checkpoint['q1_state_dict'])
binary_agent.q2.load_state_dict(checkpoint['q2_state_dict'])
binary_agent.q1.eval()
binary_agent.q2.eval()

print(f"Loaded model from: {model_path}")
print(f"State dimension: {state_dim}")
print(f"Number of actions: {n_actions} (VP1 binary: 0 or 1)")
print("Using min(Q1, Q2) for action selection (double Q-learning)") 

###
# prepare and load the data with random seed 42 with all the data in a sequence 

print("\n" + "="*70)
print("LOADING DATA")
print("="*70)

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

print("\n" + "="*70)
print("GENERATING MODEL ACTIONS")
print("="*70)

# For binary QL, VP2 is Not part of the state, nor the action
# Actions are simply binary: 0 or 1
print(f"Binary actions: 0 or 1 (VP1 only)")

# Helper function to select binary actions and return values (0 or 1)
def select_action_batch_binary(states, agent, device):
    """
    Select best binary actions for a batch of states using min(Q1, Q2)
    Returns: numpy array of binary actions [0 or 1]
    """
    with torch.no_grad():
        if states.ndim == 1:
            states = states.reshape(1, -1)

        batch_size = states.shape[0]
        state_tensor = torch.FloatTensor(states).to(device)

        # Create action tensors for both possible actions (0 and 1)
        actions_0 = torch.zeros(batch_size, 1).to(device)
        actions_1 = torch.ones(batch_size, 1).to(device)

        # Evaluate Q-values for action=0
        q1_values_0 = agent.q1(state_tensor, actions_0).squeeze()
        q2_values_0 = agent.q2(state_tensor, actions_0).squeeze()
        q_values_0 = torch.min(q1_values_0, q2_values_0)

        # Evaluate Q-values for action=1
        q1_values_1 = agent.q1(state_tensor, actions_1).squeeze()
        q2_values_1 = agent.q2(state_tensor, actions_1).squeeze()
        q_values_1 = torch.min(q1_values_1, q2_values_1)

        # Select best action for each state (argmax: 0 or 1)
        best_actions = (q_values_1 > q_values_0).float().cpu().numpy()

        return best_actions

# Get model binary actions (0 or 1)
print("Computing model binary actions for training data...")
train_model_actions_binary = select_action_batch_binary(train_data['states'], binary_agent, device)

print("Computing model binary actions for test data...")
test_model_actions_binary = select_action_batch_binary(test_data['states'], binary_agent, device)

# Get clinician binary actions (already binary in the data, just extract VP1)
# For binary CQL, actions are already binary (0 or 1), no discretization needed

train_clinician_actions_binary = train_data['actions'][:,0]  # VP1 is already binary
test_clinician_actions_binary = test_data['actions'][:,0]    # VP1 is already binary

print(f"\nBinary actions generated:")
print(f"  Train model actions:     {train_model_actions_binary.shape}")
print(f"  Train clinician actions: {train_clinician_actions_binary.shape}")
print(f"  Test model actions:      {test_model_actions_binary.shape}")
print(f"  Test clinician actions:  {test_clinician_actions_binary.shape}")

# Display sample binary actions
print(f"\nSample binary actions (first 10):")
print(f"  Model (train):     {train_model_actions_binary[:10]}")
print(f"  Clinician (train): {train_clinician_actions_binary[:10]}")

# Show action distribution (0 or 1)
print(f"\nAction distribution (binary: 0 or 1):")
for action_idx in range(n_actions):
    print(f"  Action {action_idx}: Model (train)={np.sum(train_model_actions_binary==action_idx)}, "
          f"Clinician (train)={np.sum(train_clinician_actions_binary==action_idx)}, "
          f"Model (test)={np.sum(test_model_actions_binary==action_idx)}, "
          f"Clinician (test)={np.sum(test_clinician_actions_binary==action_idx)}") 

print("\n" + "="*70)
print("TRAINING BEHAVIOR POLICY CLASSIFIERS (BINARY ACTION SPACE)")
print("="*70)

# Train logistic classifier for model binary actions (0 or 1)
print("\n1. Training logistic classifier for model binary actions (2 classes: 0 or 1)...")
clf_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42)
clf_model.fit(train_data['states'], train_model_actions_binary.astype(int))
train_acc_model = accuracy_score(train_model_actions_binary, clf_model.predict(train_data['states']))
test_acc_model = accuracy_score(test_model_actions_binary, clf_model.predict(test_data['states']))
print(f"   Train accuracy: {train_acc_model:.4f}")
print(f"   Test accuracy:  {test_acc_model:.4f}")

# Train logistic classifier for clinician binary actions (0 or 1)
print("\n2. Training logistic classifier for clinician binary actions (2 classes: 0 or 1)...")
clf_clinician = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42)
clf_clinician.fit(train_data['states'], train_clinician_actions_binary.astype(int))
train_acc_clinician = accuracy_score(train_clinician_actions_binary, clf_clinician.predict(train_data['states']))
test_acc_clinician = accuracy_score(test_clinician_actions_binary, clf_clinician.predict(test_data['states']))
print(f"   Train accuracy: {train_acc_clinician:.4f}")
print(f"   Test accuracy:  {test_acc_clinician:.4f}")

# Get probabilities on test data
print("\n" + "="*70)
print("COMPUTING BEHAVIOR POLICY PROBABILITIES")
print("="*70)

# Model policy: π_model(a_clinician | s)
test_probs_model = clf_model.predict_proba(test_data['states'])
test_prob_model = test_probs_model[np.arange(len(test_clinician_actions_binary)), test_clinician_actions_binary.astype(int)]

# Clinician policy: π_clinician(a_clinician | s)
test_probs_clinician = clf_clinician.predict_proba(test_data['states'])
test_prob_clinician = test_probs_clinician[np.arange(len(test_clinician_actions_binary)), test_clinician_actions_binary.astype(int)]

print(f"\nTest probability statistics:")
print(f"  Model policy π(a_clinician|s)     - Mean: {test_prob_model.mean():.4f}, Std: {test_prob_model.std():.4f}, Min: {test_prob_model.min():.4f}, Max: {test_prob_model.max():.4f}")
print(f"  Clinician policy π(a_clinician|s) - Mean: {test_prob_clinician.mean():.4f}, Std: {test_prob_clinician.std():.4f}, Min: {test_prob_clinician.min():.4f}, Max: {test_prob_clinician.max():.4f}")

print(f"\nSample probabilities (first 10 test transitions):")
print(f"  Index | Model π | Clinician π | Ratio")
print(f"  " + "-"*50)
for i in range(min(10, len(test_prob_model))):
    ratio = test_prob_model[i] / (test_prob_clinician[i] + 1e-10)
    print(f"  {i:5d} | {test_prob_model[i]:.4f} | {test_prob_clinician[i]:.4f} | {ratio:.4f}") 


# Finally for each of the transitions (indices) in the test data, compute is_weight * R where R is the reward on that transition index or state, this is the expected reward using the model actions on that state
# then compute the per transition average of the expected reward (using the model recommended actions), as well as the per patient average of the expected reward (using the model recommended actions)
# compare the two values with the average reward (the raw reward) per transition, and the average reward per patient

print("\n" + "="*70)
print("COMPUTING IMPORTANCE SAMPLING WEIGHTS AND REWARDS")
print("="*70)

# Set epsilon for numerical stability
eps = 1e-10

# Compute importance sampling weights (single ratio, no chain rule!)
# IS_weight = π_model(a_clinician|s) / π_clinician(a_clinician|s)
is_weight = test_prob_model / (test_prob_clinician + eps)

# simple clipping: keep 95% of the data and clip the out-lines 
isw_ci_diff_lower = np.percentile(is_weight, 2.5)
isw_ci_diff_upper = np.percentile(is_weight, 97.5)

is_weight = np.clip(is_weight, a_min = isw_ci_diff_lower, a_max = isw_ci_diff_upper)

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
patient_raw_rewards_ave = []

patient_is_rewards = []

for patient_id in unique_patients:
    patient_mask = test_patient_ids == patient_id
    patient_raw_reward = test_rewards[patient_mask].sum()
    patient_is_reward = is_weighted_rewards[patient_mask].sum()
    patient_raw_rewards.append(patient_raw_reward)
    patient_raw_rewards_ave.append(patient_raw_reward/len(test_rewards[patient_mask]))
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

weights_per_trajectory_list = [] 
total_rewards_per_trajectory_list = []
weighted_rewards_per_trajectory_list = [] 

for patient_id in unique_patients:
    patient_mask = test_patient_ids == patient_id
    """
    ## keep track of the 'standard' way of computing trajectory-level WIS estimate of the expected rewards 
    mean_model_prob = test_prob_model[patient_mask].mean()
    mean_clinician_prob = test_prob_clinician[patient_mask].mean()
    patient_rewards = test_rewards[patient_mask]
    traj_is_weight = mean_model_prob /(1e-8 + mean_clinician_prob)

    if not traj_is_weight == 0:
        #weights_per_trajectory_list.append(patient_weights.prod())
        weights_per_trajectory_list.append(traj_is_weight)
        total_rewards_per_trajectory_list.append(patient_rewards.sum())
    else: 
        print("zero encountered in trajectory level multiplied weights")
    """
    # the two-step WIS method, one step to estimate trajectory-level WIS 
    patient_weights = is_weight[patient_mask]
    patient_rewards = test_rewards[patient_mask]
    est_total_reward_per_traj = (patient_weights *  patient_rewards).sum() / patient_weights.sum() * len(patient_weights)
    
    weights_per_trajectory_list.append(patient_weights.mean())
    weighted_rewards_per_trajectory_list.append(est_total_reward_per_traj)
    total_rewards_per_trajectory_list.append(patient_rewards.sum())

# keep as comments: we can remove trajectory-level weight clipping using the two-step WIS method 
#wisw_ci_diff_lower = np.percentile(weights_per_trajectory_list, 2)
#wisw_ci_diff_upper = np.percentile(weights_per_trajectory_list, 98)
#weights_per_trajectory_list = np.clip(weights_per_trajectory_list, a_min = 0.0, a_max = 5)

# Compute mean WIS across all trajectories
weights_per_trajectory_list = np.array(weights_per_trajectory_list)
total_rewards_per_trajectory_list = np.array(total_rewards_per_trajectory_list)
weighted_rewards_per_trajectory_list = np.array(weighted_rewards_per_trajectory_list)

# step two of the two-step WIS, re-normalize trajectories to obtain the overall WIS 
wis_trajectory_level = (weights_per_trajectory_list * weighted_rewards_per_trajectory_list).sum() / weights_per_trajectory_list.sum() 

# raw clinician action average trajectory level reward for comparison
clinician_per_trajectory_mean = total_rewards_per_trajectory_list.mean()

# Compute 95% confidence interval using bootstrapping for the DIFFERENCE
print(f"\nComputing 95% CI for difference using bootstrapping (1000 iterations)...")

# First, compute the difference for each trajectory
# For clinician: use the reference average trajectory level reward 
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
