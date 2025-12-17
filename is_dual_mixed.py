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

# Import the DualMixedCQL class from the training script
from run_dualmixed_cql_allalphas import DualMixedCQL

# Model parameters
state_dim = 17  # 17 state features for dual model (VP2 not included in state)

# Initialize DualMixedCQL agent (VP1 binary + VP2 continuous)
dual_agent = DualMixedCQL(
    state_dim=state_dim,
    alpha=0.0,
    gamma=0.95,
    tau=0.8,
    lr=1e-3,
    device=device
)

# Load the trained model checkpoint
model_path = 'experiment/epoch500_new_trained_oviss_reward_models/dual_rev_cql_alpha0.0000_best.pt'

checkpoint = torch.load(model_path, map_location=device)
dual_agent.q1.load_state_dict(checkpoint['q1_state_dict'])
dual_agent.q2.load_state_dict(checkpoint['q2_state_dict'])
dual_agent.q1.eval()
dual_agent.q2.eval()

print(f"Loaded model from: {model_path}")
print(f"State dimension: {state_dim}")
print(f"Action space: VP1 binary (0 or 1) + VP2 continuous [0.0, 0.5]")
print("Using min(Q1, Q2) for action selection (double Q-learning)") 

###
# prepare and load the data with random seed 42 with all the data in a sequence, for example:
# pipeline = IntegratedDataPipelineV2(model_type='dual', random_seed=42)
# train_data, val_data, test_data = pipeline.prepare_data()

print("\n" + "="*70)
print("LOADING DATA")
print("="*70)

# Initialize data pipeline with random seed 42
#[C-change]: changed data pipeline input: to 'dual'
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

# Helper function to select dual mixed actions (VP1 binary + VP2 continuous)
def select_action_batch_dual_mixed(states, agent, device, n_vp2_samples=50):
    """
    Select best dual mixed actions for a batch of states.
    - VP1: Binary (0 or 1) selected using Q-values
    - VP2: Continuous [0.0, 0.5] sampled uniformly and selected using Q-values

    Args:
        states: [batch_size, state_dim] numpy array
        agent: DualMixedCQL agent
        device: torch device
        n_vp2_samples: Number of VP2 samples to evaluate (default: 50)

    Returns:
        vp1_actions: [batch_size] numpy array of binary actions (0 or 1)
        vp2_actions: [batch_size] numpy array of continuous actions [0.0, 0.5]
    """
    with torch.no_grad():
        if states.ndim == 1:
            states = states.reshape(1, -1)

        batch_size = states.shape[0]
        state_tensor = torch.FloatTensor(states).to(device)

        # Sample VP2 values uniformly from [0.0, 0.5]
        vp2_samples = torch.linspace(0.0, 0.5, n_vp2_samples).to(device)

        best_vp1 = []
        best_vp2 = []

        # For each state, evaluate all combinations of VP1 and VP2
        for i in range(batch_size):
            state = state_tensor[i:i+1]  # [1, state_dim]

            max_q_value = -float('inf')
            best_vp1_i = 0
            best_vp2_i = 0.0

            # Try both VP1 values (0 and 1)
            for vp1_val in [0, 1]:
                vp1_tensor = torch.full((n_vp2_samples, 1), vp1_val, dtype=torch.float32).to(device)
                vp2_tensor = vp2_samples.unsqueeze(1)  # [n_vp2_samples, 1]

                # Expand state to match VP2 samples
                state_expanded = state.expand(n_vp2_samples, -1)  # [n_vp2_samples, state_dim]

                # Concatenate: [state, vp1, vp2]
                action = torch.cat([vp1_tensor, vp2_tensor], dim=1)
                
                # Get Q-values using min(Q1, Q2)
                q1_vals = agent.q1(state_expanded, action).squeeze()
                q2_vals = agent.q2(state_expanded, action).squeeze()
                q_vals = torch.min(q1_vals, q2_vals)

                # Find best VP2 for this VP1
                max_q_for_vp1, max_idx = q_vals.max(dim=0)

                if max_q_for_vp1 > max_q_value:
                    max_q_value = max_q_for_vp1
                    best_vp1_i = vp1_val
                    best_vp2_i = vp2_samples[max_idx].item()

            best_vp1.append(best_vp1_i)
            best_vp2.append(best_vp2_i)

        return np.array(best_vp1), np.array(best_vp2)

# Get model dual mixed actions (VP1 binary + VP2 continuous)
print("Computing model dual mixed actions for training data...")
train_model_vp1, train_model_vp2 = select_action_batch_dual_mixed(
    train_data['states'],
    dual_agent,
    device,
    n_vp2_samples=50
)

print("Computing model dual mixed actions for test data...")
test_model_vp1, test_model_vp2 = select_action_batch_dual_mixed(
    test_data['states'],
    dual_agent,
    device,
    n_vp2_samples=50
)

# Extract clinician actions and clip VP2 to [0.0, 0.5]
# train_data['actions'] shape: [n_transitions, 2] where [:, 0]=VP1, [:, 1]=VP2
train_clinician_vp1 = train_data['actions'][:, 0]  # Binary: 0 or 1
train_clinician_vp2 = np.clip(train_data['actions'][:, 1], 0.0, 0.5)  # Clipped to [0.0, 0.5]

test_clinician_vp1 = test_data['actions'][:, 0]  # Binary: 0 or 1
test_clinician_vp2 = np.clip(test_data['actions'][:, 1], 0.0, 0.5)  # Clipped to [0.0, 0.5]

print(f"\nDual mixed actions generated:")
print(f"  Train model VP1:       {train_model_vp1.shape} (binary)")
print(f"  Train model VP2:       {train_model_vp2.shape} (continuous [0.0, 0.5])")
print(f"  Train clinician VP1:   {train_clinician_vp1.shape} (binary)")
print(f"  Train clinician VP2:   {train_clinician_vp2.shape} (continuous [0.0, 0.5])")
print(f"  Test model VP1:        {test_model_vp1.shape} (binary)")
print(f"  Test model VP2:        {test_model_vp2.shape} (continuous [0.0, 0.5])")
print(f"  Test clinician VP1:    {test_clinician_vp1.shape} (binary)")
print(f"  Test clinician VP2:    {test_clinician_vp2.shape} (continuous [0.0, 0.5])")

# Display sample actions
print(f"\nSample actions (first 5):")
print(f"  Model VP1 (train):     {train_model_vp1[:5]}")
print(f"  Model VP2 (train):     {train_model_vp2[:5]}")
print(f"  Clinician VP1 (train): {train_clinician_vp1[:5]}")
print(f"  Clinician VP2 (train): {train_clinician_vp2[:5]}")

# Show VP1 distribution (binary: 0 or 1)
print(f"\nVP1 distribution (binary: 0 or 1):")
for vp1_val in [0, 1]:
    print(f"  VP1={vp1_val}: Model (train)={np.sum(train_model_vp1==vp1_val)}, "
          f"Clinician (train)={np.sum(train_clinician_vp1==vp1_val)}, "
          f"Model (test)={np.sum(test_model_vp1==vp1_val)}, "
          f"Clinician (test)={np.sum(test_clinician_vp1==vp1_val)}")

# Show VP2 statistics
print(f"\nVP2 statistics (continuous [0.0, 0.5]):")
print(f"  Model (train):     Mean={train_model_vp2.mean():.4f}, Std={train_model_vp2.std():.4f}, Min={train_model_vp2.min():.4f}, Max={train_model_vp2.max():.4f}")
print(f"  Clinician (train): Mean={train_clinician_vp2.mean():.4f}, Std={train_clinician_vp2.std():.4f}, Min={train_clinician_vp2.min():.4f}, Max={train_clinician_vp2.max():.4f}")
print(f"  Model (test):      Mean={test_model_vp2.mean():.4f}, Std={test_model_vp2.std():.4f}, Min={test_model_vp2.min():.4f}, Max={test_model_vp2.max():.4f}")
print(f"  Clinician (test):  Mean={test_clinician_vp2.mean():.4f}, Std={test_clinician_vp2.std():.4f}, Min={test_clinician_vp2.min():.4f}, Max={test_clinician_vp2.max():.4f}") 

## train a logistic classifier (lg_vp1_ma) to predict the vp1 model actions (binary) using the states on the training data, then use it on the test data to get the probability of clinician vp1 actions
## train a logistic classifier (lg_vp1_ca) to predict the vp1 clinician actions (binary) using the states on the training data, then use it on the test data to get the probability of clinician vp1 actions
## train a softmax classifier (sm_vp2_ma) to predict the vp2 model actions (block discrete) using the states on the training data, then use it on the test data to get the probability of clinician vp2 actions
## train a softmax classifier (sm_vp2_ca) to predict the vp2 clinician (block discrete) using the states on the training data, then use it on the test data to get the probability of clinician vp2 actions
## save the test data probabilities, and display some of them (each of the four)

print("\n" + "="*70)
print("TRAINING BEHAVIOR POLICY CLASSIFIERS & REGRESSORS")
print("="*70)

from sklearn.linear_model import LinearRegression, Ridge
from scipy.stats import norm

# ============================================================================
# VP1 Classifiers (Binary: 0 or 1)
# ============================================================================

# 1. Train VP1 classifier for model actions
print("\n1. Training VP1 classifier for model actions (2 classes: 0 or 1)...")
clf_vp1_model = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
clf_vp1_model.fit(train_data['states'], train_model_vp1.astype(int))
train_acc_vp1_model = accuracy_score(train_model_vp1, clf_vp1_model.predict(train_data['states']))
test_acc_vp1_model = accuracy_score(test_model_vp1, clf_vp1_model.predict(test_data['states']))
print(f"   Train accuracy: {train_acc_vp1_model:.4f}")
print(f"   Test accuracy:  {test_acc_vp1_model:.4f}")

# 2. Train VP1 classifier for clinician actions
print("\n2. Training VP1 classifier for clinician actions (2 classes: 0 or 1)...")
clf_vp1_clinician = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
clf_vp1_clinician.fit(train_data['states'], train_clinician_vp1.astype(int))
train_acc_vp1_clinician = accuracy_score(train_clinician_vp1, clf_vp1_clinician.predict(train_data['states']))
test_acc_vp1_clinician = accuracy_score(test_clinician_vp1, clf_vp1_clinician.predict(test_data['states']))
print(f"   Train accuracy: {train_acc_vp1_clinician:.4f}")
print(f"   Test accuracy:  {test_acc_vp1_clinician:.4f}")

# ============================================================================
# VP2 Regressors (Continuous: [0.0, 0.5])
# ============================================================================

# 3. Train VP2 regressor for model actions
print("\n3. Training VP2 regressor for model actions (continuous [0.0, 0.5])...")
reg_vp2_model = Ridge(alpha=1.0, random_state=42)
reg_vp2_model.fit(train_data['states'], train_model_vp2)
train_pred_vp2_model = reg_vp2_model.predict(train_data['states'])
test_pred_vp2_model = reg_vp2_model.predict(test_data['states'])

# Compute residual variance for Gaussian probability density
train_residuals_vp2_model = train_model_vp2 - train_pred_vp2_model
vp2_model_std = np.std(train_residuals_vp2_model)

print(f"   Train RMSE: {np.sqrt(np.mean(train_residuals_vp2_model**2)):.4f}")
print(f"   Test RMSE:  {np.sqrt(np.mean((test_model_vp2 - test_pred_vp2_model)**2)):.4f}")
print(f"   Residual std: {vp2_model_std:.4f}")

# 4. Train VP2 regressor for clinician actions
print("\n4. Training VP2 regressor for clinician actions (continuous [0.0, 0.5])...")
reg_vp2_clinician = Ridge(alpha=1.0, random_state=42)
reg_vp2_clinician.fit(train_data['states'], train_clinician_vp2)
train_pred_vp2_clinician = reg_vp2_clinician.predict(train_data['states'])
test_pred_vp2_clinician = reg_vp2_clinician.predict(test_data['states'])

# Compute residual variance for Gaussian probability density
train_residuals_vp2_clinician = train_clinician_vp2 - train_pred_vp2_clinician
vp2_clinician_std = np.std(train_residuals_vp2_clinician)

print(f"   Train RMSE: {np.sqrt(np.mean(train_residuals_vp2_clinician**2)):.4f}")
print(f"   Test RMSE:  {np.sqrt(np.mean((test_clinician_vp2 - test_pred_vp2_clinician)**2)):.4f}")
print(f"   Residual std: {vp2_clinician_std:.4f}")

# Get probabilities on test data
print("\n" + "="*70)
print("COMPUTING BEHAVIOR POLICY PROBABILITIES")
print("="*70)

# ============================================================================
# VP1 Probabilities (from logistic classifiers)
# ============================================================================

# Model VP1 policy: π_model(VP1_clinician | s)
test_probs_vp1_model = clf_vp1_model.predict_proba(test_data['states'])
test_prob_vp1_model = test_probs_vp1_model[np.arange(len(test_clinician_vp1)), test_clinician_vp1.astype(int)]

# Clinician VP1 policy: π_clinician(VP1_clinician | s)
test_probs_vp1_clinician = clf_vp1_clinician.predict_proba(test_data['states'])
test_prob_vp1_clinician = test_probs_vp1_clinician[np.arange(len(test_clinician_vp1)), test_clinician_vp1.astype(int)]

print(f"\nVP1 probability statistics:")
print(f"  Model π(VP1_clinician|s):     Mean={test_prob_vp1_model.mean():.4f}, Std={test_prob_vp1_model.std():.4f}")
print(f"  Clinician π(VP1_clinician|s): Mean={test_prob_vp1_clinician.mean():.4f}, Std={test_prob_vp1_clinician.std():.4f}")

# ============================================================================
# VP2 Probability Densities (Gaussian assumption)
# ============================================================================

# Model VP2 policy: p_model(VP2_clinician | s) using Gaussian
test_pred_vp2_model_test = reg_vp2_model.predict(test_data['states'])
test_prob_vp2_model = norm.pdf(test_clinician_vp2, loc=test_pred_vp2_model_test, scale=vp2_model_std + 1e-6)

# Clinician VP2 policy: p_clinician(VP2_clinician | s) using Gaussian
test_pred_vp2_clinician_test = reg_vp2_clinician.predict(test_data['states'])
test_prob_vp2_clinician = norm.pdf(test_clinician_vp2, loc=test_pred_vp2_clinician_test, scale=vp2_clinician_std + 1e-6)

print(f"\nVP2 probability density statistics:")
print(f"  Model p(VP2_clinician|s):     Mean={test_prob_vp2_model.mean():.4f}, Std={test_prob_vp2_model.std():.4f}")
print(f"  Clinician p(VP2_clinician|s): Mean={test_prob_vp2_clinician.mean():.4f}, Std={test_prob_vp2_clinician.std():.4f}")

# ============================================================================
# Combined IS Weights: π(VP1) × p(VP2)
# ============================================================================

print("\n" + "="*70)
print("COMPUTING IMPORTANCE SAMPLING WEIGHTS")
print("="*70)

# Set epsilon for numerical stability
eps = 1e-10

# IS weight = [π_model(VP1) × p_model(VP2)] / [π_clinician(VP1) × p_clinician(VP2)]
is_weight_numerator = test_prob_vp1_model * test_prob_vp2_model
is_weight_denominator = test_prob_vp1_clinician * test_prob_vp2_clinician
is_weight = is_weight_numerator / (is_weight_denominator + eps)

print(f"\nIS weight statistics:")
print(f"  Mean: {is_weight.mean():.4f}, Std: {is_weight.std():.4f}")
print(f"  Min: {is_weight.min():.4f}, Max: {is_weight.max():.4f}")

print(f"\nSample probabilities (first 5 test transitions):")
print(f"  Index | VP1_model | VP1_clin | VP2_model | VP2_clin | IS_weight")
print(f"  " + "-"*80)
for i in range(min(5, len(is_weight))):
    print(f"  {i:5d} | {test_prob_vp1_model[i]:9.4f} | {test_prob_vp1_clinician[i]:8.4f} | "
          f"{test_prob_vp2_model[i]:9.4f} | {test_prob_vp2_clinician[i]:8.4f} | {is_weight[i]:9.4f}") 


# Finally for each of the transitions (indices) in the test data, compute pi_lg_vp1_ma(s) / (eps + pi_lg_vp1_ca(s)) * pi_sm_vp2_ma(s) / (esp + pi_sm_vp2_ca(s)) * R where R is the reward on that transition index or state, this is the expected reward using the model actions on that state
# then compute the per transition average of the expected reward (using the model recommended actions), as well as the per patient average of the expected reward (using the model recommended actions)
# compare the two values with the average reward (the raw reward) per transition, and the average reward per patient
# print out the results and save it in latex table in a separate latex file

print("\n" + "="*70)
print("COMPUTING IMPORTANCE SAMPLING WEIGHTS AND REWARDS")
print("="*70)

# Set epsilon for numerical stability
eps = 1e-10

# IS weight = [π_model(VP1) × p_model(VP2)] / [π_clinician(VP1) × p_clinician(VP2)]
is_weight_numerator = test_prob_vp1_model * test_prob_vp2_model
is_weight_denominator = test_prob_vp1_clinician * test_prob_vp2_clinician
is_weight = is_weight_numerator / (is_weight_denominator + eps)

isw_ci_diff_lower = np.percentile(is_weight, 2.5)
isw_ci_diff_upper = np.percentile(is_weight, 97.5)

#is_weight = np.clip(is_weight, a_min = CLIP_MIN, a_max = CLIP_MAX)
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
    
    # Get weights and rewards for this trajectory
    patient_weights = is_weight[patient_mask]
    patient_rewards = test_rewards[patient_mask]
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
