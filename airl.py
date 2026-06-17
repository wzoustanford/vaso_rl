from __future__ import annotations

import argparse
import os
import numpy as np
import pandas as pd
import gymnasium as gym

import torch
import torch.nn as nn
import torch.optim as optim

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env as sb3_make_vec_env

from imitation.algorithms.adversarial.airl import AIRL
from imitation.data.types import Trajectory
from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.util.networks import RunningNorm

import data_config as config


def parse_args():
    parser = argparse.ArgumentParser(description="Train AIRL reward model and save pipeline-compatible checkpoint")
    parser.add_argument("--csv_path", type=str, default=config.DATA_PATH, help="Path to training CSV")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n_envs", type=int, default=8, help="Number of vectorized environments")
    parser.add_argument("--horizon", type=int, default=6, help="Episode horizon for learned dynamics env")
    parser.add_argument("--dyn_epochs", type=int, default=10, help="Dynamics model training epochs")
    parser.add_argument("--airl_steps", type=int, default=200_000, help="AIRL training timesteps")
    parser.add_argument("--save_dir", type=str, default="experiment/airl", help="Directory to save AIRL checkpoint")
    parser.add_argument("--prefix", type=str, default="airl", help="Checkpoint filename prefix")
    parser.add_argument(
        "--action_schema",
        choices=["pipeline_dual", "norepi_fluid"],
        default="pipeline_dual",
        help="Action schema used for AIRL training",
    )
    parser.add_argument(
        "--state_schema",
        choices=["pipeline_dual", "auto_numeric"],
        default="pipeline_dual",
        help="State feature selection",
    )
    return parser.parse_args()


def select_columns(df: pd.DataFrame, state_schema: str, action_schema: str):
    if action_schema == "pipeline_dual":
        action_cols = list(config.DUAL_ACTIONS)
    else:
        action_cols = ["norepinephrine", "fluid"]

    if state_schema == "pipeline_dual":
        state_cols = list(config.DUAL_STATE_FEATURES)
    else:
        exclude = set(
            [
                config.PATIENT_ID_COL,
                config.TIME_COL,
                config.DEATH_COL,
                config.OPTIMAL_ACTION_COL,
                "base",
                "concordance",
            ] + action_cols
        )
        state_cols = [
            c for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
        ]

    missing = [c for c in (state_cols + action_cols) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    # Deterministically encode non-numeric columns (e.g., ethnicity, gender)
    # using sorted string categories, matching data_loader.py behavior.
    non_numeric = [c for c in (state_cols + action_cols) if not pd.api.types.is_numeric_dtype(df[c])]
    for col in non_numeric:
        str_values = df[col].astype(str)
        unique_values = sorted(str_values.unique())
        mapping = {val: i for i, val in enumerate(unique_values)}
        df[col] = str_values.map(mapping).astype(np.float32)
        print(f"Encoded {col}: {unique_values} -> {list(range(len(unique_values)))}")

    return state_cols, action_cols


args = parse_args()
CSV_PATH = args.csv_path
SEED = args.seed
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
rng = np.random.default_rng(SEED)

# -----------------------------
# 1) Load CSV and build (s,a,s') transitions
# -----------------------------

df = pd.read_csv(CSV_PATH)
STATE_COLS, ACTION_COLS = select_columns(df, args.state_schema, args.action_schema)

print("CSV path:", CSV_PATH)
print("State schema:", args.state_schema)
print("Action schema:", args.action_schema)
print("State cols:", STATE_COLS)
print("Action cols:", ACTION_COLS)

# Sort by subject/time and build next-state by shifting within each subject
df = df.sort_values([config.PATIENT_ID_COL, config.TIME_COL]).reset_index(drop=True)

# next state within subject
df_next = df.groupby(config.PATIENT_ID_COL)[STATE_COLS].shift(-1)

# Keep only rows that have a valid next-state (i.e., not last row)
mask = ~df_next.isna().any(axis=1)
df = df.loc[mask].reset_index(drop=True)
df_next = df_next.loc[mask].reset_index(drop=True)

S = df[STATE_COLS].to_numpy(dtype=np.float32)
A = df[ACTION_COLS].to_numpy(dtype=np.float32)
SP = df_next.to_numpy(dtype=np.float32)

state_dim = S.shape[1]
action_dim = A.shape[1]
print("Transitions:", S.shape[0], "state_dim:", state_dim, "action_dim:", action_dim)

# Simple normalization (important!)
s_mean, s_std = S.mean(axis=0), S.std(axis=0) + 1e-6
a_mean, a_std = A.mean(axis=0), A.std(axis=0) + 1e-6

S_n = (S - s_mean) / s_std
SP_n = (SP - s_mean) / s_std
A_n = (A - a_mean) / a_std


# -----------------------------
# 2) Train learned dynamics model: (s,a) -> s'
# -----------------------------

class DynamicsModel(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden=(256, 256)):
        super().__init__()
        in_dim = state_dim + action_dim
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, state_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, sa: torch.Tensor) -> torch.Tensor:
        return self.net(sa)


dyn = DynamicsModel(state_dim, action_dim).to(DEVICE)
opt = optim.Adam(dyn.parameters(), lr=3e-4)

# train/val split
N = S_n.shape[0]
idx = rng.permutation(N)
split = int(0.9 * N)
tr, va = idx[:split], idx[split:]

S_tr, A_tr, SP_tr = S_n[tr], A_n[tr], SP_n[tr]
S_va, A_va, SP_va = S_n[va], A_n[va], SP_n[va]

def batch_iter(Sb, Ab, SPb, batch_size=2048):
    n = Sb.shape[0]
    order = rng.permutation(n)
    for i in range(0, n, batch_size):
        j = order[i:i+batch_size]
        yield Sb[j], Ab[j], SPb[j]

for epoch in range(args.dyn_epochs):
    dyn.train()
    losses = []
    for sb, ab, spb in batch_iter(S_tr, A_tr, SP_tr):
        sb_t = torch.from_numpy(sb).to(DEVICE)
        ab_t = torch.from_numpy(ab).to(DEVICE)
        sp_t = torch.from_numpy(spb).to(DEVICE)

        pred = dyn(torch.cat([sb_t, ab_t], dim=-1))
        loss = ((pred - sp_t) ** 2).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    dyn.eval()
    with torch.no_grad():
        sb_t = torch.from_numpy(S_va).to(DEVICE)
        ab_t = torch.from_numpy(A_va).to(DEVICE)
        sp_t = torch.from_numpy(S_va*0 + SP_va).to(DEVICE)
        pred = dyn(torch.cat([sb_t, ab_t], dim=-1))
        val_loss = ((pred - sp_t) ** 2).mean().item()

    print(f"[dyn] epoch {epoch:02d} train_mse={np.mean(losses):.6f} val_mse={val_loss:.6f}")


# -----------------------------
# 3) Gymnasium env backed by learned dynamics
# -----------------------------

class LearnedDynamicsEnv(gym.Env):
    """
    Env state is normalized s. Actions are normalized a (2D).
    step(): s' = dyn([s,a]), horizon-limited episodes.
    """
    metadata = {"render_modes": []}

    def __init__(self, dyn_model: nn.Module, init_states: np.ndarray, horizon: int = 6):
        super().__init__()
        self.dyn = dyn_model
        self.init_states = init_states
        self.horizon = horizon
        self.t = 0
        self.s = None

        self.observation_space = gym.spaces.Box(
            low=-10, high=10, shape=(state_dim,), dtype=np.float32
        )
        # In normalized space, typical values are roughly ~N(0,1); clamp to something sane.
        self.action_space = gym.spaces.Box(
            low=-5, high=5, shape=(action_dim,), dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.s = self.init_states[rng.integers(0, self.init_states.shape[0])].copy()
        return self.s, {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        with torch.no_grad():
            s_t = torch.from_numpy(self.s).to(DEVICE).float().view(1, -1)
            a_t = torch.from_numpy(action).to(DEVICE).float().view(1, -1)
            sp = self.dyn(torch.cat([s_t, a_t], dim=-1)).view(-1).cpu().numpy().astype(np.float32)

        self.s = sp
        self.t += 1

        terminated = False
        truncated = self.t >= self.horizon

        # AIRL learns reward; env reward can be 0.
        reward = 0.0
        info = {}
        return self.s, reward, terminated, truncated, info


# make vec env for SB3 + imitation
def make_env():
    return LearnedDynamicsEnv(dyn, init_states=S_n, horizon=args.horizon)

venv = sb3_make_vec_env(make_env, n_envs=args.n_envs, seed=SEED)
print("Env observation space:", venv.observation_space)
print("Env action space:", venv.action_space)


# -----------------------------
# 4) Convert CSV trajectories into imitation Trajectory demonstrations
# -----------------------------

# Build per-subject trajectories in normalized space
traj_list = []
for sid, g in df.groupby(config.PATIENT_ID_COL, sort=False):
    g = g.sort_values(config.TIME_COL)
    s = ((g[STATE_COLS].to_numpy(np.float32) - s_mean) / s_std)
    a = ((g[ACTION_COLS].to_numpy(np.float32) - a_mean) / a_std)

    # Need obs length = acts length + 1 for Trajectory
    # We can append the next-state from our precomputed shift (or just repeat last)
    sp = ((g[STATE_COLS].shift(-1).ffill().to_numpy(np.float32) - s_mean) / s_std)
    obs = np.vstack([s, sp[-1:]])  # (T+1, state_dim)
    acts = a  # (T, action_dim)

    # "terminal" here is true at end of sequence
    traj_list.append(Trajectory(obs=obs, acts=acts, infos=None, terminal=True))

print("Num demonstrations:", len(traj_list))


# -----------------------------
# 5) AIRL training with PPO generator
# -----------------------------

# policy learner
learner = PPO(
    env=venv,
    policy=MlpPolicy,
    n_steps=2048,
    batch_size=2048,
    ent_coef=0.001,
    learning_rate=3e-4,
    gamma=0.99,
    n_epochs=5,
    seed=SEED,
)

# reward network for AIRL: disciminator
reward_net = BasicShapedRewardNet(
    observation_space=venv.observation_space,
    action_space=venv.action_space,
    normalize_input_layer=RunningNorm,
)

airl_trainer = AIRL(
    demonstrations=traj_list,
    demo_batch_size=2048,
    gen_replay_buffer_capacity=50_000,
    n_disc_updates_per_round=16,
    venv=venv,
    gen_algo=learner,
    reward_net=reward_net,
)

airl_trainer.train(args.airl_steps)
print("AIRL training complete.")

# -----------------------------
# 6) Save AIRL reward checkpoint (pipeline-compatible)
# -----------------------------
os.makedirs(args.save_dir, exist_ok=True)
checkpoint_path = os.path.join(args.save_dir, f"{args.prefix}_reward_model.pt")

checkpoint = {
    "format": "airl_reward_net_v1",
    "reward_net_state_dict": reward_net.state_dict(),
    "state_dim": state_dim,
    "action_dim": action_dim,
    "state_cols": STATE_COLS,
    "action_cols": ACTION_COLS,
    "state_mean": s_mean.astype(np.float32),
    "state_std": s_std.astype(np.float32),
    "action_mean": a_mean.astype(np.float32),
    "action_std": a_std.astype(np.float32),
    "seed": SEED,
    "csv_path": CSV_PATH,
}

torch.save(checkpoint, checkpoint_path)
print(f"Saved AIRL reward checkpoint: {checkpoint_path}")
