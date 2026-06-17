"""
Analyze learned reward distributions across patient subgroups.

The script expects rewards extracted by extract_irl_rewards.py, e.g.
irl_analysis/all_irl_rewards_test.pkl. It writes patient-level reward summaries,
subgroup aggregate tables, and optional boxplots.
"""

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import data_config as config


DEFAULT_REWARDS_PATH = "irl_analysis/all_irl_rewards_test.pkl"
DEFAULT_OUTPUT_DIR = "irl_analysis/subgroups/reward_distributions"
NON_REWARD_KEYS = {"metadata", "trajectory_ids", "trajectory_lengths"}


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze reward distributions by subgroup.")
    parser.add_argument("--data-path", default=config.DATA_PATH, help="CSV data path.")
    parser.add_argument("--rewards-pickle", default=DEFAULT_REWARDS_PATH, help="Extracted rewards pickle.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for CSVs and figures.")
    parser.add_argument("--age-bins", default="0,50,65,80,200", help="Comma-separated age bin edges.")
    parser.add_argument("--age-labels", default="<50,50-64,65-79,80+", help="Comma-separated age labels.")
    parser.add_argument("--no-plots", action="store_true", help="Skip boxplot generation.")
    return parser.parse_args()


def load_rewards(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Reward pickle not found: {path}. Run extract_irl_rewards.py first or pass --rewards-pickle."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def reward_model_names(rewards_data):
    names = []
    for key, value in rewards_data.items():
        if key not in NON_REWARD_KEYS and isinstance(value, dict):
            names.append(key)
    return names


def parse_age_bins(edges_text, labels_text):
    edges = [float(x.strip()) for x in edges_text.split(",") if x.strip()]
    labels = [x.strip() for x in labels_text.split(",") if x.strip()]
    if len(labels) != len(edges) - 1:
        raise ValueError("--age-labels must contain exactly len(--age-bins)-1 labels.")
    return edges, labels


def load_patient_metadata(data_path, patient_ids, age_edges, age_labels):
    data = pd.read_csv(data_path)
    rows = []
    for patient_id, patient_data in data.groupby(config.PATIENT_ID_COL, sort=False):
        if patient_id not in patient_ids:
            continue
        first = patient_data.iloc[0]
        last = patient_data.iloc[-1]
        age = float(first["age"])
        rows.append(
            {
                "patient_id": int(patient_id),
                "age": age,
                "gender": str(first["gender"]),
                "ethnicity": str(first["ethnicity"]),
                "outcome": int(last[config.DEATH_COL]),
                "outcome_label": "survivor" if int(last[config.DEATH_COL]) == 0 else "non_survivor",
                "n_timesteps": int(len(patient_data)),
                "mean_sofa": float(patient_data["sofa"].mean()),
                "mean_mbp": float(patient_data["mbp"].mean()),
                "mean_lactate": float(patient_data["lactate"].mean()),
            }
        )

    metadata = pd.DataFrame(rows)
    metadata["age_group"] = pd.cut(
        metadata["age"],
        bins=age_edges,
        labels=age_labels,
        right=False,
        include_lowest=True,
    ).astype(str)
    return metadata


def build_patient_reward_table(rewards_data, metadata):
    meta_by_pid = metadata.set_index("patient_id")
    rows = []
    for model_name in reward_model_names(rewards_data):
        for patient_id, rewards in rewards_data[model_name].items():
            patient_id = int(patient_id)
            if patient_id not in meta_by_pid.index:
                continue
            values = np.asarray(rewards, dtype=float)
            if values.size == 0:
                continue
            row = meta_by_pid.loc[patient_id].to_dict()
            row.update(
                {
                    "patient_id": patient_id,
                    "model": model_name,
                    "reward_sum": float(values.sum()),
                    "reward_mean": float(values.mean()),
                    "reward_std": float(values.std()),
                    "n_rewards": int(values.size),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_group(values):
    q25, q75 = np.percentile(values, [25, 75])
    return pd.Series(
        {
            "n_patients": int(values.size),
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
            "median": float(np.median(values)),
            "iqr": float(q75 - q25),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    )


def build_aggregate_tables(patient_rewards):
    subgroup_cols = ["age_group", "gender", "ethnicity"]
    tables = []
    for subgroup_col in subgroup_cols:
        for reward_metric in ["reward_sum", "reward_mean"]:
            grouped = (
                patient_rewards.groupby(["model", subgroup_col], dropna=False)[reward_metric]
                .apply(lambda x: summarize_group(x.to_numpy(dtype=float)))
                .reset_index()
            )
            grouped["subgroup"] = subgroup_col
            grouped = grouped.rename(columns={subgroup_col: "subgroup_value", "level_2": "stat"})
            pivoted = grouped.pivot_table(
                index=["model", "subgroup", "subgroup_value"],
                columns="stat",
                values=reward_metric,
                aggfunc="first",
            ).reset_index()
            pivoted["reward_metric"] = reward_metric
            tables.append(pivoted)

            by_outcome = (
                patient_rewards.groupby(["model", "outcome_label", subgroup_col], dropna=False)[reward_metric]
                .apply(lambda x: summarize_group(x.to_numpy(dtype=float)))
                .reset_index()
            )
            by_outcome["subgroup"] = subgroup_col
            by_outcome = by_outcome.rename(columns={subgroup_col: "subgroup_value", "level_3": "stat"})
            by_outcome_pivot = by_outcome.pivot_table(
                index=["model", "outcome_label", "subgroup", "subgroup_value"],
                columns="stat",
                values=reward_metric,
                aggfunc="first",
            ).reset_index()
            by_outcome_pivot["reward_metric"] = reward_metric
            tables.append(by_outcome_pivot)
    return pd.concat(tables, ignore_index=True, sort=False)


def write_boxplots(patient_rewards, output_dir):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping boxplots.")
        return

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    for subgroup_col in ["age_group", "gender", "ethnicity"]:
        for model_name, model_df in patient_rewards.groupby("model"):
            fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
            model_df.boxplot(column="reward_sum", by=subgroup_col, ax=ax, rot=30)
            ax.set_title(f"{model_name}: trajectory reward by {subgroup_col}")
            ax.set_xlabel(subgroup_col)
            ax.set_ylabel("Trajectory reward sum")
            fig.suptitle("")
            safe_model = model_name.replace("/", "_")
            fig.savefig(figure_dir / f"{safe_model}_reward_sum_by_{subgroup_col}.png", dpi=200)
            plt.close(fig)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rewards_data = load_rewards(args.rewards_pickle)
    patient_ids = {int(pid) for model in reward_model_names(rewards_data) for pid in rewards_data[model].keys()}
    age_edges, age_labels = parse_age_bins(args.age_bins, args.age_labels)
    metadata = load_patient_metadata(args.data_path, patient_ids, age_edges, age_labels)
    patient_rewards = build_patient_reward_table(rewards_data, metadata)
    if patient_rewards.empty:
        raise ValueError("No patient rewards matched the CSV metadata.")

    aggregate = build_aggregate_tables(patient_rewards)

    patient_path = output_dir / "patient_reward_summaries.csv"
    aggregate_path = output_dir / "subgroup_reward_distribution_summary.csv"
    patient_rewards.to_csv(patient_path, index=False)
    aggregate.to_csv(aggregate_path, index=False)

    if not args.no_plots:
        write_boxplots(patient_rewards, output_dir)

    print(f"Wrote patient-level rewards: {patient_path}")
    print(f"Wrote subgroup summaries: {aggregate_path}")


if __name__ == "__main__":
    main()
