# vaso_rl

Code for *"Realistic CDSS Drug Dosing with End-to-end Recurrent Q-learning for Dual Vasopressor Control"* (Zou et al.), NeurIPS 2025 Workshop on Learning from Time Series for Health (TS4H). [arXiv:2510.01508](https://arxiv.org/abs/2510.01508)

## Overview

Reinforcement learning (RL) for Clinical Decision Support Systems (CDSS) is often met with skepticism from clinicians when learned policies recommend dosing decisions that are impractical to operationalize at the bedside. This repository implements an end-to-end offline RL framework for **dual vasopressor control** — the joint titration of norepinephrine (first-line) and vasopressin (second-line) — in ICU patients with septic shock. Unlike prior work that has largely focused on single-vasopressor control during early treatment phases, our approach targets realistic dosing behavior across the full therapy trajectory (initiation, titration, and weaning).

The core contribution is a systematic study of **action space design** as a bridge between RL optimization and clinical adoption. We compare discrete, continuous, and directional (stepwise) formulations of the dosing action space, combining offline Conservative Q-Learning (CQL) with a recurrent replay buffer that captures temporal dependencies in ICU time-series data. Empirically, discrete and directional-discrete action spaces yield more interpretable policies while preserving — and in several settings improving — expected outcomes under off-policy evaluation.

## Setup

Place `sample_data_oviss.csv` in the `vaso_rl/` directory.

## Training

Binary Q-learning with CQL penalty (alpha specified in script):
```bash
python3 run_binary_cql_alpha00.py
```

Dual Mixed Q-learning with CQL penalty:
```bash
python3 run_dualmixed_cql_allalphas.py
```

Block Discrete (BD) Q-learning:
```bash
python3 run_block_discrete_cql_allalphas.py
```

Directional Stepwise Q-learning:
```bash
python3 run_unified_stepwise_cql_allalphas.py --alpha 0.0 --max_step 0.2
```

LSTM Block Discrete Q-learning:
```bash
python3 run_lstm_block_discrete_cql_with_logging.py
```

## Off-policy Evaluation

**Fitted Q-Evaluation (FQE).** See `ope_exp.py` and `fqe_gaussian_analysis.py`. These scripts consume the `.pkl` checkpoints saved during training.

**Weighted Importance Sampling (WIS).** See `is_block_discrete.py` for an example of evaluating the Block Discrete model. Equivalent scripts are provided for the other models: `is_binary.py`, `is_dual_mixed.py`, and `is_lstm_bd.py`.

## Reward Functions

- **NeurIPS (simple) reward:** defined in `integrated_data_pipeline_v2_simple_reward.py`
- **OVISS reward:** defined in `integrated_data_pipeline_v2.py`

## Hyperparameters

**NeurIPS (simple reward):**

| Hyperparameter | Value | Description |
|---|---|---|
| Learning rate | 1e-3 | Adam optimizer learning rate |
| Batch size | 128 | Training batch size |
| Gamma (γ) | 0.95 | Discount factor |
| Tau (τ) | 0.8 | Soft target network update rate |
| Gradient clipping | 1.0 | Maximum gradient norm |
| Epochs | 100 | Training epochs |
| Validation batches | 10 | Batches used for validation |
| Random seed | 42 | For reproducibility |

**OVISS reward:**

| Hyperparameter | Value | Description |
|---|---|---|
| Learning rate | 1e-3 | Adam optimizer learning rate |
| Batch size | 128 | Training batch size |
| Gamma (γ) | 0.99 | Discount factor (aligned with OVISS) |
| Tau (τ) | 0.95 | Soft target network update rate (aligned with OVISS FQI setup) |
| Gradient clipping | 1.0 | Maximum gradient norm |
| Epochs | 100 | Training epochs |
| Validation batches | 10 | Batches used for validation |
| Random seed | 42 | For reproducibility |

## Citation

```bibtex
@inproceedings{zou2025vasorl,
  title     = {Realistic CDSS Drug Dosing with End-to-end Recurrent Q-learning for Dual Vasopressor Control},
  author    = {Zou, Will Y. and Feng, Jean and Kalimouttou, Alexandre and Zhang, Jennifer Yuntong and Seymour, Christopher W. and Pirracchio, Romain},
  booktitle = {NeurIPS 2025 Workshop on Learning from Time Series for Health (TS4H)},
  year      = {2025},
  url       = {https://arxiv.org/abs/2510.01508}
}
```
