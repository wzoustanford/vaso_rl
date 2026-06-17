#!/bin/bash
# Run U-Net-labelled and transformer-labelled CQL experiments with mortality-only rewards.
#
# Mortality reward definition:
#   terminal transition for patients who died = 1.0
#   all other transitions = 0.0
#
# Usage:
#   ./run_mortality_reward_experiments.sh [options passed to both experiment scripts]
#
# Examples:
#   ./run_mortality_reward_experiments.sh --test
#   ./run_mortality_reward_experiments.sh --ql_epochs 100 --vp2_bins 10
#   ./run_mortality_reward_experiments.sh --use_lstm --ql_epochs 500

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=============================================="
echo "Mortality-Only Reward Experiments"
echo "=============================================="
echo "Reward: binary terminal mortality signal"
echo "  death at terminal transition: 1.0"
echo "  otherwise: 0.0"
echo "=============================================="

echo ""
echo "Running U-Net-labelled mortality-only experiment..."
"${SCRIPT_DIR}/run_experiment.sh" unet --mortality_reward_only "$@"

echo ""
echo "Running transformer-labelled mortality-only experiment..."
"${SCRIPT_DIR}/run_experiment_trans.sh" --mortality_reward_only "$@"

echo ""
echo "Mortality-only reward experiments complete."
