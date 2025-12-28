#!/bin/bash
# Experiment pipeline for IRL reward learning + Q-Learning + WIS evaluation
# Usage: ./run_experiment.sh <algorithm> [options]
# Example: ./run_experiment.sh gcl --test
#          ./run_experiment.sh maxent --irl_epochs 50 --ql_epochs 100
#          ./run_experiment.sh iq_learn --iq_init_temp 0.01 --iq_tau 0.01 --iq_lr 1e-3 --iq_div chi

set -e  # Exit on error

# Default values
ALGORITHM="${1:-gcl}"
IRL_EPOCHS=100
QL_EPOCHS=100
VP2_BINS=5
TEST_MODE=false
SUFFIX=""

# GCL-specific defaults
GCL_TAU=0.005
GCL_LR=1e-4
GCL_COST_LR=1e-2

# IQ-Learn-specific defaults
IQ_INIT_TEMP=0.001
IQ_TAU=0.005
IQ_LR=1e-4
IQ_DIV="chi"

# Parse arguments
shift  # Remove first argument (algorithm)
while [[ $# -gt 0 ]]; do
    case $1 in
        --test)
            TEST_MODE=true
            IRL_EPOCHS=2
            QL_EPOCHS=2
            SUFFIX="_test"
            shift
            ;;
        --irl_epochs)
            IRL_EPOCHS="$2"
            shift 2
            ;;
        --ql_epochs)
            QL_EPOCHS="$2"
            shift 2
            ;;
        --vp2_bins)
            VP2_BINS="$2"
            shift 2
            ;;
        --suffix)
            SUFFIX="$2"
            shift 2
            ;;
        --gcl_tau)
            GCL_TAU="$2"
            shift 2
            ;;
        --gcl_lr)
            GCL_LR="$2"
            shift 2
            ;;
        --gcl_cost_lr)
            GCL_COST_LR="$2"
            shift 2
            ;;
        --iq_init_temp)
            IQ_INIT_TEMP="$2"
            shift 2
            ;;
        --iq_tau)
            IQ_TAU="$2"
            shift 2
            ;;
        --iq_lr)
            IQ_LR="$2"
            shift 2
            ;;
        --iq_div)
            IQ_DIV="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "Experiment Pipeline"
echo "=============================================="
echo "Algorithm: $ALGORITHM"
echo "IRL Epochs: $IRL_EPOCHS"
echo "QL Epochs: $QL_EPOCHS"
echo "VP2 Bins: $VP2_BINS"
echo "Test Mode: $TEST_MODE"
echo "Suffix: $SUFFIX"
if [ "$ALGORITHM" == "gcl" ]; then
    echo "GCL tau: $GCL_TAU"
    echo "GCL lr: $GCL_LR"
    echo "GCL cost_lr: $GCL_COST_LR"
fi
if [ "$ALGORITHM" == "iq_learn" ]; then
    echo "IQ init_temp: $IQ_INIT_TEMP"
    echo "IQ tau: $IQ_TAU"
    echo "IQ lr: $IQ_LR"
    echo "IQ div: $IQ_DIV"
fi
echo "=============================================="

# Define paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="${SCRIPT_DIR}/experiment"
IRL_DIR="${EXPERIMENT_DIR}/irl"
QL_DIR="${EXPERIMENT_DIR}/ql"
RESULTS_DIR="${EXPERIMENT_DIR}/irl_results"

# Create directories
mkdir -p "$IRL_DIR"
mkdir -p "$QL_DIR"
mkdir -p "$RESULTS_DIR"

# Step 1: IRL Training (skip for manual)
if [ "$ALGORITHM" != "manual" ]; then
    echo ""
    echo "=============================================="
    echo "Step 1: IRL Training ($ALGORITHM)"
    echo "=============================================="

    case $ALGORITHM in
        maxent)
            python "${SCRIPT_DIR}/maxent_irl_full_recovery.py" \
                --epochs "$IRL_EPOCHS" \
                --batch_size 128 \
                --prefix "maxent${SUFFIX}" \
                --save_dir "$IRL_DIR"
            REWARD_MODEL_PATH="${IRL_DIR}/maxent${SUFFIX}_reward_model.pt"
            ;;
        gcl)
            python "${SCRIPT_DIR}/run_gcl_learn_block_discrete.py" \
                --epochs "$IRL_EPOCHS" \
                --tau "$GCL_TAU" \
                --lr "$GCL_LR" \
                --cost_lr "$GCL_COST_LR" \
                --prefix "gcl${SUFFIX}" \
                --save_dir "$IRL_DIR"
            REWARD_MODEL_PATH="${IRL_DIR}/gcl${SUFFIX}_cost_model.pt"
            ;;
        iq_learn)
            python "${SCRIPT_DIR}/run_iq_learn_block_discrete.py" \
                --epochs "$IRL_EPOCHS" \
                --init_temp "$IQ_INIT_TEMP" \
                --tau "$IQ_TAU" \
                --lr "$IQ_LR" \
                --div "$IQ_DIV" \
                --prefix "iq_learn${SUFFIX}" \
                --save_dir "$IRL_DIR"
            REWARD_MODEL_PATH="${IRL_DIR}/iq_learn${SUFFIX}_q_model.pt"
            ;;
        *)
            echo "Unknown algorithm: $ALGORITHM"
            exit 1
            ;;
    esac

    echo "IRL training complete. Model saved to: $REWARD_MODEL_PATH"
else
    echo ""
    echo "=============================================="
    echo "Step 1: Skipping IRL Training (using manual reward)"
    echo "=============================================="
    REWARD_MODEL_PATH=""
fi

# Step 2: Q-Learning
echo ""
echo "=============================================="
echo "Step 2: Q-Learning Training"
echo "=============================================="

if [ -z "$REWARD_MODEL_PATH" ]; then
    # Manual reward
    python "${SCRIPT_DIR}/run_block_discrete_cql_allalphas.py" \
        --single_alpha 0.0 \
        --vp2_bins "$VP2_BINS" \
        --epochs "$QL_EPOCHS" \
        --suffix "${SUFFIX}" \
        --save_dir "$QL_DIR"
else
    # Learned reward
    python "${SCRIPT_DIR}/run_block_discrete_cql_allalphas.py" \
        --single_alpha 0.0 \
        --vp2_bins "$VP2_BINS" \
        --epochs "$QL_EPOCHS" \
        --reward_model_path "$REWARD_MODEL_PATH" \
        --suffix "${SUFFIX}" \
        --save_dir "$QL_DIR"
fi

# Find the saved Q-learning model (alpha format is 0.0000)
QL_MODEL_PATH="${QL_DIR}/${ALGORITHM}${SUFFIX}_alpha0.0000_bins${VP2_BINS}_best.pt"
echo "Q-Learning complete. Model saved to: $QL_MODEL_PATH"

# Step 3: WIS Evaluation
echo ""
echo "=============================================="
echo "Step 3: WIS Evaluation"
echo "=============================================="

# Results file named after the model (alpha format is 0.0000)
RESULTS_FILE="${RESULTS_DIR}/${ALGORITHM}${SUFFIX}_alpha0.0000_bins${VP2_BINS}_best_wis.txt"

python "${SCRIPT_DIR}/is_block_discrete.py" \
    --model_path "$QL_MODEL_PATH" \
    --vp2_bins "$VP2_BINS" \
    2>&1 | tee "$RESULTS_FILE"

echo ""
echo "=============================================="
echo "Experiment Complete!"
echo "=============================================="
echo "Algorithm: $ALGORITHM"
echo "IRL Model: ${REWARD_MODEL_PATH:-N/A (manual)}"
echo "QL Model: $QL_MODEL_PATH"
echo "Results: $RESULTS_FILE"
echo "=============================================="
