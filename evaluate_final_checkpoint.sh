#!/bin/bash
# Evaluate an existing Q-learning final checkpoint without retraining.
# Usage: ./evaluate_final_checkpoint.sh <algorithm> [options]
#
# Examples:
#   ./evaluate_final_checkpoint.sh manual
#   ./evaluate_final_checkpoint.sh gcl --vp2_bins 10 --suffix _test
#   ./evaluate_final_checkpoint.sh manual --ql_model_path experiment/ql/manual_alpha0.0000_bins5_final.pt
#   ./evaluate_final_checkpoint.sh gcl --reward_combine_lambda 0.3 --experiment_dir ./combined_exp

set -e

ALGORITHM="${1:-manual}"
VP2_BINS=5
SUFFIX=""
USE_LSTM=false
REWARD_COMBINE_LAMBDA=""
COMBINED_OR_TRAIN_DATA_PATH=""
EVAL_DATA_PATH="oviss_sample_upmc.csv"
EXPERIMENT_BASE_DIR=""
QL_MODEL_PATH_OVERRIDE=""

shift
while [[ $# -gt 0 ]]; do
    case $1 in
        --vp2_bins)
            VP2_BINS="$2"
            shift 2
            ;;
        --suffix)
            SUFFIX="$2"
            shift 2
            ;;
        --use_lstm)
            USE_LSTM=true
            shift
            ;;
        --reward_combine_lambda)
            REWARD_COMBINE_LAMBDA="$2"
            shift 2
            ;;
        --combined_or_train_data_path)
            COMBINED_OR_TRAIN_DATA_PATH="$2"
            shift 2
            ;;
        --eval_data_path)
            EVAL_DATA_PATH="$2"
            shift 2
            ;;
        --experiment_dir)
            EXPERIMENT_BASE_DIR="$2"
            shift 2
            ;;
        --ql_model_path)
            QL_MODEL_PATH_OVERRIDE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "$EXPERIMENT_BASE_DIR" ]; then
    EXPERIMENT_DIR="${EXPERIMENT_BASE_DIR}"
else
    EXPERIMENT_DIR="${SCRIPT_DIR}/experiment"
fi

QL_DIR="${EXPERIMENT_DIR}/ql"
RESULTS_DIR="${EXPERIMENT_DIR}/irl_results"
mkdir -p "$QL_DIR"
mkdir -p "$RESULTS_DIR"

if [ "$USE_LSTM" == "true" ]; then
    if [ -n "$REWARD_COMBINE_LAMBDA" ]; then
        LAMBDA_STR=$(echo "$REWARD_COMBINE_LAMBDA" | sed 's/0*$//' | sed 's/\.$//')
        MODEL_PREFIX="lstm_${ALGORITHM}_combined_manual_lambda${LAMBDA_STR}${SUFFIX}"
    else
        MODEL_PREFIX="lstm_${ALGORITHM}${SUFFIX}"
    fi
else
    if [ -n "$REWARD_COMBINE_LAMBDA" ]; then
        LAMBDA_STR=$(echo "$REWARD_COMBINE_LAMBDA" | sed 's/0*$//' | sed 's/\.$//')
        MODEL_PREFIX="${ALGORITHM}_combined_manual_lambda${LAMBDA_STR}${SUFFIX}"
    else
        MODEL_PREFIX="${ALGORITHM}${SUFFIX}"
    fi
fi

if [ -n "$QL_MODEL_PATH_OVERRIDE" ]; then
    QL_MODEL_PATH="$QL_MODEL_PATH_OVERRIDE"
else
    QL_MODEL_PATH="${QL_DIR}/${MODEL_PREFIX}_alpha0.0000_bins${VP2_BINS}_final.pt"
fi

if [ ! -f "$QL_MODEL_PATH" ]; then
    echo "Error: Q-learning model not found: $QL_MODEL_PATH"
    exit 1
fi

echo "=============================================="
echo "Evaluation-Only Pipeline"
echo "=============================================="
echo "Algorithm: $ALGORITHM"
echo "VP2 Bins: $VP2_BINS"
echo "Suffix: $SUFFIX"
echo "Experiment dir: $EXPERIMENT_DIR"
echo "Q-Learning model: $QL_MODEL_PATH"
if [ -n "$REWARD_COMBINE_LAMBDA" ]; then
    echo "Reward combine lambda: $REWARD_COMBINE_LAMBDA"
fi
if [ -n "$EVAL_DATA_PATH" ]; then
    echo "Dataset mode: DUAL-DATASET"
    echo "  Train data: ${COMBINED_OR_TRAIN_DATA_PATH:-default}"
    echo "  Eval data:  $EVAL_DATA_PATH"
else
    echo "Dataset mode: SINGLE-DATASET"
    echo "  Data path: ${COMBINED_OR_TRAIN_DATA_PATH:-default}"
fi
echo "=============================================="

echo ""
echo "=============================================="
echo "Step 1: WIS Evaluation"
echo "=============================================="

if [ "$USE_LSTM" == "true" ]; then
    echo "LSTM WIS evaluation not yet implemented - skipping"
    RESULTS_FILE="N/A (LSTM)"
else
    RESULTS_FILE="${RESULTS_DIR}/$(basename "${QL_MODEL_PATH%.pt}")_wis.txt"

    WIS_CMD="python ${SCRIPT_DIR}/is_block_discrete.py \
        --model_path $QL_MODEL_PATH \
        --vp2_bins $VP2_BINS"

    if [ -n "$COMBINED_OR_TRAIN_DATA_PATH" ]; then
        WIS_CMD="$WIS_CMD --combined_or_train_data_path $COMBINED_OR_TRAIN_DATA_PATH"
    fi
    if [ -n "$EVAL_DATA_PATH" ]; then
        WIS_CMD="$WIS_CMD --eval_data_path $EVAL_DATA_PATH"
    fi

    eval $WIS_CMD 2>&1 | tee "$RESULTS_FILE"
fi

echo ""
echo "=============================================="
echo "Evaluation Complete!"
echo "=============================================="
echo "QL Model: $QL_MODEL_PATH"
echo "Results: $RESULTS_FILE"
echo "=============================================="
