#!/bin/bash
# Transformer reward learning + Q-learning + WIS evaluation
# Usage: ./run_experiment_trans.sh [options]
#
# Examples:
#   ./run_experiment_trans.sh --test
#   ./run_experiment_trans.sh --trans_epochs 100 --ql_epochs 100
#   ./run_experiment_trans.sh --skip_irl --irl_model_path experiments/transformer/model_epoch_100.pt
#   ./run_experiment_trans.sh --use_lstm --ql_epochs 500
#   ./run_experiment_trans.sh --combined_or_train_data_path sample_data_oviss.csv

set -e

TRANS_EPOCHS=5
QL_EPOCHS=50
VP2_BINS=5
TEST_MODE=false
SUFFIX=""

TRANS_BATCH_SIZE=64
TRANS_D_MODEL=64
TRANS_NHEAD=4
TRANS_NUM_LAYERS=2
TRANS_D_FF=128
TRANS_DROPOUT=0.1
TRANS_D=5
TRANS_GAMMA=0.99
TRANS_LR=1e-4
SKIP_IRL=false
RESUME_IRL_MODEL_PATH=""

IRL_VP2_BINS=""

USE_LSTM=false
LSTM_SEQ_LEN=5
LSTM_BURN_IN=2
LSTM_OVERLAP=2
LSTM_HIDDEN=64
LSTM_LAYERS=2
LSTM_TAU=0.8
LSTM_GAMMA=0.95
LSTM_BATCH_SIZE=32

REWARD_COMBINE_LAMBDA=""
COMBINED_OR_TRAIN_DATA_PATH=""
EVAL_DATA_PATH="oviss_sample_upmc.csv"
EXPERIMENT_BASE_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --test)
            TEST_MODE=true
            TRANS_EPOCHS=2
            QL_EPOCHS=2
            SUFFIX="_test"
            shift
            ;;
        --trans_epochs)
            TRANS_EPOCHS="$2"
            shift 2
            ;;
        --trans_batch_size)
            TRANS_BATCH_SIZE="$2"
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
        --d_model)
            TRANS_D_MODEL="$2"
            shift 2
            ;;
        --nhead)
            TRANS_NHEAD="$2"
            shift 2
            ;;
        --num_layers)
            TRANS_NUM_LAYERS="$2"
            shift 2
            ;;
        --d_ff)
            TRANS_D_FF="$2"
            shift 2
            ;;
        --dropout)
            TRANS_DROPOUT="$2"
            shift 2
            ;;
        --trans_d)
            TRANS_D="$2"
            shift 2
            ;;
        --trans_gamma)
            TRANS_GAMMA="$2"
            shift 2
            ;;
        --trans_lr)
            TRANS_LR="$2"
            shift 2
            ;;
        --skip_irl)
            SKIP_IRL=true
            shift
            ;;
        --resume_irl_model_path)
            RESUME_IRL_MODEL_PATH="$2"
            shift 2
            ;;
        --irl_model_path)
            IRL_MODEL_PATH="$2"
            SKIP_IRL=true
            shift 2
            ;;
        --irl_vp2_bins)
            IRL_VP2_BINS="$2"
            shift 2
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
        --use_lstm)
            USE_LSTM=true
            shift
            ;;
        --lstm_seq_len)
            LSTM_SEQ_LEN="$2"
            shift 2
            ;;
        --lstm_burn_in)
            LSTM_BURN_IN="$2"
            shift 2
            ;;
        --lstm_overlap)
            LSTM_OVERLAP="$2"
            shift 2
            ;;
        --lstm_hidden)
            LSTM_HIDDEN="$2"
            shift 2
            ;;
        --lstm_layers)
            LSTM_LAYERS="$2"
            shift 2
            ;;
        --lstm_tau)
            LSTM_TAU="$2"
            shift 2
            ;;
        --lstm_gamma)
            LSTM_GAMMA="$2"
            shift 2
            ;;
        --lstm_batch_size)
            LSTM_BATCH_SIZE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "Transformer Experiment Pipeline"
echo "=============================================="
echo "IRL Epochs: $TRANS_EPOCHS"
echo "QL Epochs: $QL_EPOCHS"
echo "VP2 Bins: $VP2_BINS"
echo "Test Mode: $TEST_MODE"
echo "Skip IRL: $SKIP_IRL"
echo "Suffix: $SUFFIX"
echo "Transformer batch_size: $TRANS_BATCH_SIZE"
echo "Transformer d_model: $TRANS_D_MODEL"
echo "Transformer nhead: $TRANS_NHEAD"
echo "Transformer num_layers: $TRANS_NUM_LAYERS"
echo "Transformer d_ff: $TRANS_D_FF"
echo "Transformer dropout: $TRANS_DROPOUT"
echo "Transformer D: $TRANS_D"
echo "Transformer gamma: $TRANS_GAMMA"
echo "Transformer lr: $TRANS_LR"
if [ -n "$IRL_MODEL_PATH" ]; then
    echo "Pre-trained IRL model: $IRL_MODEL_PATH"
fi
if [ -n "$RESUME_IRL_MODEL_PATH" ]; then
    echo "Resume IRL model: $RESUME_IRL_MODEL_PATH"
fi
if [ -n "$IRL_VP2_BINS" ]; then
    echo "IRL model VP2 bins: $IRL_VP2_BINS (different from Q-learning VP2 bins: $VP2_BINS)"
fi
if [ -n "$REWARD_COMBINE_LAMBDA" ]; then
    echo "Reward combine lambda: $REWARD_COMBINE_LAMBDA"
    echo "  Combined reward = (1-lambda)*manual + lambda*IRL"
fi
if [ -n "$EVAL_DATA_PATH" ]; then
    echo "Dataset mode: DUAL-DATASET"
    echo "  Train data: ${COMBINED_OR_TRAIN_DATA_PATH:-default}"
    echo "  Eval data:  $EVAL_DATA_PATH"
else
    echo "Dataset mode: SINGLE-DATASET"
    echo "  Data path: ${COMBINED_OR_TRAIN_DATA_PATH:-default}"
fi
echo "Experiment dir: ${EXPERIMENT_BASE_DIR:-./experiment (default)}"
if [ "$USE_LSTM" == "true" ]; then
    echo "Q-Learning: LSTM (recurrent)"
    echo "  LSTM seq_len: $LSTM_SEQ_LEN"
    echo "  LSTM burn_in: $LSTM_BURN_IN"
    echo "  LSTM overlap: $LSTM_OVERLAP"
    echo "  LSTM hidden: $LSTM_HIDDEN"
    echo "  LSTM layers: $LSTM_LAYERS"
    echo "  LSTM tau: $LSTM_TAU"
    echo "  LSTM gamma: $LSTM_GAMMA"
    echo "  LSTM batch_size: $LSTM_BATCH_SIZE"
else
    echo "Q-Learning: Standard (non-recurrent)"
fi
echo "=============================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "$EXPERIMENT_BASE_DIR" ]; then
    EXPERIMENT_DIR="${EXPERIMENT_BASE_DIR}"
else
    EXPERIMENT_DIR="${SCRIPT_DIR}/experiment"
fi
IRL_DIR="${EXPERIMENT_DIR}/irl"
QL_DIR="${EXPERIMENT_DIR}/ql"
RESULTS_DIR="${EXPERIMENT_DIR}/irl_results"

mkdir -p "$IRL_DIR"
mkdir -p "$QL_DIR"
mkdir -p "$RESULTS_DIR"

if [ "$SKIP_IRL" == "true" ] && [ -n "$IRL_MODEL_PATH" ]; then
    echo ""
    echo "=============================================="
    echo "Step 1: Skipping IRL Training (using pre-trained model)"
    echo "=============================================="
    REWARD_MODEL_PATH="$IRL_MODEL_PATH"
    echo "Using pre-trained IRL model: $REWARD_MODEL_PATH"
elif [ "$SKIP_IRL" == "true" ]; then
    echo ""
    echo "=============================================="
    echo "Step 1: Skipping IRL Training (no model provided)"
    echo "=============================================="
    REWARD_MODEL_PATH=""
else
    echo ""
    echo "=============================================="
    echo "Step 1: IRL Training (transformer)"
    echo "=============================================="

    TRANS_DIR="${EXPERIMENT_DIR}/transformer${SUFFIX}"
    mkdir -p "$TRANS_DIR"
    TRANS_CMD="python ${SCRIPT_DIR}/transformer_reward_generator_tanh.py \
        --epochs $TRANS_EPOCHS \
        --batch_size $TRANS_BATCH_SIZE \
        --lr $TRANS_LR \
        --d_model $TRANS_D_MODEL \
        --nhead $TRANS_NHEAD \
        --num_layers $TRANS_NUM_LAYERS \
        --d_ff $TRANS_D_FF \
        --dropout $TRANS_DROPOUT \
        --D $TRANS_D \
        --gamma $TRANS_GAMMA \
        --vp2_bins $VP2_BINS \
        --experiment_dir $TRANS_DIR"

    if [ -n "$COMBINED_OR_TRAIN_DATA_PATH" ]; then
        TRANS_CMD="$TRANS_CMD --combined_or_train_data_path $COMBINED_OR_TRAIN_DATA_PATH"
    fi
    if [ -n "$EVAL_DATA_PATH" ]; then
        TRANS_CMD="$TRANS_CMD --eval_data_path $EVAL_DATA_PATH"
    fi
    if [ -n "$RESUME_IRL_MODEL_PATH" ]; then
        TRANS_CMD="$TRANS_CMD --resume_model_path $RESUME_IRL_MODEL_PATH"
    fi

    eval $TRANS_CMD

    REWARD_MODEL_PATH=$(ls -t "${TRANS_DIR}_${TRANS_D_MODEL}d_${TRANS_NUM_LAYERS}l_tanh"/model_epoch_*.pt 2>/dev/null | head -1)
    if [ -z "$REWARD_MODEL_PATH" ]; then
        echo "Error: No transformer model found in ${TRANS_DIR}_${TRANS_D_MODEL}d_${TRANS_NUM_LAYERS}l_tanh"
        exit 1
    fi

    echo "IRL training complete. Model saved to: $REWARD_MODEL_PATH"
fi

echo ""
echo "=============================================="
if [ "$USE_LSTM" == "true" ]; then
    echo "Step 2: LSTM Q-Learning Training (Recurrent)"
else
    echo "Step 2: Q-Learning Training (Standard)"
fi
echo "=============================================="

if [ "$USE_LSTM" == "true" ]; then
    QL_CMD="python ${SCRIPT_DIR}/run_lstm_block_discrete_cql_with_logging.py \
        --alpha 0.0 \
        --vp2_bins $VP2_BINS \
        --epochs $QL_EPOCHS \
        --sequence_length $LSTM_SEQ_LEN \
        --burn_in_length $LSTM_BURN_IN \
        --overlap $LSTM_OVERLAP \
        --hidden_dim $LSTM_HIDDEN \
        --lstm_hidden $LSTM_HIDDEN \
        --num_lstm_layers $LSTM_LAYERS \
        --tau $LSTM_TAU \
        --gamma $LSTM_GAMMA \
        --batch_size $LSTM_BATCH_SIZE \
        --save_dir $QL_DIR \
        --log_dir ${EXPERIMENT_DIR}/logs"

    if [ -n "$SUFFIX" ]; then
        QL_CMD="$QL_CMD --suffix $SUFFIX"
    fi
    if [ -n "$REWARD_MODEL_PATH" ]; then
        QL_CMD="$QL_CMD --reward_model_path $REWARD_MODEL_PATH"
    fi
    if [ -n "$IRL_VP2_BINS" ]; then
        QL_CMD="$QL_CMD --irl_vp2_bins $IRL_VP2_BINS"
    fi
    if [ -n "$REWARD_COMBINE_LAMBDA" ]; then
        QL_CMD="$QL_CMD --reward_combine_lambda $REWARD_COMBINE_LAMBDA"
    fi
    if [ -n "$COMBINED_OR_TRAIN_DATA_PATH" ]; then
        QL_CMD="$QL_CMD --combined_or_train_data_path $COMBINED_OR_TRAIN_DATA_PATH"
    fi
    if [ -n "$EVAL_DATA_PATH" ]; then
        QL_CMD="$QL_CMD --eval_data_path $EVAL_DATA_PATH"
    fi

    mkdir -p "${EXPERIMENT_DIR}/logs"
    eval $QL_CMD

    if [ -n "$REWARD_COMBINE_LAMBDA" ]; then
        LAMBDA_STR=$(echo "$REWARD_COMBINE_LAMBDA" | sed 's/0*$//' | sed 's/\.$//')
        MODEL_PREFIX="lstm_transformer_combined_manual_lambda${LAMBDA_STR}${SUFFIX}"
    else
        MODEL_PREFIX="lstm_transformer${SUFFIX}"
    fi
else
    QL_CMD="python ${SCRIPT_DIR}/run_block_discrete_cql_allalphas.py \
        --single_alpha 0.0 \
        --vp2_bins $VP2_BINS \
        --epochs $QL_EPOCHS \
        --save_dir $QL_DIR"

    if [ -n "$SUFFIX" ]; then
        QL_CMD="$QL_CMD --suffix $SUFFIX"
    fi
    if [ -n "$REWARD_MODEL_PATH" ]; then
        QL_CMD="$QL_CMD --reward_model_path $REWARD_MODEL_PATH"
    fi
    if [ -n "$IRL_VP2_BINS" ]; then
        QL_CMD="$QL_CMD --irl_vp2_bins $IRL_VP2_BINS"
    fi
    if [ -n "$REWARD_COMBINE_LAMBDA" ]; then
        QL_CMD="$QL_CMD --reward_combine_lambda $REWARD_COMBINE_LAMBDA"
    fi
    if [ -n "$COMBINED_OR_TRAIN_DATA_PATH" ]; then
        QL_CMD="$QL_CMD --combined_or_train_data_path $COMBINED_OR_TRAIN_DATA_PATH"
    fi
    if [ -n "$EVAL_DATA_PATH" ]; then
        QL_CMD="$QL_CMD --eval_data_path $EVAL_DATA_PATH"
    fi

    eval $QL_CMD

    if [ -n "$REWARD_COMBINE_LAMBDA" ]; then
        LAMBDA_STR=$(echo "$REWARD_COMBINE_LAMBDA" | sed 's/0*$//' | sed 's/\.$//')
        MODEL_PREFIX="transformer_combined_manual_lambda${LAMBDA_STR}${SUFFIX}"
    else
        MODEL_PREFIX="transformer${SUFFIX}"
    fi
fi

QL_MODEL_PATH="${QL_DIR}/${MODEL_PREFIX}_alpha0.0000_bins${VP2_BINS}_best.pt"
echo "Q-Learning complete. Model saved to: $QL_MODEL_PATH"

echo ""
echo "=============================================="
echo "Step 3: WIS Evaluation"
echo "=============================================="

if [ "$USE_LSTM" == "true" ]; then
    echo "LSTM WIS evaluation not yet implemented - skipping"
    echo "LSTM model saved to: $QL_MODEL_PATH"
    RESULTS_FILE="N/A (LSTM)"
else
    RESULTS_FILE="${RESULTS_DIR}/${MODEL_PREFIX}_alpha0.0000_bins${VP2_BINS}_best_wis.txt"

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
echo "Experiment Complete!"
echo "=============================================="
echo "IRL Model: ${REWARD_MODEL_PATH:-N/A}"
echo "QL Model: $QL_MODEL_PATH"
echo "Results: $RESULTS_FILE"
if [ "$USE_LSTM" == "true" ]; then
    echo "Q-Learning type: LSTM (recurrent)"
else
    echo "Q-Learning type: Standard"
fi
echo "=============================================="
