#!/bin/bash
# Experiment pipeline for IRL reward learning + Q-Learning + WIS evaluation
# Usage: ./run_experiment.sh <algorithm> [options]
#
# Algorithms: manual, maxent, gcl, iq_learn, airl, unet, semi_supervised_unet
#
# Examples:
#   ./run_experiment.sh gcl --test
#   ./run_experiment.sh maxent --irl_epochs 50 --ql_epochs 100
#   ./run_experiment.sh iq_learn --iq_init_temp 0.01 --iq_tau 0.01 --iq_lr 1e-3 --iq_div chi
#   ./run_experiment.sh airl --airl_steps 200000 --airl_dyn_epochs 10
#   ./run_experiment.sh airl --wis_reward_type manual   # Fixed clinician baseline across methods
#   ./run_experiment.sh airl --wis_reward_type irl      # Evaluate on learned AIRL reward scale
#   ./run_experiment.sh unet --unet_epochs 100 --unet_conv_h_dim 16
#   ./run_experiment.sh unet --skip_irl --irl_model_path experiments/unet/model_epoch_100.pt
#   ./run_experiment.sh semi_supervised_unet --unet_epochs 100 --unet_conv_h_dim 64
#   ./run_experiment.sh manual  # Use manual reward, skip IRL training
#   ./run_experiment.sh manual --vp2_bins 10  # Use 10 bins for VP2 discretization
#
# LSTM Q-Learning (use --use_lstm to enable recurrent Q-learning):
#   ./run_experiment.sh manual --use_lstm --vp2_bins 10 --ql_epochs 500
#   ./run_experiment.sh gcl --use_lstm --ql_epochs 500
#   ./run_experiment.sh manual --use_lstm --lstm_seq_len 10 --lstm_burn_in 4 --lstm_hidden 128
#
# Dual-dataset mode (train on one dataset, evaluate on another):
#   ./run_experiment.sh manual --combined_or_train_data_path sample_data_oviss.csv --eval_data_path oviss_sample_upmc.csv
#   ./run_experiment.sh gcl --combined_or_train_data_path oviss_sample_upmc.csv --eval_data_path sample_data_oviss.csv
#
# Custom experiment directory (all checkpoints and results saved here):
#   ./run_experiment.sh gcl --experiment_dir ./combined_exp
#   ./run_experiment.sh manual --experiment_dir ./my_experiments/exp1 --vp2_bins 10

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

# U-Net-specific defaults
UNET_EPOCHS=500
UNET_CONV_H_DIM=64
UNET_D=5
UNET_GAMMA=0.99
UNET_LR=1e-4
SKIP_IRL=false

# IRL model vp2_bins (for loading pre-trained models with different vp2_bins)
IRL_VP2_BINS=""  # Empty means use same as VP2_BINS

# AIRL-specific defaults
AIRL_STEPS=200000
AIRL_DYN_EPOCHS=10

# LSTM Q-learning defaults
USE_LSTM=false
LSTM_SEQ_LEN=5
LSTM_BURN_IN=2
LSTM_OVERLAP=2
LSTM_HIDDEN=64
LSTM_LAYERS=2
LSTM_TAU=0.8
LSTM_GAMMA=0.95
LSTM_BATCH_SIZE=32

# Reward combination defaults
REWARD_COMBINE_LAMBDA=""  # Empty means pure IRL reward

# WIS evaluation reward defaults
WIS_REWARD_TYPE="manual"  # 'manual' or 'irl'

# Dataset configuration defaults
COMBINED_OR_TRAIN_DATA_PATH=""  # Empty means use default config.DATA_PATH
EVAL_DATA_PATH=""  # Empty means single-dataset mode (split train data into train/val/test)

# Output directory default
EXPERIMENT_BASE_DIR=""  # Empty means use default ./experiment

# Parse arguments
shift  # Remove first argument (algorithm)
while [[ $# -gt 0 ]]; do
    case $1 in
        --test)
            TEST_MODE=true
            IRL_EPOCHS=2
            QL_EPOCHS=2
            AIRL_STEPS=1000
            AIRL_DYN_EPOCHS=2
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
        --unet_epochs)
            UNET_EPOCHS="$2"
            shift 2
            ;;
        --unet_conv_h_dim)
            UNET_CONV_H_DIM="$2"
            shift 2
            ;;
        --unet_d)
            UNET_D="$2"
            shift 2
            ;;
        --unet_gamma)
            UNET_GAMMA="$2"
            shift 2
            ;;
        --unet_lr)
            UNET_LR="$2"
            shift 2
            ;;
        --skip_irl)
            SKIP_IRL=true
            shift
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
        --airl_steps)
            AIRL_STEPS="$2"
            shift 2
            ;;
        --airl_dyn_epochs)
            AIRL_DYN_EPOCHS="$2"
            shift 2
            ;;
        --reward_combine_lambda)
            REWARD_COMBINE_LAMBDA="$2"
            shift 2
            ;;
        --wis_reward_type)
            WIS_REWARD_TYPE="$2"
            if [[ "$WIS_REWARD_TYPE" != "manual" && "$WIS_REWARD_TYPE" != "irl" ]]; then
                echo "Error: --wis_reward_type must be 'manual' or 'irl'"
                exit 1
            fi
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
echo "Experiment Pipeline"
echo "=============================================="
echo "Algorithm: $ALGORITHM"
echo "IRL Epochs: $IRL_EPOCHS"
echo "QL Epochs: $QL_EPOCHS"
echo "VP2 Bins: $VP2_BINS"
echo "Test Mode: $TEST_MODE"
echo "Skip IRL: $SKIP_IRL"
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
if [ "$ALGORITHM" == "airl" ]; then
    echo "AIRL steps: $AIRL_STEPS"
    echo "AIRL dynamics epochs: $AIRL_DYN_EPOCHS"
fi
if [ "$ALGORITHM" == "unet" ] || [ "$ALGORITHM" == "semi_supervised_unet" ]; then
    echo "U-Net epochs: $UNET_EPOCHS"
    echo "U-Net conv_h_dim: $UNET_CONV_H_DIM"
    echo "U-Net D: $UNET_D"
    echo "U-Net gamma: $UNET_GAMMA"
    echo "U-Net lr: $UNET_LR"
    if [ "$ALGORITHM" == "semi_supervised_unet" ]; then
        echo "Semi-supervised: mortality diffusion ENABLED"
    fi
fi
if [ -n "$IRL_MODEL_PATH" ]; then
    echo "Pre-trained IRL model: $IRL_MODEL_PATH"
fi
if [ -n "$IRL_VP2_BINS" ]; then
    echo "IRL model VP2 bins: $IRL_VP2_BINS (different from Q-learning VP2 bins: $VP2_BINS)"
fi
if [ -n "$REWARD_COMBINE_LAMBDA" ]; then
    echo "Reward combine lambda: $REWARD_COMBINE_LAMBDA"
    echo "  Combined reward = (1-lambda)*manual + lambda*IRL"
fi
echo "WIS reward type: $WIS_REWARD_TYPE"
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

# Define paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "$EXPERIMENT_BASE_DIR" ]; then
    EXPERIMENT_DIR="${EXPERIMENT_BASE_DIR}"
else
    EXPERIMENT_DIR="${SCRIPT_DIR}/experiment"
fi
IRL_DIR="${EXPERIMENT_DIR}/irl"
QL_DIR="${EXPERIMENT_DIR}/ql"
RESULTS_DIR="${EXPERIMENT_DIR}/irl_results"

# Create directories
mkdir -p "$IRL_DIR"
mkdir -p "$QL_DIR"
mkdir -p "$RESULTS_DIR"

# Step 1: IRL Training (skip for manual or if --skip_irl is set)
if [ "$ALGORITHM" == "manual" ]; then
    echo ""
    echo "=============================================="
    echo "Step 1: Skipping IRL Training (using manual reward)"
    echo "=============================================="
    REWARD_MODEL_PATH=""
elif [ "$SKIP_IRL" == "true" ] && [ -n "$IRL_MODEL_PATH" ]; then
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
    echo "Step 1: IRL Training ($ALGORITHM)"
    echo "=============================================="

    case $ALGORITHM in
        maxent)
            MAXENT_CMD="python ${SCRIPT_DIR}/maxent_irl_full_recovery.py \
                --epochs $IRL_EPOCHS \
                --batch_size 128 \
                --prefix maxent${SUFFIX} \
                --save_dir $IRL_DIR"
            if [ -n "$COMBINED_OR_TRAIN_DATA_PATH" ]; then
                MAXENT_CMD="$MAXENT_CMD --combined_or_train_data_path $COMBINED_OR_TRAIN_DATA_PATH"
            fi
            if [ -n "$EVAL_DATA_PATH" ]; then
                MAXENT_CMD="$MAXENT_CMD --eval_data_path $EVAL_DATA_PATH"
            fi
            eval $MAXENT_CMD
            REWARD_MODEL_PATH="${IRL_DIR}/maxent${SUFFIX}_reward_model.pt"
            ;;
        gcl)
            GCL_CMD="python ${SCRIPT_DIR}/run_gcl_learn_block_discrete.py \
                --epochs $IRL_EPOCHS \
                --tau $GCL_TAU \
                --lr $GCL_LR \
                --cost_lr $GCL_COST_LR \
                --vp2_bins $VP2_BINS \
                --prefix gcl${SUFFIX} \
                --save_dir $IRL_DIR"
            if [ -n "$COMBINED_OR_TRAIN_DATA_PATH" ]; then
                GCL_CMD="$GCL_CMD --combined_or_train_data_path $COMBINED_OR_TRAIN_DATA_PATH"
            fi
            if [ -n "$EVAL_DATA_PATH" ]; then
                GCL_CMD="$GCL_CMD --eval_data_path $EVAL_DATA_PATH"
            fi
            eval $GCL_CMD
            REWARD_MODEL_PATH="${IRL_DIR}/gcl${SUFFIX}_cost_model.pt"
            ;;
        iq_learn)
            IQ_CMD="python ${SCRIPT_DIR}/run_iq_learn_block_discrete.py \
                --epochs $IRL_EPOCHS \
                --init_temp $IQ_INIT_TEMP \
                --tau $IQ_TAU \
                --lr $IQ_LR \
                --div $IQ_DIV \
                --vp2_bins $VP2_BINS \
                --prefix iq_learn${SUFFIX} \
                --save_dir $IRL_DIR"
            if [ -n "$COMBINED_OR_TRAIN_DATA_PATH" ]; then
                IQ_CMD="$IQ_CMD --combined_or_train_data_path $COMBINED_OR_TRAIN_DATA_PATH"
            fi
            if [ -n "$EVAL_DATA_PATH" ]; then
                IQ_CMD="$IQ_CMD --eval_data_path $EVAL_DATA_PATH"
            fi
            eval $IQ_CMD
            REWARD_MODEL_PATH="${IRL_DIR}/iq_learn${SUFFIX}_q_model.pt"
            ;;
        airl)
            AIRL_CMD="python ${SCRIPT_DIR}/airl.py \
                --airl_steps $AIRL_STEPS \
                --dyn_epochs $AIRL_DYN_EPOCHS \
                --save_dir $IRL_DIR \
                --prefix airl${SUFFIX} \
                --action_schema pipeline_dual \
                --state_schema pipeline_dual"
            if [ -n "$COMBINED_OR_TRAIN_DATA_PATH" ]; then
                AIRL_CMD="$AIRL_CMD --csv_path $COMBINED_OR_TRAIN_DATA_PATH"
            fi
            eval $AIRL_CMD
            REWARD_MODEL_PATH="${IRL_DIR}/airl${SUFFIX}_reward_model.pt"
            ;;
        unet)
            UNET_DIR="${EXPERIMENT_DIR}/unet${SUFFIX}"
            mkdir -p "$UNET_DIR"
            UNET_CMD="python ${SCRIPT_DIR}/unet_reward_generator_tanh.py \
                --epochs $UNET_EPOCHS \
                --conv_h_dim $UNET_CONV_H_DIM \
                --D $UNET_D \
                --gamma $UNET_GAMMA \
                --lr $UNET_LR \
                --vp2_bins $VP2_BINS \
                --experiment_dir $UNET_DIR"
            if [ -n "$COMBINED_OR_TRAIN_DATA_PATH" ]; then
                UNET_CMD="$UNET_CMD --combined_or_train_data_path $COMBINED_OR_TRAIN_DATA_PATH"
            fi
            if [ -n "$EVAL_DATA_PATH" ]; then
                UNET_CMD="$UNET_CMD --eval_data_path $EVAL_DATA_PATH"
            fi
            eval $UNET_CMD
            # Find the latest model (tanh version adds _tanh suffix)
            REWARD_MODEL_PATH=$(ls -t "${UNET_DIR}_${UNET_CONV_H_DIM}_tanh"/model_epoch_*.pt 2>/dev/null | head -1)
            if [ -z "$REWARD_MODEL_PATH" ]; then
                echo "Error: No U-Net model found in ${UNET_DIR}_${UNET_CONV_H_DIM}_tanh"
                exit 1
            fi
            ;;
        semi_supervised_unet)
            UNET_DIR="${EXPERIMENT_DIR}/semi_supervised_unet${SUFFIX}"
            mkdir -p "$UNET_DIR"
            SS_UNET_CMD="python ${SCRIPT_DIR}/semi_supervised_unet_reward_generator.py \
                --epochs $UNET_EPOCHS \
                --conv_h_dim $UNET_CONV_H_DIM \
                --D $UNET_D \
                --gamma $UNET_GAMMA \
                --lr $UNET_LR \
                --vp2_bins $VP2_BINS \
                --use_mortality_diffusion \
                --experiment_dir $UNET_DIR"
            if [ -n "$COMBINED_OR_TRAIN_DATA_PATH" ]; then
                SS_UNET_CMD="$SS_UNET_CMD --combined_or_train_data_path $COMBINED_OR_TRAIN_DATA_PATH"
            fi
            if [ -n "$EVAL_DATA_PATH" ]; then
                SS_UNET_CMD="$SS_UNET_CMD --eval_data_path $EVAL_DATA_PATH"
            fi
            eval $SS_UNET_CMD
            # Find the latest model
            REWARD_MODEL_PATH=$(ls -t "${UNET_DIR}"_"${UNET_CONV_H_DIM}"/model_epoch_*.pt 2>/dev/null | head -1)
            if [ -z "$REWARD_MODEL_PATH" ]; then
                echo "Error: No semi-supervised U-Net model found in ${UNET_DIR}_${UNET_CONV_H_DIM}"
                exit 1
            fi
            ;;
        *)
            echo "Unknown algorithm: $ALGORITHM"
            exit 1
            ;;
    esac

    echo "IRL training complete. Model saved to: $REWARD_MODEL_PATH"
fi

# Step 2: Q-Learning
echo ""
echo "=============================================="
if [ "$USE_LSTM" == "true" ]; then
    echo "Step 2: LSTM Q-Learning Training (Recurrent)"
else
    echo "Step 2: Q-Learning Training (Standard)"
fi
echo "=============================================="

if [ "$USE_LSTM" == "true" ]; then
    # Build LSTM Q-Learning command
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

    # Create logs directory
    mkdir -p "${EXPERIMENT_DIR}/logs"

    eval $QL_CMD

    # Determine the model prefix for LSTM (includes lstm_ prefix)
    if [ -n "$REWARD_COMBINE_LAMBDA" ]; then
        LAMBDA_STR=$(echo "$REWARD_COMBINE_LAMBDA" | sed 's/0*$//' | sed 's/\.$//')
        MODEL_PREFIX="lstm_${ALGORITHM}_combined_manual_lambda${LAMBDA_STR}${SUFFIX}"
    else
        MODEL_PREFIX="lstm_${ALGORITHM}${SUFFIX}"
    fi
else
    # Build standard Q-Learning command
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

    # Determine the model prefix for standard Q-learning
    if [ -n "$REWARD_COMBINE_LAMBDA" ]; then
        LAMBDA_STR=$(echo "$REWARD_COMBINE_LAMBDA" | sed 's/0*$//' | sed 's/\.$//')
        MODEL_PREFIX="${ALGORITHM}_combined_manual_lambda${LAMBDA_STR}${SUFFIX}"
    else
        MODEL_PREFIX="${ALGORITHM}${SUFFIX}"
    fi
fi

# Find the saved Q-learning model (alpha format is 0.0000)
QL_MODEL_PATH="${QL_DIR}/${MODEL_PREFIX}_alpha0.0000_bins${VP2_BINS}_best.pt"
echo "Q-Learning complete. Model saved to: $QL_MODEL_PATH"

# Step 3: WIS Evaluation
echo ""
echo "=============================================="
echo "Step 3: WIS Evaluation"
echo "=============================================="

if [ "$USE_LSTM" == "true" ]; then
    echo "LSTM WIS evaluation not yet implemented - skipping"
    echo "LSTM model saved to: $QL_MODEL_PATH"
    RESULTS_FILE="N/A (LSTM)"
else
    # Results file named after the model (alpha format is 0.0000)
    RESULTS_FILE="${RESULTS_DIR}/${MODEL_PREFIX}_alpha0.0000_bins${VP2_BINS}_best_wis.txt"

    # Build WIS evaluation command with optional dataset paths
    WIS_CMD="python ${SCRIPT_DIR}/is_block_discrete.py \
        --model_path $QL_MODEL_PATH \
        --vp2_bins $VP2_BINS"

    if [ "$WIS_REWARD_TYPE" == "irl" ]; then
        if [ -z "$REWARD_MODEL_PATH" ]; then
            echo "Error: --wis_reward_type irl requires an IRL reward model path"
            exit 1
        fi
        WIS_CMD="$WIS_CMD --reward_type irl --irl_model_path $REWARD_MODEL_PATH"
    else
        WIS_CMD="$WIS_CMD --reward_type manual"
    fi

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
echo "Algorithm: $ALGORITHM"
echo "IRL Model: ${REWARD_MODEL_PATH:-N/A (manual)}"
echo "QL Model: $QL_MODEL_PATH"
echo "Results: $RESULTS_FILE"
if [ "$USE_LSTM" == "true" ]; then
    echo "Q-Learning type: LSTM (recurrent)"
else
    echo "Q-Learning type: Standard"
fi
echo "=============================================="
