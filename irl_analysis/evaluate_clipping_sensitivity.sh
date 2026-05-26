#!/bin/bash
# Evaluate WIS sensitivity to importance-weight truncation thresholds.
# Usage: ./evaluate_clipping_sensitivity.sh <algorithm> [options]
#
# Examples:
#   ./evaluate_clipping_sensitivity.sh gcl
#   ./evaluate_clipping_sensitivity.sh maxent --vp2_bins 5 --suffix _test
#   ./evaluate_clipping_sensitivity.sh transformer --irl_model_path experiment/transformer_64d_2l_tanh/model_epoch_5.pt
#   ./evaluate_clipping_sensitivity.sh gcl --ql_model_path experiment/ql/gcl_alpha0.0000_bins5_final.pt
#   ./evaluate_clipping_sensitivity.sh gcl --thresholds "0.1:99.9,0.5:99.5,1:99,5:95"

set -e

ALGORITHM="${1:-manual}"
VP2_BINS=5
SUFFIX=""
REWARD_COMBINE_LAMBDA=""
COMBINED_OR_TRAIN_DATA_PATH=""
EVAL_DATA_PATH="oviss_sample_upmc.csv"
EXPERIMENT_BASE_DIR=""
QL_MODEL_PATH_OVERRIDE=""
IRL_MODEL_PATH=""
IRL_VP2_BINS=""
EVAL_SET="test"
THRESHOLDS="0.1:99.9,1:99,2.5:97.5,5:95"
INCLUDE_NO_CLIPPING=true

if [[ $# -gt 0 ]]; then
    shift
fi
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
        --irl_model_path)
            IRL_MODEL_PATH="$2"
            shift 2
            ;;
        --irl_vp2_bins)
            IRL_VP2_BINS="$2"
            shift 2
            ;;
        --eval_set)
            EVAL_SET="$2"
            shift 2
            ;;
        --thresholds)
            THRESHOLDS="$2"
            shift 2
            ;;
        --no_baseline)
            INCLUDE_NO_CLIPPING=false
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "$EXPERIMENT_BASE_DIR" ]; then
    EXPERIMENT_DIR="$EXPERIMENT_BASE_DIR"
else
    EXPERIMENT_DIR="${SCRIPT_DIR}/experiment"
fi

QL_DIR="${EXPERIMENT_DIR}/ql"
RESULTS_DIR="${EXPERIMENT_DIR}/irl_results"

if [ -n "$REWARD_COMBINE_LAMBDA" ]; then
    LAMBDA_STR=$(echo "$REWARD_COMBINE_LAMBDA" | sed 's/0*$//' | sed 's/\.$//')
    MODEL_PREFIX="${ALGORITHM}_combined_manual_lambda${LAMBDA_STR}${SUFFIX}"
else
    MODEL_PREFIX="${ALGORITHM}${SUFFIX}"
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

MODEL_BASENAME="$(basename "${QL_MODEL_PATH%.pt}")"
SENSITIVITY_DIR="${RESULTS_DIR}/clipping_sensitivity/${MODEL_BASENAME}"
SUMMARY_FILE="${SENSITIVITY_DIR}/summary.tsv"
mkdir -p "$SENSITIVITY_DIR"

echo "threshold_label	lower_pct	upper_pct	clipping_bounds	standard_is	wis_transition	wis_trajectory	trajectory_diff	ci_lower	ci_upper	log_file" > "$SUMMARY_FILE"

echo "=============================================="
echo "WIS Clipping Sensitivity Evaluation"
echo "=============================================="
echo "Algorithm: $ALGORITHM"
echo "Q-Learning model: $QL_MODEL_PATH"
echo "VP2 Bins: $VP2_BINS"
echo "Eval set: $EVAL_SET"
echo "Thresholds: $THRESHOLDS"
echo "Include no-clipping baseline: $INCLUDE_NO_CLIPPING"
echo "Output dir: $SENSITIVITY_DIR"
echo "Summary: $SUMMARY_FILE"
if [ -n "$IRL_MODEL_PATH" ]; then
    echo "IRL reward model: $IRL_MODEL_PATH"
fi
echo "=============================================="

run_one_setting() {
    local label="$1"
    local lower_pct="$2"
    local upper_pct="$3"
    local disable_clipping="$4"
    local log_file="${SENSITIVITY_DIR}/${MODEL_BASENAME}_${label}_wis.txt"

    echo ""
    echo "=============================================="
    echo "Running clipping setting: $label"
    echo "Log: $log_file"
    echo "=============================================="

    local cmd=(
        python "${SCRIPT_DIR}/is_block_discrete.py"
        --model_path "$QL_MODEL_PATH"
        --vp2_bins "$VP2_BINS"
        --eval_set "$EVAL_SET"
    )

    if [ "$disable_clipping" == "true" ]; then
        cmd+=(--disable_is_clipping)
    else
        cmd+=(--is_clip_lower_pct "$lower_pct" --is_clip_upper_pct "$upper_pct")
    fi
    if [ -n "$IRL_MODEL_PATH" ]; then
        cmd+=(--reward_type irl --irl_model_path "$IRL_MODEL_PATH")
    fi
    if [ -n "$IRL_VP2_BINS" ]; then
        cmd+=(--irl_vp2_bins "$IRL_VP2_BINS")
    fi
    if [ -n "$REWARD_COMBINE_LAMBDA" ]; then
        cmd+=(--reward_combine_lambda "$REWARD_COMBINE_LAMBDA")
    fi
    if [ -n "$COMBINED_OR_TRAIN_DATA_PATH" ]; then
        cmd+=(--combined_or_train_data_path "$COMBINED_OR_TRAIN_DATA_PATH")
    fi
    if [ -n "$EVAL_DATA_PATH" ]; then
        cmd+=(--eval_data_path "$EVAL_DATA_PATH")
    fi

    "${cmd[@]}" 2>&1 | tee "$log_file"

    local clipping_bounds
    local standard_is
    local wis_transition
    local wis_trajectory
    local trajectory_diff
    local ci_lower
    local ci_upper

    clipping_bounds=$(awk -F': ' '/IS clipping bounds/ {print $2; exit}' "$log_file")
    if [ -z "$clipping_bounds" ]; then
        clipping_bounds="disabled"
    fi
    standard_is=$(awk -F': *' '/Model policy \(standard IS\)/ {print $2; exit}' "$log_file")
    wis_transition=$(awk -F': *' '/Model policy \(weighted IS\)/ {print $2; exit}' "$log_file")
    wis_trajectory=$(awk -F': *' '/Model policy \(WIS\)/ {print $2; exit}' "$log_file")
    trajectory_diff=$(awk -F': ' '/Difference \(Model - Clinician\)/ {split($2, a, " "); print a[1]; exit}' "$log_file")
    ci_lower=$(awk '/Difference \(Model - Clinician\)/ {sub(/^.*95% CI: \[/, ""); sub(/,.*$/, ""); print; exit}' "$log_file")
    ci_upper=$(awk '/Difference \(Model - Clinician\)/ {sub(/^.*, /, ""); sub(/\].*$/, ""); print; exit}' "$log_file")

    echo "${label}	${lower_pct}	${upper_pct}	${clipping_bounds}	${standard_is}	${wis_transition}	${wis_trajectory}	${trajectory_diff}	${ci_lower}	${ci_upper}	${log_file}" >> "$SUMMARY_FILE"
}

if [ "$INCLUDE_NO_CLIPPING" == "true" ]; then
    run_one_setting "no_clipping" "NA" "NA" "true"
fi

IFS=',' read -ra THRESHOLD_ITEMS <<< "$THRESHOLDS"
for item in "${THRESHOLD_ITEMS[@]}"; do
    lower_pct="${item%%:*}"
    upper_pct="${item##*:}"
    if [ "$lower_pct" == "$upper_pct" ]; then
        echo "Error: invalid threshold '$item'. Expected lower:upper, e.g. 0.5:99.5"
        exit 1
    fi

    label="clip_${lower_pct}_${upper_pct}"
    label="${label//./p}"
    run_one_setting "$label" "$lower_pct" "$upper_pct" "false"
done

echo ""
echo "=============================================="
echo "Clipping sensitivity evaluation complete"
echo "=============================================="
echo "Summary: $SUMMARY_FILE"
echo "Logs: $SENSITIVITY_DIR"
echo "=============================================="
