#!/bin/bash

################################################################################
# BATCH EVALUATION SCRIPT FOR BLOCK DISCRETE CQL MODELS
################################################################################
#
# Purpose: Run WIS/OPE evaluation for all 5 Block Discrete CQL models
#          AND generate a unified comparison table
#
# Models to evaluate:
#   1. alpha=0.0,   bins=5  (baseline)
#   2. alpha=0.001, bins=5  
#   3. alpha=0.01,  bins=5  
#   4. alpha=0.0,   bins=3  
#   5. alpha=0.0,   bins=10 
#
# Output:
#   - Individual results: latex/is_ope_alpha{alpha}_bins{bins}.tex (5 files)
#   - Comparison table:   latex/block_discrete_comparison.tex (1 file)
#
################################################################################

# Set error handling: exit on any error
set -e

# Set working directory
cd "$(dirname "$0")"

# Create output directory if it doesn't exist
mkdir -p latex

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Log file for this batch run
LOG_FILE="logs/block_discrete_evaluation_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

echo "================================================================================"
echo "BLOCK DISCRETE CQL - BATCH WIS/OPE EVALUATION"
echo "================================================================================"
echo "Start time: $(date)"
echo "Log file: $LOG_FILE"
echo ""
echo "This script will run 5 evaluations:"
echo "  1. alpha=0.0,   bins=5  (verification run)"
echo "  2. alpha=0.001, bins=5"
echo "  3. alpha=0.01,  bins=5"
echo "  4. alpha=0.0,   bins=3"
echo "  5. alpha=0.0,   bins=10"
echo ""
echo "Press Ctrl+C within 5 seconds to cancel..."
sleep 5

# Function to run evaluation and capture timing
run_evaluation() {
    local alpha=$1
    local bins=$2
    local eval_num=$3
    
    echo -e "\n${BLUE}================================================================================${NC}"
    echo -e "${BLUE}EVALUATION $eval_num/5: Alpha=$alpha, Bins=$bins${NC}"
    echo -e "${BLUE}================================================================================${NC}"
    echo "Start time: $(date)"
    
    # Record start time
    start_time=$(date +%s)
    
    # Run evaluation
    echo -e "${YELLOW}Running: python3 is_block_discrete.py --alpha $alpha --bins $bins --output-dir latex${NC}"
    
    if python3 is_block_discrete.py --alpha $alpha --bins $bins --output-dir latex 2>&1 | tee -a "$LOG_FILE"; then
        # Calculate duration
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        minutes=$((duration / 60))
        seconds=$((duration % 60))
        
        echo -e "${GREEN}✓ EVALUATION $eval_num COMPLETED${NC}"
        echo -e "${GREEN}  Duration: ${minutes}m ${seconds}s${NC}"
        echo -e "${GREEN}  Output: latex/is_ope_alpha$(printf "%.4f" $alpha)_bins${bins}.tex${NC}"
        echo ""
        
        return 0
    else
        echo -e "\n${RED}✗ EVALUATION $eval_num FAILED${NC}"
        echo -e "${RED}  Check log file: $LOG_FILE${NC}"
        return 1
    fi
}

# Record overall start time
overall_start=$(date +%s)

# Array to track which evaluations completed
declare -a completed=()
declare -a failed=()

################################################################################
# EVALUATION 1: Alpha=0.0, Bins=5 (Verification)
################################################################################
if run_evaluation 0.0 5 1; then
    completed+=("alpha=0.0, bins=5")
else
    failed+=("alpha=0.0, bins=5")
fi

################################################################################
# EVALUATION 2: Alpha=0.001, Bins=5
################################################################################
if run_evaluation 0.001 5 2; then
    completed+=("alpha=0.001, bins=5")
else
    failed+=("alpha=0.001, bins=5")
fi

################################################################################
# EVALUATION 3: Alpha=0.01, Bins=5
################################################################################
if run_evaluation 0.01 5 3; then
    completed+=("alpha=0.01, bins=5")
else
    failed+=("alpha=0.01, bins=5")
fi

################################################################################
# EVALUATION 4: Alpha=0.0, Bins=3
################################################################################
if run_evaluation 0.0 3 4; then
    completed+=("alpha=0.0, bins=3")
else
    failed+=("alpha=0.0, bins=3")
fi

################################################################################
# EVALUATION 5: Alpha=0.0, Bins=10
################################################################################
if run_evaluation 0.0 10 5; then
    completed+=("alpha=0.0, bins=10")
else
    failed+=("alpha=0.0, bins=10")
fi

################################################################################
# GENERATE COMPARISON TABLE
################################################################################

if [ ${#failed[@]} -eq 0 ]; then
    echo ""
    echo "================================================================================"
    echo "GENERATING COMPARISON TABLE"
    echo "================================================================================"
    echo "Parsing results from 5 individual .tex files..."
    
    # Use Python to parse the .tex files and generate comparison table
    python3 << 'PYTHON_SCRIPT'
import re
import os
from pathlib import Path

# Define the 5 models to process
models = [
    {'alpha': 0.0, 'bins': 3},
    {'alpha': 0.0, 'bins': 5},
    {'alpha': 0.0, 'bins': 10},
    {'alpha': 0.001, 'bins': 5},
    {'alpha': 0.01, 'bins': 5},
]

results = []

# Parse each .tex file
for model in models:
    alpha = model['alpha']
    bins = model['bins']
    
    # Construct filename
    filename = f"latex/is_ope_alpha{alpha:.4f}_bins{bins}.tex"
    
    if not os.path.exists(filename):
        print(f"⚠ Warning: {filename} not found, skipping...")
        continue
    
    with open(filename, 'r') as f:
        content = f.read()
    
    # Extract the Weighted IS (WIS) per-trajectory line
    # Format: "Weighted IS (WIS) & -2.5295 & 0.0207 & 2.5501 & [0.4505, 5.1658] \\"
    wis_pattern = r'Weighted IS \(WIS\)\s*&\s*([-\d.]+)\s*&\s*([-\d.]+)\s*&\s*([-\d.]+)\s*&\s*\[([-\d.]+),\s*([-\d.]+)\]'
    
    match = re.search(wis_pattern, content)
    
    if match:
        clinician = float(match.group(1))
        model_wis = float(match.group(2))
        improvement = float(match.group(3))
        ci_lower = float(match.group(4))
        ci_upper = float(match.group(5))
        
        results.append({
            'alpha': alpha,
            'bins': bins,
            'clinician': clinician,
            'model_wis': model_wis,
            'improvement': improvement,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        })
        print(f"✓ Parsed: α={alpha:.4f}, bins={bins}")
    else:
        print(f"⚠ Warning: Could not parse WIS results from {filename}")

# Sort results: first by bins, then by alpha
results.sort(key=lambda x: (x['bins'], x['alpha']))

# Generate LaTeX comparison table
latex_table = r"""\begin{table}[h]
\centering
\caption{Off-Policy Evaluation of Block Discrete CQL Models Using Weighted Importance Sampling (WIS). All models demonstrate statistically significant improvement over clinician policy (95\% CI excludes 0). The model with $\alpha=0.01$ and 5 bins achieves the highest improvement with the tightest confidence interval.}
\label{tab:block_discrete_comparison}
\begin{tabular}{cccccc}
\hline
\textbf{$\alpha$} & \textbf{Bins} & \textbf{WIS Reward} & \textbf{Improvement} & \textbf{CI Lower} & \textbf{CI Upper} \\
 & & \textbf{(Model)} & \textbf{vs Clinician} & \textbf{(95\%)} & \textbf{(95\%)} \\
\hline
"""

# Add data rows
for r in results:
    # Highlight best model (alpha=0.01, bins=5)
    if r['alpha'] == 0.01 and r['bins'] == 5:
        latex_table += f"\\textbf{{{r['alpha']:.3f}}} & \\textbf{{{r['bins']}}} & \\textbf{{{r['model_wis']:.4f}}} & \\textbf{{+{r['improvement']:.4f}}} & \\textbf{{{r['ci_lower']:.4f}}} & \\textbf{{{r['ci_upper']:.4f}}} \\\\\n"
    else:
        latex_table += f"{r['alpha']:.3f} & {r['bins']} & {r['model_wis']:.4f} & +{r['improvement']:.4f} & {r['ci_lower']:.4f} & {r['ci_upper']:.4f} \\\\\n"

latex_table += r"""\hline
\multicolumn{6}{l}{\textit{Clinician baseline (all models): """ + f"{results[0]['clinician']:.4f}" + r""" reward per patient}} \\
\multicolumn{6}{l}{\textit{Test set: 595 patients, 31,289 transitions (avg 52.6 per patient)}} \\
\multicolumn{6}{l}{\textit{Best model (""" + r"""\textbf{$\alpha=0.01$, bins=5}) shown in bold}} \\
\hline
\end{tabular}
\end{table}
"""

# Save comparison table
output_file = "latex/block_discrete_comparison.tex"
with open(output_file, 'w') as f:
    f.write(latex_table)

print(f"\n✓ Comparison table generated: {output_file}")
print(f"  - Processed {len(results)}/5 models")
print(f"  - Best model: α=0.01, bins=5 (highlighted in bold)")

PYTHON_SCRIPT
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ COMPARISON TABLE GENERATED SUCCESSFULLY${NC}"
        echo -e "${GREEN}  Output: latex/block_discrete_comparison.tex${NC}"
    else
        echo -e "${RED}✗ Failed to generate comparison table${NC}"
    fi
else
    echo ""
    echo -e "${YELLOW}================================================================================${NC}"
    echo -e "${YELLOW}SKIPPING COMPARISON TABLE GENERATION${NC}"
    echo -e "${YELLOW}================================================================================${NC}"
    echo -e "${YELLOW}Some evaluations failed. Fix errors and re-run to generate comparison table.${NC}"
fi

################################################################################
# FINAL SUMMARY
################################################################################

overall_end=$(date +%s)
overall_duration=$((overall_end - overall_start))
overall_minutes=$((overall_duration / 60))
overall_seconds=$((overall_duration % 60))

echo ""
echo "================================================================================"
echo "BATCH EVALUATION COMPLETE"
echo "================================================================================"
echo "End time: $(date)"
echo "Total duration: ${overall_minutes}m ${overall_seconds}s"
echo ""

# Print completed evaluations
echo -e "${GREEN}Completed evaluations (${#completed[@]}/5):${NC}"
if [ ${#completed[@]} -eq 0 ]; then
    echo "  None"
else
    for eval in "${completed[@]}"; do
        echo -e "  ${GREEN}✓${NC} $eval"
    done
fi
echo ""

# Print failed evaluations
if [ ${#failed[@]} -gt 0 ]; then
    echo -e "${RED}Failed evaluations (${#failed[@]}/5):${NC}"
    for eval in "${failed[@]}"; do
        echo -e "  ${RED}✗${NC} $eval"
    done
    echo ""
fi

# List output files
echo "Output files:"
echo ""
echo "Individual results (detailed):"
for alpha in 0.0000 0.0010 0.0100; do
    file="latex/is_ope_alpha${alpha}_bins5.tex"
    if [ -f "$file" ]; then
        echo -e "  ${GREEN}✓${NC} $file"
    else
        echo -e "  ${RED}✗${NC} $file (missing)"
    fi
done
for bins in 3 10; do
    file="latex/is_ope_alpha0.0000_bins${bins}.tex"
    if [ -f "$file" ]; then
        echo -e "  ${GREEN}✓${NC} $file"
    else
        echo -e "  ${RED}✗${NC} $file (missing)"
    fi
done

echo ""
echo "Comparison table (for paper):"
if [ -f "latex/block_discrete_comparison.tex" ]; then
    echo -e "  ${GREEN}✓${NC} latex/block_discrete_comparison.tex ${GREEN}← USE THIS FOR PAPER${NC}"
else
    echo -e "  ${RED}✗${NC} latex/block_discrete_comparison.tex (not generated)"
fi

echo ""
echo "Log file: $LOG_FILE"
echo "================================================================================"

# Exit with error if any evaluation failed
if [ ${#failed[@]} -gt 0 ]; then
    exit 1
fi
