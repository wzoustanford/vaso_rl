#!/bin/bash
# Quick Reference: Re-run Complete WIS/OPE Analysis
# This script regenerates all comparison tables from existing evaluation results

echo "======================================"
echo "WIS/OPE Analysis - Quick Re-run"
echo "======================================"
echo ""

# Check we're in the right directory
if [ ! -f "generate_comprehensive_comparison.py" ]; then
    echo "❌ Error: Must be run from vaso_rl directory"
    exit 1
fi

# Activate virtual environment if needed
if [ ! -d "venv" ]; then
    echo "❌ Error: Virtual environment not found"
    exit 1
fi

# Run the comparison generator
echo "📊 Generating comprehensive comparison tables..."
echo ""
#!/bin/bash
# Quick script to re-run all evaluations and regenerate comparison tables
# 
# Usage:
#   ./rerun_analysis.sh              # Run all evaluations + tables
#   ./rerun_analysis.sh --only-tables # Only regenerate tables (fast)
#   ./rerun_analysis.sh --skip-existing # Skip already completed evaluations

echo "Re-running all evaluations and generating tables..."
./venv/bin/python rerun_all_evaluations.py "$@"

echo ""
echo "======================================"
echo "✅ Analysis Complete!"
echo "======================================"
echo ""
echo "Generated tables (in latex/ directory):"
echo "  1. table_comprehensive_all_models.tex"
echo "  2. table_simple_reward_comparison.tex"
echo "  3. table_oviss_reward_comparison.tex"
echo "  4. table_model_summary.tex"
echo "  5. table_architecture_details.tex"
echo ""
echo "To view results:"
echo "  cat latex/table_comprehensive_all_models.tex"
echo ""
echo "For detailed documentation:"
echo "  cat EVALUATION_RESULTS_README.md"
echo ""
