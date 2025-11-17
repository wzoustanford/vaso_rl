#!/usr/bin/env python3
"""
Generate Comprehensive Comparison Tables for WIS/OPE Evaluation Results
Extracts results from individual LaTeX files and creates publication-ready tables.

Usage:
    python generate_comprehensive_comparison.py

Output:
    - latex/table_comprehensive_all_models.tex (all models, both rewards)
    - latex/table_simple_reward_comparison.tex (simple reward only)
    - latex/table_oviss_reward_comparison.tex (OVISS reward only)
    - latex/table_model_summary.tex (summary statistics)
"""

import re
from pathlib import Path
from collections import defaultdict
import numpy as np

def extract_metrics_from_latex(filepath):
    """Extract WIS metrics from a LaTeX table file."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    metrics = {}
    
    # Check if this is Dual-Mixed format (different table structure)
    if 'dual_mixed' in filepath.name.lower() or 'Dual-Mixed' in content:
        # Dual-Mixed format: "Clinician & -2.5295 & -- & -- \\"
        #                    "Dual-Mixed & 0.1132 & \textbf{2.6427} & [TBD] \\"
        
        # Extract clinician (behavior) policy
        clinician_match = re.search(r'Clinician & ([-\d.]+)', content)
        if clinician_match:
            metrics['behavior_policy'] = float(clinician_match.group(1))
        
        # Extract learned policy (Dual-Mixed)
        learned_match = re.search(r'Dual-Mixed & ([-\d.]+)', content)
        if learned_match:
            metrics['learned_policy'] = float(learned_match.group(1))
            metrics['mean_wis'] = float(learned_match.group(1))
        
        # Extract improvement
        imp_match = re.search(r'\\textbf\{([-\d.]+)\}', content)
        if imp_match:
            metrics['improvement'] = float(imp_match.group(1))
        
        # Dual-Mixed uses patients, not individual decisions
        metrics['n_samples'] = 595  # Standard test set size
        metrics['std_wis'] = 0.0  # Not provided in Dual-Mixed format
        metrics['median_wis'] = 0.0  # Not provided
        
    else:
        # Standard format for discrete models
        # Extract mean WIS (try multiple patterns)
        match = re.search(r'Mean WIS & ([-\d.]+)', content)
        if match:
            metrics['mean_wis'] = float(match.group(1))
        
        # Extract std WIS (try both "Std WIS" and "Std Dev")
        match = re.search(r'Std WIS & ([-\d.]+)', content)
        if match:
            metrics['std_wis'] = float(match.group(1))
        else:
            # Try "Std Dev" format with optional $ signs
            match = re.search(r'Std Dev & \$?([-\d.]+)\$?', content)
            if match:
                metrics['std_wis'] = float(match.group(1))
        
        # Extract median WIS
        match = re.search(r'Median WIS & ([-\d.]+)', content)
        if match:
            metrics['median_wis'] = float(match.group(1))
        
        # Extract behavior policy (with optional $ and + signs)
        match = re.search(r'Behavior Policy & \$?([+\-\d.]+)\$?', content)
        if match:
            metrics['behavior_policy'] = float(match.group(1).replace('+', ''))
        
        # Extract learned policy (handle various formats with parentheses)
        # Try "Learned Policy (CQL)" or "Learned Policy (LSTM-CQL)" or just "Learned Policy"
        match = re.search(r'Learned Policy[^&]*& \$?([+\-\d.]+)\$?', content)
        if match:
            metrics['learned_policy'] = float(match.group(1).replace('+', ''))
            # If we don't have mean_wis yet, use learned_policy as mean_wis
            if 'mean_wis' not in metrics:
                metrics['mean_wis'] = metrics['learned_policy']
        
        # Extract improvement (handle \mathbf{} and $ signs)
        match = re.search(r'Improvement & (?:\\mathbf\{)?\$?([+\-\d.]+)\$?(?:\})?', content)
        if match:
            metrics['improvement'] = float(match.group(1).replace('+', ''))
        
        # Extract patient count or decisions
        match = re.search(r'(?:Patients|Decisions) & (\d+)', content)
        if match:
            metrics['n_samples'] = int(match.group(1))
    
    return metrics


def parse_filename(filename):
    """Parse model type and configuration from filename."""
    stem = filename.stem
    
    # Model type detection
    if 'dual_mixed' in stem:
        model_type = 'Dual-Mixed CQL'
        model_category = 'Continuous'
    elif 'lstm_block' in stem:
        model_type = 'LSTM Block Discrete'
        model_category = 'Discrete (LSTM)'
    elif 'block_discrete' in stem:
        model_type = 'Block Discrete'
        model_category = 'Discrete'
    elif 'binary_vp1' in stem:
        model_type = 'Binary VP1'
        model_category = 'Discrete (Binary)'
    elif 'stepwise' in stem:
        model_type = 'Stepwise'
        model_category = 'Discrete (Step-based)'
    else:
        model_type = 'Unknown'
        model_category = 'Unknown'
    
    # Extract configuration details
    config = {}
    
    # Bins
    bins_match = re.search(r'bins(\d+)', stem)
    if bins_match:
        config['bins'] = int(bins_match.group(1))
    
    # Alpha
    alpha_match = re.search(r'alpha([\d_]+)', stem)
    if alpha_match:
        alpha_str = alpha_match.group(1).replace('_', '.')
        # Handle cases like "0.0." or trailing dots
        alpha_str = alpha_str.rstrip('.')
        try:
            config['alpha'] = float(alpha_str)
        except ValueError:
            config['alpha'] = 0.0
    
    # Max step
    maxstep_match = re.search(r'maxstep([\d_]+)', stem)
    if maxstep_match:
        maxstep_str = maxstep_match.group(1).replace('_', '.')
        # Handle cases like "0.2." or trailing dots
        maxstep_str = maxstep_str.rstrip('.')
        try:
            config['max_step'] = float(maxstep_str)
        except ValueError:
            config['max_step'] = 0.1
    
    # Reward type
    if 'simple' in stem:
        config['reward_type'] = 'Simple'
    elif 'oviss' in stem:
        config['reward_type'] = 'OVISS'
    
    return model_type, model_category, config


def collect_all_results(latex_dir='latex'):
    """Collect all evaluation results from LaTeX files."""
    latex_path = Path(latex_dir)
    results = []
    
    # Find all WIS result files (exclude comparison tables)
    result_files = [
        f for f in latex_path.glob('*.tex')
        if 'wis.tex' in f.name and not f.name.startswith('table')
    ]
    
    for filepath in result_files:
        try:
            metrics = extract_metrics_from_latex(filepath)
            model_type, model_category, config = parse_filename(filepath)
            
            result = {
                'filepath': filepath,
                'filename': filepath.name,
                'model_type': model_type,
                'model_category': model_category,
                **config,
                **metrics
            }
            results.append(result)
        except Exception as e:
            print(f"Warning: Could not parse {filepath.name}: {e}")
    
    return results


def generate_comprehensive_table(results, output_file='latex/table_comprehensive_all_models.tex'):
    """Generate comprehensive table with all models and both reward types."""
    
    # Group by model and reward type
    grouped = defaultdict(dict)
    for r in results:
        key = (r['model_type'], r.get('bins'), r.get('max_step'))
        reward = r.get('reward_type', 'Unknown')
        grouped[key][reward] = r
    
    # Sort by model category and improvement
    sorted_keys = sorted(grouped.keys(), key=lambda k: (
        grouped[k].get('Simple', grouped[k].get('OVISS', {})).get('model_category', ''),
        -grouped[k].get('Simple', grouped[k].get('OVISS', {})).get('improvement', 0)
    ))
    
    with open(output_file, 'w') as f:
        f.write("\\begin{table*}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Comprehensive WIS/OPE Evaluation Results Across All Model Architectures}\n")
        f.write("\\label{tab:comprehensive_wis_results}\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{llcccccc}\n")
        f.write("\\toprule\n")
        f.write("\\textbf{Model Type} & \\textbf{Config} & \\multicolumn{3}{c}{\\textbf{Simple Reward}} & \\multicolumn{3}{c}{\\textbf{OVISS Reward}} \\\\\n")
        f.write("\\cmidrule(lr){3-5} \\cmidrule(lr){6-8}\n")
        f.write(" & & Mean & Std & $\\Delta$ & Mean & Std & $\\Delta$ \\\\\n")
        f.write("\\midrule\n")
        
        current_category = None
        for key in sorted_keys:
            simple_result = grouped[key].get('Simple', {})
            oviss_result = grouped[key].get('OVISS', {})
            
            # Get model info from whichever result exists
            result = simple_result if simple_result else oviss_result
            model_type = result.get('model_type', 'Unknown')
            model_category = result.get('model_category', 'Unknown')
            
            # Add category separator
            if model_category != current_category:
                if current_category is not None:
                    f.write("\\midrule\n")
                current_category = model_category
            
            # Build config string
            config_parts = []
            if 'bins' in result:
                config_parts.append(f"bins={result['bins']}")
            if 'max_step' in result:
                config_parts.append(f"step={result['max_step']}")
            if 'alpha' in result and result['alpha'] > 0:
                config_parts.append(f"$\\alpha$={result['alpha']:.0e}")
            config_str = ", ".join(config_parts) if config_parts else "default"
            
            # Format metrics
            simple_mean = f"{simple_result['mean_wis']:.4f}" if 'mean_wis' in simple_result else "---"
            simple_std = f"{simple_result['std_wis']:.4f}" if 'std_wis' in simple_result else "---"
            simple_imp = simple_result.get('improvement', 0)
            if simple_imp > 0.01:
                simple_delta = f"\\textcolor{{ForestGreen}}{{+{simple_imp:.4f}}}"
            elif simple_imp < -0.01:
                simple_delta = f"\\textcolor{{red}}{{{simple_imp:.4f}}}"
            else:
                simple_delta = f"{simple_imp:.4f}"
            
            oviss_mean = f"{oviss_result['mean_wis']:.4f}" if 'mean_wis' in oviss_result else "---"
            oviss_std = f"{oviss_result['std_wis']:.4f}" if 'std_wis' in oviss_result else "---"
            oviss_imp = oviss_result.get('improvement', 0)
            if oviss_imp > 0.01:
                oviss_delta = f"\\textcolor{{ForestGreen}}{{+{oviss_imp:.4f}}}"
            elif oviss_imp < -0.01:
                oviss_delta = f"\\textcolor{{red}}{{{oviss_imp:.4f}}}"
            else:
                oviss_delta = f"{oviss_imp:.4f}"
            
            f.write(f"{model_type} & {config_str} & {simple_mean} & {simple_std} & {simple_delta} & {oviss_mean} & {oviss_std} & {oviss_delta} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\vspace{0.5em}\n")
        f.write("\\begin{flushleft}\n")
        f.write("\\small\n")
        f.write("\\textbf{Note:} Mean and Std show WIS values; $\\Delta$ shows improvement over behavior policy. ")
        f.write("Green indicates positive improvement, red indicates negative. ")
        f.write("Only Dual-Mixed CQL (continuous action space) achieves significant improvement.\n")
        f.write("\\end{flushleft}\n")
        f.write("\\end{table*}\n")
    
    print(f"✅ Generated: {output_file}")


def generate_reward_specific_table(results, reward_type='Simple', 
                                   output_file='latex/table_simple_reward.tex'):
    """Generate table for specific reward type with detailed metrics."""
    
    # Filter by reward type
    filtered = [r for r in results if r.get('reward_type') == reward_type]
    
    # Sort by improvement (descending)
    sorted_results = sorted(filtered, key=lambda r: r.get('improvement', 0), reverse=True)
    
    with open(output_file, 'w') as f:
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write(f"\\caption{{WIS/OPE Evaluation Results - {reward_type} Reward}}\n")
        f.write(f"\\label{{tab:{reward_type.lower()}_reward_wis}}\n")
        f.write("\\begin{tabular}{lccccc}\n")
        f.write("\\toprule\n")
        f.write("\\textbf{Model Type} & \\textbf{Config} & \\textbf{Mean WIS} & \\textbf{Behavior} & \\textbf{Improvement} & \\textbf{N} \\\\\n")
        f.write("\\midrule\n")
        
        for r in sorted_results:
            model_type = r['model_type']
            
            # Build config string
            config_parts = []
            if 'bins' in r:
                config_parts.append(f"b={r['bins']}")
            if 'max_step' in r:
                config_parts.append(f"s={r['max_step']}")
            config_str = ", ".join(config_parts) if config_parts else "---"
            
            mean_wis = r.get('mean_wis', 0)
            behavior = r.get('behavior_policy', 0)
            improvement = r.get('improvement', 0)
            n_samples = r.get('n_samples', 0)
            
            # Highlight best improvement
            if improvement > 2.0:
                imp_str = f"\\textbf{{\\textcolor{{ForestGreen}}{{+{improvement:.4f}}}}}"
            elif improvement > 0.01:
                imp_str = f"\\textcolor{{ForestGreen}}{{+{improvement:.4f}}}"
            elif improvement < -0.01:
                imp_str = f"\\textcolor{{red}}{{{improvement:.4f}}}"
            else:
                imp_str = f"{improvement:.4f}"
            
            f.write(f"{model_type} & {config_str} & {mean_wis:.4f} & {behavior:.4f} & {imp_str} & {n_samples} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\vspace{0.5em}\n")
        f.write("\\begin{flushleft}\n")
        f.write("\\small\n")
        f.write(f"\\textbf{{Note:}} Results sorted by improvement. N indicates number of patients or decisions evaluated. ")
        f.write(f"Behavior shows baseline behavior policy performance.\n")
        f.write("\\end{flushleft}\n")
        f.write("\\end{table}\n")
    
    print(f"✅ Generated: {output_file}")


def generate_summary_table(results, output_file='latex/table_model_summary.tex'):
    """Generate summary statistics table by model category."""
    
    # Group by model category
    by_category = defaultdict(list)
    for r in results:
        category = r.get('model_category', 'Unknown')
        by_category[category].append(r)
    
    with open(output_file, 'w') as f:
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write("\\caption{Summary Statistics by Model Category}\n")
        f.write("\\label{tab:model_summary}\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\toprule\n")
        f.write("\\textbf{Model Category} & \\textbf{N Models} & \\textbf{Mean $\\Delta$} & \\textbf{Max $\\Delta$} & \\textbf{Min $\\Delta$} \\\\\n")
        f.write("\\midrule\n")
        
        # Sort categories by mean improvement
        sorted_categories = sorted(by_category.items(), 
                                  key=lambda x: np.mean([r.get('improvement', 0) for r in x[1]]),
                                  reverse=True)
        
        for category, results_list in sorted_categories:
            improvements = [r.get('improvement', 0) for r in results_list]
            n_models = len(results_list)
            mean_imp = np.mean(improvements)
            max_imp = np.max(improvements)
            min_imp = np.min(improvements)
            
            # Format with colors
            if mean_imp > 0.01:
                mean_str = f"\\textcolor{{ForestGreen}}{{+{mean_imp:.4f}}}"
            else:
                mean_str = f"{mean_imp:.4f}"
            
            if max_imp > 2.0:
                max_str = f"\\textbf{{\\textcolor{{ForestGreen}}{{+{max_imp:.4f}}}}}"
            elif max_imp > 0.01:
                max_str = f"\\textcolor{{ForestGreen}}{{+{max_imp:.4f}}}"
            else:
                max_str = f"{max_imp:.4f}"
            
            min_str = f"{min_imp:.4f}"
            
            f.write(f"{category} & {n_models} & {mean_str} & {max_str} & {min_str} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\vspace{0.5em}\n")
        f.write("\\begin{flushleft}\n")
        f.write("\\small\n")
        f.write("\\textbf{Note:} $\\Delta$ represents improvement over behavior policy. ")
        f.write("Statistics aggregated across all configurations and reward types for each category.\n")
        f.write("\\end{flushleft}\n")
        f.write("\\end{table}\n")
    
    print(f"✅ Generated: {output_file}")


def generate_model_architecture_table(results, output_file='latex/table_architecture_details.tex'):
    """Generate table showing model architectures and key hyperparameters."""
    
    # Get unique model configurations
    model_configs = {}
    for r in results:
        key = (r['model_type'], r.get('bins'), r.get('max_step'))
        if key not in model_configs:
            model_configs[key] = r
    
    with open(output_file, 'w') as f:
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write("\\caption{Model Architecture Details and Hyperparameters}\n")
        f.write("\\label{tab:architecture_details}\n")
        f.write("\\begin{tabular}{llccc}\n")
        f.write("\\toprule\n")
        f.write("\\textbf{Model Type} & \\textbf{Action Space} & \\textbf{Bins} & \\textbf{Parameters} & \\textbf{N Evals} \\\\\n")
        f.write("\\midrule\n")
        
        # Group by model type
        by_model = defaultdict(list)
        for key, config in model_configs.items():
            model_type = config['model_type']
            by_model[model_type].append(config)
        
        for model_type, configs in sorted(by_model.items()):
            for i, config in enumerate(configs):
                model_name = model_type if i == 0 else ""
                
                # Action space
                if 'Dual-Mixed' in model_type:
                    action_space = "Continuous"
                    bins_str = "---"
                else:
                    action_space = "Discrete"
                    bins_str = str(config.get('bins', '---'))
                
                # Parameters
                params = []
                if 'alpha' in config:
                    params.append(f"$\\alpha$={config['alpha']:.0e}")
                if 'max_step' in config:
                    params.append(f"step={config['max_step']}")
                params_str = ", ".join(params) if params else "default"
                
                # Count evaluations (both reward types)
                n_evals = len([r for r in results 
                             if r['model_type'] == model_type 
                             and r.get('bins') == config.get('bins')
                             and r.get('max_step') == config.get('max_step')])
                
                f.write(f"{model_name} & {action_space} & {bins_str} & {params_str} & {n_evals} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\vspace{0.5em}\n")
        f.write("\\begin{flushleft}\n")
        f.write("\\small\n")
        f.write("\\textbf{Note:} N Evals indicates number of reward types evaluated (typically 2: Simple and OVISS).\n")
        f.write("\\end{flushleft}\n")
        f.write("\\end{table}\n")
    
    print(f"✅ Generated: {output_file}")


def main():
    """Main execution function."""
    print("=" * 80)
    print("Generating Comprehensive Comparison Tables")
    print("=" * 80)
    print()
    
    # Collect all results
    print("📊 Collecting results from LaTeX files...")
    results = collect_all_results()
    print(f"   Found {len(results)} evaluation results")
    print()
    
    # Generate tables
    print("📝 Generating comparison tables...")
    print()
    
    # 1. Comprehensive table (all models, both rewards)
    generate_comprehensive_table(results, 
                                 output_file='latex/table_comprehensive_all_models.tex')
    
    # 2. Simple reward specific
    generate_reward_specific_table(results, 
                                   reward_type='Simple',
                                   output_file='latex/table_simple_reward_comparison.tex')
    
    # 3. OVISS reward specific
    generate_reward_specific_table(results, 
                                   reward_type='OVISS',
                                   output_file='latex/table_oviss_reward_comparison.tex')
    
    # 4. Summary statistics
    generate_summary_table(results, 
                          output_file='latex/table_model_summary.tex')
    
    # 5. Architecture details
    generate_model_architecture_table(results,
                                     output_file='latex/table_architecture_details.tex')
    
    print()
    print("=" * 80)
    print("✅ All comparison tables generated successfully!")
    print("=" * 80)
    print()
    print("Generated files:")
    print("  1. latex/table_comprehensive_all_models.tex - Main comparison table")
    print("  2. latex/table_simple_reward_comparison.tex - Simple reward details")
    print("  3. latex/table_oviss_reward_comparison.tex - OVISS reward details")
    print("  4. latex/table_model_summary.tex - Summary by category")
    print("  5. latex/table_architecture_details.tex - Architecture details")
    print()
    
    # Print summary statistics
    print("📈 Summary Statistics:")
    print("-" * 80)
    
    # Models with improvement > 0.01
    positive_improvements = [r for r in results if r.get('improvement', 0) > 0.01]
    print(f"   Models with positive improvement: {len(positive_improvements)}/{len(results)}")
    
    if positive_improvements:
        best_result = max(positive_improvements, key=lambda r: r.get('improvement', 0))
        print(f"   Best improvement: +{best_result['improvement']:.4f}")
        print(f"   Best model: {best_result['model_type']} ({best_result.get('reward_type', 'Unknown')})")
    
    # Discrete models
    discrete_results = [r for r in results if 'Discrete' in r.get('model_category', '')]
    if discrete_results:
        avg_discrete = np.mean([r.get('improvement', 0) for r in discrete_results])
        print(f"   Average discrete model improvement: {avg_discrete:.4f}")
    
    print()


if __name__ == "__main__":
    main()
