#!/usr/bin/env python3
"""
Comprehensive Re-Evaluation Script
====================================
This script re-runs ALL model evaluations from scratch and generates comparison tables.

Usage:
    python rerun_all_evaluations.py                    # Run all evaluations
    python rerun_all_evaluations.py --skip-existing    # Skip already evaluated models
    python rerun_all_evaluations.py --only-tables      # Only regenerate tables (current behavior)
"""

import subprocess
import sys
import os
import argparse
from pathlib import Path

# ANSI color codes
GREEN = '\033[92m'
BLUE = '\033[94m'
YELLOW = '\033[93m'
RED = '\033[91m'
BOLD = '\033[1m'
RESET = '\033[0m'

def print_header(text):
    print(f"\n{BOLD}{BLUE}{'='*80}{RESET}")
    print(f"{BOLD}{BLUE}{text:^80}{RESET}")
    print(f"{BOLD}{BLUE}{'='*80}{RESET}\n")

def print_section(text):
    print(f"\n{BOLD}{GREEN}{text}{RESET}")
    print(f"{GREEN}{'-'*80}{RESET}")

def print_error(text):
    print(f"{RED}❌ {text}{RESET}")

def print_success(text):
    print(f"{GREEN}✅ {text}{RESET}")

def print_warning(text):
    print(f"{YELLOW}⚠️  {text}{RESET}")

def run_evaluation(script_name, args, description):
    """Run a single evaluation script."""
    print(f"\n{BOLD}Running: {description}{RESET}")
    print(f"Command: ./venv/bin/python {script_name} {' '.join(args)}")
    
    cmd = ['./venv/bin/python', script_name] + args
    
    try:
        result = subprocess.run(
            cmd,
            cwd='/Users/foluwa/Downloads/_projects/illumenti/vaso_rl',
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout per evaluation
        )
        
        if result.returncode == 0:
            print_success(f"Completed: {description}")
            # Print last few lines of output
            output_lines = result.stdout.strip().split('\n')
            for line in output_lines[-5:]:
                print(f"  {line}")
            return True
        else:
            print_error(f"Failed: {description}")
            print(f"Error: {result.stderr[-500:]}")  # Last 500 chars of error
            return False
            
    except subprocess.TimeoutExpired:
        print_error(f"Timeout: {description}")
        return False
    except Exception as e:
        print_error(f"Exception: {description} - {str(e)}")
        return False

def check_output_exists(script_name, args):
    """Check if output LaTeX file already exists."""
    # Parse expected output filename from args
    latex_dir = Path('/Users/foluwa/Downloads/_projects/illumenti/vaso_rl/latex')
    
    # Extract key parameters to determine filename
    alpha = None
    reward = 'simple'
    bins = None
    max_step = None
    step_dim = None
    
    for i, arg in enumerate(args):
        if arg == '--alpha' and i+1 < len(args):
            alpha = args[i+1]
        elif arg == '--reward_type' and i+1 < len(args):
            reward = args[i+1]
        elif arg == '--bins' and i+1 < len(args):
            bins = args[i+1]
        elif arg == '--max_step' and i+1 < len(args):
            max_step = args[i+1]
        elif arg == '--step_dim' and i+1 < len(args):
            step_dim = args[i+1]
    
    # Determine expected filename pattern
    if 'dual_mixed' in script_name:
        pattern = f"alpha_{alpha}_{reward}_wis.tex"
    elif 'binary_vp1' in script_name:
        pattern = f"binary_vp1_{reward}_wis.tex"
    elif 'lstm' in script_name:
        pattern = f"lstm_block_discrete_bins_{bins}_{reward}_wis.tex"
    elif 'stepwise' in script_name:
        pattern = f"stepwise_bins_{bins}_max_step_{max_step}_alpha_{alpha}_{reward}_wis.tex"
    elif 'block_discrete' in script_name:
        pattern = f"block_discrete_bins_{bins}_{reward}_wis.tex"
    else:
        return False
    
    output_file = latex_dir / pattern
    return output_file.exists()

def main():
    parser = argparse.ArgumentParser(
        description='Re-run all model evaluations and generate comparison tables',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip evaluations that already have output files')
    parser.add_argument('--only-tables', action='store_true',
                       help='Only regenerate comparison tables (skip evaluations)')
    
    args = parser.parse_args()
    
    print_header("Comprehensive Model Re-Evaluation Pipeline")
    
    # Change to project directory
    os.chdir('/Users/foluwa/Downloads/_projects/illumenti/vaso_rl')
    
    # Define all evaluations to run
    evaluations = [
        # Dual-Mixed CQL (continuous action space) - uses --reward-type (with dash)
        ('is_dual_mixed.py', ['--alpha', '0.0', '--reward-type', 'simple'], 
         'Dual-Mixed CQL (α=0.0, Simple)'),
        ('is_dual_mixed.py', ['--alpha', '0.0', '--reward-type', 'oviss'], 
         'Dual-Mixed CQL (α=0.0, OVISS)'),
        
        # Block Discrete (bins=3)
        ('is_block_discrete_universal_v2.py', [
            '--model_path', 'experiment/block_discrete_cql_alpha0.0000_bins3_best.pt',
            '--bins', '3', '--reward_type', 'simple'], 
         'Block Discrete (bins=3, Simple)'),
        ('is_block_discrete_universal_v2.py', [
            '--model_path', 'experiment/block_discrete_cql_alpha0.0000_bins3_best.pt',
            '--bins', '3', '--reward_type', 'oviss'], 
         'Block Discrete (bins=3, OVISS)'),
        
        # Block Discrete (bins=5)
        ('is_block_discrete_universal_v2.py', [
            '--model_path', 'experiment/block_discrete_cql_alpha0.0000_bins5_best.pt',
            '--bins', '5', '--reward_type', 'simple'], 
         'Block Discrete (bins=5, Simple)'),
        ('is_block_discrete_universal_v2.py', [
            '--model_path', 'experiment/block_discrete_cql_alpha0.0000_bins5_best.pt',
            '--bins', '5', '--reward_type', 'oviss'], 
         'Block Discrete (bins=5, OVISS)'),
        
        # Block Discrete (bins=10)
        ('is_block_discrete_universal_v2.py', [
            '--model_path', 'experiment/block_discrete_cql_alpha0.0000_bins10_best.pt',
            '--bins', '10', '--reward_type', 'simple'], 
         'Block Discrete (bins=10, Simple)'),
        ('is_block_discrete_universal_v2.py', [
            '--model_path', 'experiment/block_discrete_cql_alpha0.0000_bins10_best.pt',
            '--bins', '10', '--reward_type', 'oviss'], 
         'Block Discrete (bins=10, OVISS)'),
        
        # Binary VP1
        ('is_binary_vp1_universal.py', [
            '--model_path', 'experiment/binary_cql_unified_alpha00_best.pt',
            '--reward_type', 'simple'], 
         'Binary VP1 (Simple)'),
        ('is_binary_vp1_universal.py', [
            '--model_path', 'experiment/binary_cql_unified_alpha00_best.pt',
            '--reward_type', 'oviss'], 
         'Binary VP1 (OVISS)'),
        
        # LSTM Block Discrete (bins=5)
        ('is_lstm_block_discrete.py', [
            '--model_path', 'experiment/lstm_block_discrete_alpha0.0000_bins5_best.pt',
            '--bins', '5', '--reward_type', 'simple'], 
         'LSTM Block Discrete (bins=5, Simple)'),
        ('is_lstm_block_discrete.py', [
            '--model_path', 'experiment/lstm_block_discrete_alpha0.0000_bins5_best.pt',
            '--bins', '5', '--reward_type', 'oviss'], 
         'LSTM Block Discrete (bins=5, OVISS)'),
        
        # Stepwise (bins=5, max_step=0.1, alpha=1e-6)
        ('is_stepwise.py', [
            '--model_path', 'experiment/stepwise_cql_alpha0.000001_best.pt',
            '--bins', '5', '--reward_type', 'simple'], 
         'Stepwise (bins=5, max_step=0.1, α=1e-6, Simple)'),
        ('is_stepwise.py', [
            '--model_path', 'experiment/stepwise_cql_alpha0.000001_best.pt',
            '--bins', '5', '--reward_type', 'oviss'], 
         'Stepwise (bins=5, max_step=0.1, α=1e-6, OVISS)'),
        
        # Stepwise (bins=9, max_step=0.2, alpha=0.0)
        ('is_stepwise.py', [
            '--model_path', 'experiment/stepwise_cql_alpha0.000000_maxstep0.2_best.pt',
            '--bins', '9', '--reward_type', 'simple'], 
         'Stepwise (bins=9, max_step=0.2, α=0.0, Simple)'),
        ('is_stepwise.py', [
            '--model_path', 'experiment/stepwise_cql_alpha0.000000_maxstep0.2_best.pt',
            '--bins', '9', '--reward_type', 'oviss'], 
         'Stepwise (bins=9, max_step=0.2, α=0.0, OVISS)'),
        
        # Stepwise (bins=5, max_step=0.1, alpha=1e-4)
        ('is_stepwise.py', [
            '--model_path', 'experiment/stepwise_cql_alpha0.000100_maxstep0.1_best.pt',
            '--bins', '5', '--reward_type', 'simple'], 
         'Stepwise (bins=5, max_step=0.1, α=1e-4, Simple)'),
        ('is_stepwise.py', [
            '--model_path', 'experiment/stepwise_cql_alpha0.000100_maxstep0.1_best.pt',
            '--bins', '5', '--reward_type', 'oviss'], 
         'Stepwise (bins=5, max_step=0.1, α=1e-4, OVISS)'),
    ]
    
    if not args.only_tables:
        print_section("Step 1/2: Running Model Evaluations")
        print(f"Total evaluations to run: {len(evaluations)}")
        
        success_count = 0
        failed_count = 0
        skipped_count = 0
        
        for i, (script, script_args, description) in enumerate(evaluations, 1):
            print(f"\n{BOLD}[{i}/{len(evaluations)}]{RESET} {description}")
            
            if args.skip_existing and check_output_exists(script, script_args):
                print_warning(f"Skipping (output exists): {description}")
                skipped_count += 1
                continue
            
            if run_evaluation(script, script_args, description):
                success_count += 1
            else:
                failed_count += 1
        
        print_section("Evaluation Summary")
        print(f"  {GREEN}✅ Successful: {success_count}/{len(evaluations)}{RESET}")
        if skipped_count > 0:
            print(f"  {YELLOW}⏭️  Skipped: {skipped_count}/{len(evaluations)}{RESET}")
        if failed_count > 0:
            print(f"  {RED}❌ Failed: {failed_count}/{len(evaluations)}{RESET}")
    else:
        print_warning("Skipping evaluations (--only-tables mode)")
    
    # Generate comparison tables
    print_section("Step 2/2: Generating Comparison Tables")
    
    try:
        result = subprocess.run(
            ['./venv/bin/python', 'generate_comprehensive_comparison.py'],
            cwd='/Users/foluwa/Downloads/_projects/illumenti/vaso_rl',
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            print_success("Comparison tables generated successfully!")
            print(result.stdout)
        else:
            print_error("Failed to generate comparison tables")
            print(result.stderr)
            return 1
            
    except Exception as e:
        print_error(f"Exception while generating tables: {str(e)}")
        return 1
    
    print_header("✅ Complete!")
    print(f"\n{BOLD}Next steps:{RESET}")
    print(f"  1. Check results in latex/table_comprehensive_all_models.tex")
    print(f"  2. Review individual results in latex/*_wis.tex")
    print(f"  3. Integrate tables into your paper using \\input{{latex/table_comprehensive_all_models.tex}}")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
