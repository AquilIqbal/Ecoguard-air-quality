"""
Model Comparison Script
=======================

Compares two approaches:
1. 15 Major Cities (with API fallback)
2. Global Model (all 57 cities)

Runs both training scripts and generates comparison report.

Author: EcoGuard Team
Date: February 2026
"""

import subprocess
import pandas as pd
import os
import time


def run_training_script(script_name, description):
    """
    Run a training script and capture output
    
    Args:
        script_name: Name of Python script to run
        description: Description for logging
        
    Returns:
        Success status, execution time
    """
    print("\n" + "=" * 70)
    print(f"🚀 RUNNING: {description}")
    print("=" * 70)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            ['python', script_name],
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        elapsed = time.time() - start_time
        
        # Print output
        print(result.stdout)
        
        if result.returncode == 0:
            print(f"\n✅ {description} completed in {elapsed/60:.1f} minutes")
            return True, elapsed
        else:
            print(f"\n❌ {description} failed!")
            print("Error:", result.stderr)
            return False, elapsed
            
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print(f"\n⏱️ {description} timed out after {elapsed/60:.1f} minutes")
        return False, elapsed
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n❌ Error running {description}: {e}")
        return False, elapsed


def load_comparison_results():
    """
    Load results from both approaches
    
    Returns:
        Dictionary with comparison data
    """
    results = {}
    
    # Load 15-city results
    if os.path.exists('models/model_comparison.csv'):
        df_15 = pd.read_csv('models/model_comparison.csv')
        results['15_cities'] = df_15
        print("✅ Loaded 15-city results")
    else:
        print("⚠️  15-city results not found")
        results['15_cities'] = None
    
    # Load global model results
    if os.path.exists('models/model_comparison_global.csv'):
        df_global = pd.read_csv('models/model_comparison_global.csv')
        results['global'] = df_global
        print("✅ Loaded global model results")
    else:
        print("⚠️  Global model results not found")
        results['global'] = None
    
    return results


def create_comparison_report(results, time_15, time_global):
    """
    Create comprehensive comparison report
    
    Args:
        results: Dictionary with model results
        time_15: Training time for 15-city approach
        time_global: Training time for global approach
    """
    print("\n" + "=" * 70)
    print("📊 COMPREHENSIVE COMPARISON REPORT")
    print("=" * 70)
    
    # Performance comparison
    print("\n1️⃣  PERFORMANCE METRICS")
    print("-" * 70)
    
    if results['15_cities'] is not None and results['global'] is not None:
        print("\n15-City Approach (Major Metros):")
        print(results['15_cities'].to_string(index=False))
        
        print("\n\nGlobal Model (All 57 Cities):")
        print(results['global'].to_string(index=False))
        
        # Get best from each
        best_15 = results['15_cities'].iloc[0]
        best_global = results['global'].iloc[0]
        
        print("\n" + "=" * 70)
        print("🏆 BEST MODEL COMPARISON")
        print("=" * 70)
        
        comparison = pd.DataFrame({
            'Approach': ['15 Cities', 'Global (57)'],
            'Model': [best_15['Model'], best_global['Model']],
            'MAE': [best_15['MAE'], best_global['MAE']],
            'RMSE': [best_15['RMSE'], best_global['RMSE']],
            'R²': [best_15['R²'], best_global['R²']]
        })
        
        print("\n" + comparison.to_string(index=False))
        
        # Determine winner
        if best_15['R²'] > best_global['R²']:
            winner = "15 Cities Approach"
            diff = best_15['R²'] - best_global['R²']
        else:
            winner = "Global Model"
            diff = best_global['R²'] - best_15['R²']
        
        print(f"\n🎯 WINNER: {winner}")
        print(f"   R² advantage: +{diff:.4f} ({diff*100:.2f}%)")
    
    # Training time comparison
    print("\n\n2️⃣  TRAINING TIME")
    print("-" * 70)
    print(f"15 Cities:  {time_15/60:.1f} minutes")
    print(f"Global:     {time_global/60:.1f} minutes")
    print(f"Difference: {abs(time_global-time_15)/60:.1f} minutes")
    
    # Coverage comparison
    print("\n\n3️⃣  COVERAGE")
    print("-" * 70)
    print("15 Cities Approach:")
    print("  ✅ ML Predictions: 15 major cities")
    print("  ✅ API Fallback: 42 other cities")
    print("  📊 Total Coverage: 57 cities")
    print("  👥 User Coverage: ~85% get ML predictions")
    
    print("\nGlobal Model:")
    print("  ✅ ML Predictions: ALL 57 cities")
    print("  ❌ API Fallback: Not needed")
    print("  📊 Total Coverage: 57 cities")
    print("  👥 User Coverage: 100% get ML predictions")
    
    # Complexity comparison
    print("\n\n4️⃣  IMPLEMENTATION COMPLEXITY")
    print("-" * 70)
    print("15 Cities Approach:")
    print("  Features: ~74")
    print("  Code: Simple if-else logic")
    print("  Maintenance: Easy")
    print("  Deployment: Requires API integration")
    
    print("\nGlobal Model:")
    print("  Features: ~131 (+57 city encodings)")
    print("  Code: One-hot encoding logic")
    print("  Maintenance: Easy")
    print("  Deployment: Standalone (no API for predictions)")
    
    # Recommendation
    print("\n\n5️⃣  RECOMMENDATION")
    print("-" * 70)
    
    if results['15_cities'] is not None and results['global'] is not None:
        if best_global['R²'] >= 0.80 and best_global['R²'] >= best_15['R²'] - 0.03:
            print("✅ RECOMMENDED: Global Model")
            print("\nReasons:")
            print("  • Works for ALL 57 cities")
            print("  • No API fallback needed")
            print("  • Simpler deployment")
            print("  • Good accuracy maintained")
            print("  • Better for dissertation (comprehensive coverage)")
        else:
            print("✅ RECOMMENDED: 15 Cities Approach")
            print("\nReasons:")
            print("  • Higher accuracy for major cities")
            print("  • Focused ML effort")
            print("  • Strategic resource allocation")
            print("  • Professional hybrid approach")
    
    # Save report
    with open('models/COMPARISON_REPORT.txt', 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("ECOGUARD - MODEL APPROACH COMPARISON\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Training Time:\n")
        f.write(f"  15 Cities: {time_15/60:.1f} min\n")
        f.write(f"  Global: {time_global/60:.1f} min\n\n")
        
        if results['15_cities'] is not None:
            f.write("15-City Results:\n")
            f.write(results['15_cities'].to_string(index=False) + "\n\n")
        
        if results['global'] is not None:
            f.write("Global Model Results:\n")
            f.write(results['global'].to_string(index=False) + "\n\n")
    
    print("\n💾 Saved: models/COMPARISON_REPORT.txt")


def main():
    """
    Main comparison function
    """
    print("\n" + "⚔️  " + "=" * 66 + " ⚔️ ")
    print("   ECOGUARD - MODEL APPROACH COMPARISON")
    print("⚔️  " + "=" * 66 + " ⚔️ \n")
    
    print("This script will:")
    print("  1. Train models with 15-city approach")
    print("  2. Train global model (all 57 cities)")
    print("  3. Compare results")
    print("  4. Recommend best approach")
    print("\n⏱️  Estimated time: 40-60 minutes total\n")
    
    input("Press Enter to start comparison...")
    
    # Run 15-city training
    success_15, time_15 = run_training_script(
        'train_all_models.py',
        '15 Major Cities Approach'
    )
    
    if not success_15:
        print("\n❌ 15-city training failed. Fix errors and try again.")
        return
    
    # Run global training
    success_global, time_global = run_training_script(
        'train_global_model.py',
        'Global Model (All 57 Cities)'
    )
    
    if not success_global:
        print("\n❌ Global training failed. Fix errors and try again.")
        return
    
    # Load and compare results
    print("\n" + "=" * 70)
    print("📊 LOADING RESULTS FOR COMPARISON")
    print("=" * 70)
    
    results = load_comparison_results()
    
    # Create report
    create_comparison_report(results, time_15, time_global)
    
    print("\n" + "⚔️  " + "=" * 66 + " ⚔️ ")
    print("   COMPARISON COMPLETE! 🎉")
    print("⚔️  " + "=" * 66 + " ⚔️ \n")
    
    print("📁 Files created:")
    print("   - models/model_comparison.csv (15 cities)")
    print("   - models/model_comparison_global.csv (global)")
    print("   - models/COMPARISON_REPORT.txt (summary)")
    
    print("\n💡 Review the comparison report to decide which approach to use!")


if __name__ == "__main__":
    main()