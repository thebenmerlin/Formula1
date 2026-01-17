"""
Model Evaluation and Analysis

Provides detailed evaluation, sensitivity analysis, and predictions
for trained F1 design models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import sys
import joblib
import argparse
from typing import Dict, List, Any, Optional

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.config import (
    MODELS_DIR, OUTPUTS_DIR, TARGET_NAMES,
    DESIGN_RANGES, REFERENCE, ML_CONFIG
)
from src.models.features import FeaturePipeline, prepare_data


def load_model(model_name: str, models_dir: str = MODELS_DIR) -> Any:
    """Load a trained model from disk."""
    model_path = os.path.join(models_dir, f'{model_name}_model.joblib')
    return joblib.load(model_path)


def load_pipeline(models_dir: str = MODELS_DIR) -> FeaturePipeline:
    """Load the feature pipeline from disk."""
    return FeaturePipeline.load(os.path.join(models_dir, 'feature_pipeline.joblib'))


def evaluate_on_test(
    model: Any,
    data: Dict,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Evaluate model on held-out test set.
    
    Args:
        model: Trained model
        data: Data dict from prepare_data()
        verbose: Print results
        
    Returns:
        Dict with test metrics
    """
    X_test = data['X_test']
    y_test = data['y_test']
    target_names = data['target_names']
    
    y_pred = model.predict(X_test)
    
    results = {}
    
    if verbose:
        print("\nTest Set Evaluation")
        print("=" * 50)
    
    for i, target in enumerate(target_names):
        y_true = y_test[:, i]
        y_p = y_pred[:, i]
        
        rmse = np.sqrt(mean_squared_error(y_true, y_p))
        mae = mean_absolute_error(y_true, y_p)
        r2 = r2_score(y_true, y_p)
        mape = np.mean(np.abs((y_true - y_p) / y_true)) * 100
        
        results[target] = {'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape}
        
        if verbose:
            print(f"\n{target}:")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE: {mae:.4f}")
            print(f"  R²: {r2:.4f}")
            print(f"  MAPE: {mape:.2f}%")
    
    # Overall
    total_r2 = r2_score(y_test, y_pred)
    results['overall_r2'] = total_r2
    
    if verbose:
        print(f"\nOverall R²: {total_r2:.4f}")
    
    return results


def sensitivity_analysis(
    model: Any,
    pipeline: FeaturePipeline,
    param_name: str,
    n_points: int = 20,
    other_params: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Analyze sensitivity of outputs to a single design parameter.
    
    Varies one parameter while keeping others at reference values.
    
    Args:
        model: Trained model
        pipeline: Feature pipeline
        param_name: Parameter to vary
        n_points: Number of points to evaluate
        other_params: Optional dict to override reference values
        
    Returns:
        DataFrame with parameter values and predicted outputs
    """
    # Get parameter range
    bounds = DESIGN_RANGES.get_bounds()
    param_range = bounds[param_name]
    
    # Create parameter sweep
    param_values = np.linspace(param_range[0], param_range[1], n_points)
    
    # Base design (reference or custom)
    base_design = {
        'm': REFERENCE.m,
        'C_L': REFERENCE.C_L,
        'C_D': REFERENCE.C_D,
        'alpha_elec': REFERENCE.alpha_elec,
        'E_deploy': REFERENCE.E_deploy,
        'gamma_cool': REFERENCE.gamma_cool
    }
    
    if other_params:
        base_design.update(other_params)
    
    # Evaluate at each point
    results = []
    for val in param_values:
        design = base_design.copy()
        design[param_name] = val
        
        # Create DataFrame for pipeline
        df = pd.DataFrame([design])
        
        # Transform and predict
        X = pipeline.transform(df)
        y_pred = model.predict(X)[0]
        
        result = {param_name: val}
        for i, target in enumerate(TARGET_NAMES):
            result[target] = y_pred[i]
        results.append(result)
    
    return pd.DataFrame(results)


def full_sensitivity_analysis(
    model: Any,
    pipeline: FeaturePipeline,
    output_dir: str = OUTPUTS_DIR
) -> Dict[str, pd.DataFrame]:
    """
    Run sensitivity analysis for all parameters.
    
    Args:
        model: Trained model
        pipeline: Feature pipeline
        output_dir: Directory for plots
        
    Returns:
        Dict mapping param name to sensitivity DataFrame
    """
    param_names = DESIGN_RANGES.get_param_names()
    
    all_sensitivity = {}
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, param in enumerate(param_names):
        df = sensitivity_analysis(model, pipeline, param)
        all_sensitivity[param] = df
        
        ax = axes[i]
        
        # Plot lap_time (primary)
        color = 'steelblue'
        ax.plot(df[param], df['lap_time'], color=color, linewidth=2)
        ax.set_xlabel(param, fontsize=10)
        ax.set_ylabel('Lap Time (s)', color=color, fontsize=10)
        ax.tick_params(axis='y', labelcolor=color)
        
        # Secondary axis for energy
        ax2 = ax.twinx()
        color = 'darkorange'
        ax2.plot(df[param], df['energy_used'], color=color, linewidth=2, linestyle='--')
        ax2.set_ylabel('Energy (MJ)', color=color, fontsize=10)
        ax2.tick_params(axis='y', labelcolor=color)
        
        ax.set_title(f'Sensitivity: {param}')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Parameter Sensitivity Analysis', fontsize=14, y=1.02)
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, 'sensitivity_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved sensitivity plots to: {output_dir}/sensitivity_analysis.png")
    
    return all_sensitivity


def predict_design(
    model: Any,
    pipeline: FeaturePipeline,
    design: Dict[str, float]
) -> Dict[str, float]:
    """
    Predict performance for a single design.
    
    Args:
        model: Trained model
        pipeline: Feature pipeline
        design: Dict with design parameters
        
    Returns:
        Dict with predicted outputs
    """
    df = pd.DataFrame([design])
    X = pipeline.transform(df)
    y_pred = model.predict(X)[0]
    
    return {target: y_pred[i] for i, target in enumerate(TARGET_NAMES)}


def compare_designs(
    model: Any,
    pipeline: FeaturePipeline,
    designs: List[Dict[str, float]],
    names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compare multiple designs.
    
    Args:
        model: Trained model
        pipeline: Feature pipeline
        designs: List of design dicts
        names: Optional names for designs
        
    Returns:
        DataFrame with all designs and predictions
    """
    if names is None:
        names = [f'Design_{i}' for i in range(len(designs))]
    
    results = []
    
    for name, design in zip(names, designs):
        pred = predict_design(model, pipeline, design)
        result = {'name': name}
        result.update(design)
        result.update(pred)
        results.append(result)
    
    return pd.DataFrame(results)


def generate_evaluation_report(
    model_name: str = 'gradient_boosting',
    output_dir: str = OUTPUTS_DIR
) -> str:
    """
    Generate comprehensive evaluation report.
    
    Args:
        model_name: Name of model to evaluate
        output_dir: Output directory
        
    Returns:
        Path to report file
    """
    print("=" * 60)
    print(f"Generating Evaluation Report: {model_name}")
    print("=" * 60)
    
    # Load model and pipeline
    model = load_model(model_name)
    pipeline = load_pipeline()
    
    # Load data
    data = prepare_data()
    
    # Test evaluation
    test_results = evaluate_on_test(model, data)
    
    # Sensitivity analysis
    print("\nRunning sensitivity analysis...")
    sensitivity = full_sensitivity_analysis(model, pipeline, output_dir)
    
    # Example design comparisons
    print("\nComparing example designs...")
    
    example_designs = [
        {  # Baseline
            'm': 770, 'C_L': 1.0, 'C_D': 1.0,
            'alpha_elec': 0.47, 'E_deploy': 7.0, 'gamma_cool': 1.0
        },
        {  # Low drag (Monza style)
            'm': 760, 'C_L': 0.85, 'C_D': 0.88,
            'alpha_elec': 0.50, 'E_deploy': 8.0, 'gamma_cool': 0.9
        },
        {  # High downforce (Monaco style)
            'm': 780, 'C_L': 1.25, 'C_D': 1.15,
            'alpha_elec': 0.45, 'E_deploy': 6.5, 'gamma_cool': 1.1
        },
        {  # Aggressive electric
            'm': 750, 'C_L': 1.1, 'C_D': 1.0,
            'alpha_elec': 0.58, 'E_deploy': 8.5, 'gamma_cool': 1.2
        }
    ]
    
    design_names = ['Baseline', 'Low Drag', 'High Downforce', 'Aggressive Electric']
    
    comparison = compare_designs(model, pipeline, example_designs, design_names)
    
    print("\nDesign Comparison:")
    print(comparison[['name', 'lap_time', 'energy_used', 'thermal_risk']].to_string(index=False))
    
    # Save comparison
    comparison_path = os.path.join(output_dir, 'design_comparison.csv')
    comparison.to_csv(comparison_path, index=False)
    
    # Create summary plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Lap time comparison
    ax = axes[0]
    bars = ax.barh(comparison['name'], comparison['lap_time'], color='steelblue')
    ax.set_xlabel('Lap Time (s)')
    ax.set_title('Lap Time Comparison')
    for bar, val in zip(bars, comparison['lap_time']):
        ax.text(val + 0.5, bar.get_y() + bar.get_height()/2, f'{val:.1f}s',
                va='center', fontsize=9)
    
    # Energy comparison
    ax = axes[1]
    bars = ax.barh(comparison['name'], comparison['energy_used'], color='darkorange')
    ax.set_xlabel('Energy Used (MJ)')
    ax.set_title('Energy Comparison')
    for bar, val in zip(bars, comparison['energy_used']):
        ax.text(val + 0.1, bar.get_y() + bar.get_height()/2, f'{val:.2f}',
                va='center', fontsize=9)
    
    # Trade-off scatter
    ax = axes[2]
    ax.scatter(comparison['lap_time'], comparison['energy_used'], 
               c=comparison['thermal_risk'], cmap='RdYlGn_r', s=100, edgecolors='black')
    for i, name in enumerate(comparison['name']):
        ax.annotate(name, (comparison['lap_time'].iloc[i], comparison['energy_used'].iloc[i]),
                    textcoords='offset points', xytext=(5, 5), fontsize=8)
    ax.set_xlabel('Lap Time (s)')
    ax.set_ylabel('Energy Used (MJ)')
    ax.set_title('Lap Time vs Energy Trade-off')
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'design_comparison.png'), dpi=150)
    plt.close()
    
    print(f"\nReport saved to: {output_dir}")
    
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate F1 ML models")
    parser.add_argument(
        '--model', '-m', type=str, default='gradient_boosting',
        choices=['ridge', 'gradient_boosting', 'random_forest', 'mlp'],
        help="Model to evaluate"
    )
    
    args = parser.parse_args()
    
    generate_evaluation_report(args.model)
