"""
ML Model Training Pipeline

Trains multiple models on synthetic F1 design data:
1. Ridge Regression (baseline)
2. Gradient Boosting (main model)
3. MLP Neural Network (optional)

Multi-output regression for: lap_time, energy_used, thermal_risk
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import argparse
import joblib
from datetime import datetime
from typing import Dict, List, Tuple, Any

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.config import ML_CONFIG, MODELS_DIR, OUTPUTS_DIR, TARGET_NAMES
from src.models.features import prepare_data, FeaturePipeline


def create_models() -> Dict[str, Any]:
    """
    Create all model instances with configured hyperparameters.
    
    Returns:
        Dict mapping model name to model instance
    """
    models = {}
    
    # 1. Ridge Regression (baseline)
    # Multi-output wrapper for multi-target regression
    ridge = Ridge(alpha=ML_CONFIG.ridge_alpha)
    models['ridge'] = ridge  # Ridge natively supports multi-output
    
    # 2. Gradient Boosting
    # Need MultiOutputRegressor wrapper since GB is single-output
    gb_base = GradientBoostingRegressor(
        n_estimators=ML_CONFIG.gb_n_estimators,
        max_depth=ML_CONFIG.gb_max_depth,
        learning_rate=ML_CONFIG.gb_learning_rate,
        min_samples_leaf=ML_CONFIG.gb_min_samples_leaf,
        random_state=ML_CONFIG.random_seed
    )
    models['gradient_boosting'] = MultiOutputRegressor(gb_base)
    
    # 3. Random Forest (alternative ensemble)
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        random_state=ML_CONFIG.random_seed,
        n_jobs=-1
    )
    models['random_forest'] = rf  # RF natively supports multi-output
    
    # 4. MLP Neural Network
    mlp = MLPRegressor(
        hidden_layer_sizes=ML_CONFIG.mlp_hidden_layers,
        max_iter=ML_CONFIG.mlp_max_iter,
        alpha=ML_CONFIG.mlp_alpha,
        random_state=ML_CONFIG.random_seed,
        early_stopping=True,
        validation_fraction=0.1
    )
    models['mlp'] = mlp
    
    return models


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_names: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate model on test set.
    
    Args:
        model: Fitted model
        X_test: Test features
        y_test: Test targets
        target_names: Names of target variables
        
    Returns:
        Dict with metrics for each target
    """
    y_pred = model.predict(X_test)
    
    # Ensure 2D shape
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    
    results = {}
    
    for i, target in enumerate(target_names):
        y_true = y_test[:, i]
        y_p = y_pred[:, i]
        
        rmse = np.sqrt(mean_squared_error(y_true, y_p))
        mae = mean_absolute_error(y_true, y_p)
        r2 = r2_score(y_true, y_p)
        
        # Relative error (percentage)
        mape = np.mean(np.abs((y_true - y_p) / y_true)) * 100
        
        results[target] = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        }
    
    # Overall metrics
    total_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    total_r2 = r2_score(y_test, y_pred)
    
    results['overall'] = {
        'rmse': total_rmse,
        'r2': total_r2
    }
    
    return results


def train_and_evaluate(
    data: Dict,
    models: Dict[str, Any],
    verbose: bool = True
) -> Dict[str, Dict]:
    """
    Train all models and evaluate on validation set.
    
    Args:
        data: Prepared data from features.prepare_data()
        models: Dict of model instances
        verbose: Print progress
        
    Returns:
        Dict with trained models and evaluation results
    """
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    target_names = data['target_names']
    
    results = {}
    
    for name, model in models.items():
        if verbose:
            print(f"\nTraining {name}...")
        
        # Train
        model.fit(X_train, y_train)
        
        # Evaluate on validation set
        val_metrics = evaluate_model(model, X_val, y_val, target_names)
        
        results[name] = {
            'model': model,
            'val_metrics': val_metrics
        }
        
        if verbose:
            print(f"  Validation R²: {val_metrics['overall']['r2']:.4f}")
            print(f"  Validation RMSE: {val_metrics['overall']['rmse']:.4f}")
            for target in target_names:
                m = val_metrics[target]
                print(f"    {target}: RMSE={m['rmse']:.4f}, R²={m['r2']:.4f}, MAPE={m['mape']:.2f}%")
    
    return results


def get_feature_importance(
    model: Any,
    model_name: str,
    feature_names: List[str],
    target_names: List[str]
) -> pd.DataFrame:
    """
    Extract feature importance from model.
    
    Args:
        model: Fitted model
        model_name: Name of the model
        feature_names: Names of features
        target_names: Names of targets
        
    Returns:
        DataFrame with feature importances
    """
    importances = []
    
    if model_name == 'ridge':
        # Ridge: use absolute coefficient values
        coefs = model.coef_  # Shape: (n_targets, n_features)
        for i, target in enumerate(target_names):
            imp = np.abs(coefs[i])
            imp = imp / imp.sum()  # Normalize
            for j, feat in enumerate(feature_names):
                importances.append({
                    'feature': feat,
                    'target': target,
                    'importance': imp[j]
                })
    
    elif model_name == 'gradient_boosting':
        # Gradient Boosting: extract from each estimator
        for i, (est, target) in enumerate(zip(model.estimators_, target_names)):
            imp = est.feature_importances_
            for j, feat in enumerate(feature_names):
                importances.append({
                    'feature': feat,
                    'target': target,
                    'importance': imp[j]
                })
    
    elif model_name == 'random_forest':
        # Random Forest: average across all outputs
        imp = model.feature_importances_
        for j, feat in enumerate(feature_names):
            importances.append({
                'feature': feat,
                'target': 'all',
                'importance': imp[j]
            })
    
    elif model_name == 'mlp':
        # MLP: use first layer weights magnitude (approximate)
        weights = np.abs(model.coefs_[0])  # First layer
        imp = weights.mean(axis=1)  # Average across hidden units
        imp = imp / imp.sum()
        for j, feat in enumerate(feature_names):
            importances.append({
                'feature': feat,
                'target': 'all',
                'importance': imp[j]
            })
    
    return pd.DataFrame(importances)


def save_models(
    results: Dict,
    data: Dict,
    output_dir: str = MODELS_DIR
) -> Dict[str, str]:
    """
    Save trained models and pipeline to disk.
    
    Args:
        results: Training results with models
        data: Data dict with pipeline
        output_dir: Directory to save models
        
    Returns:
        Dict mapping model name to saved path
    """
    os.makedirs(output_dir, exist_ok=True)
    
    saved_paths = {}
    
    # Save each model
    for name, res in results.items():
        model_path = os.path.join(output_dir, f'{name}_model.joblib')
        joblib.dump(res['model'], model_path)
        saved_paths[name] = model_path
        print(f"Saved {name} to: {model_path}")
    
    # Save feature pipeline
    pipeline_path = os.path.join(output_dir, 'feature_pipeline.joblib')
    data['pipeline'].save(pipeline_path)
    saved_paths['pipeline'] = pipeline_path
    
    # Save metadata
    metadata = {
        'feature_names': data['feature_names'],
        'target_names': data['target_names'],
        'train_size': len(data['X_train']),
        'val_size': len(data['X_val']),
        'test_size': len(data['X_test']),
        'timestamp': datetime.now().isoformat()
    }
    
    # Add validation metrics to metadata
    for name, res in results.items():
        metadata[f'{name}_val_r2'] = res['val_metrics']['overall']['r2']
        metadata[f'{name}_val_rmse'] = res['val_metrics']['overall']['rmse']
    
    metadata_path = os.path.join(output_dir, 'training_metadata.joblib')
    joblib.dump(metadata, metadata_path)
    saved_paths['metadata'] = metadata_path
    
    return saved_paths


def plot_comparison(
    results: Dict,
    data: Dict,
    output_dir: str = OUTPUTS_DIR
):
    """
    Create comparison plots for all models.
    
    Args:
        results: Training results
        data: Data dict
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    target_names = data['target_names']
    X_val = data['X_val']
    y_val = data['y_val']
    
    # Model comparison bar chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    model_names = list(results.keys())
    r2_scores = [results[m]['val_metrics']['overall']['r2'] for m in model_names]
    rmse_scores = [results[m]['val_metrics']['overall']['rmse'] for m in model_names]
    
    ax = axes[0]
    bars = ax.bar(model_names, r2_scores, color=['steelblue', 'darkorange', 'seagreen', 'indianred'])
    ax.set_ylabel('R² Score')
    ax.set_title('Model Comparison: R² (higher is better)')
    ax.set_ylim(0, 1.05)
    for bar, score in zip(bars, r2_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{score:.3f}', ha='center', fontsize=10)
    
    ax = axes[1]
    bars = ax.bar(model_names, rmse_scores, color=['steelblue', 'darkorange', 'seagreen', 'indianred'])
    ax.set_ylabel('RMSE')
    ax.set_title('Model Comparison: RMSE (lower is better)')
    for bar, score in zip(bars, rmse_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{score:.3f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=150)
    plt.close()
    
    # Prediction vs Actual scatter plots (for best model)
    best_model_name = max(results, key=lambda m: results[m]['val_metrics']['overall']['r2'])
    best_model = results[best_model_name]['model']
    y_pred = best_model.predict(X_val)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for i, (ax, target) in enumerate(zip(axes, target_names)):
        y_true = y_val[:, i]
        y_p = y_pred[:, i]
        
        ax.scatter(y_true, y_p, alpha=0.5, s=10)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_p.min())
        max_val = max(y_true.max(), y_p.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        ax.set_xlabel(f'Actual {target}')
        ax.set_ylabel(f'Predicted {target}')
        ax.set_title(f'{target}\nR² = {r2_score(y_true, y_p):.4f}')
    
    plt.suptitle(f'Best Model: {best_model_name}', y=1.02, fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'prediction_scatter.png'), dpi=150)
    plt.close()
    
    print(f"\nPlots saved to: {output_dir}")


def run_training_pipeline(
    data_path: str = None,
    save: bool = True,
    plot: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Run complete training pipeline.
    
    Args:
        data_path: Path to training data CSV
        save: Whether to save models
        plot: Whether to generate plots
        verbose: Print progress
        
    Returns:
        Dict with results, data, and saved paths
    """
    print("=" * 60)
    print("F1 ML Design System - Model Training")
    print("=" * 60)
    
    # Prepare data
    print("\n1. Loading and preparing data...")
    data = prepare_data(data_path=data_path)
    
    # Create models
    print("\n2. Creating models...")
    models = create_models()
    for name in models:
        print(f"  - {name}")
    
    # Train and evaluate
    print("\n3. Training and evaluating...")
    results = train_and_evaluate(data, models, verbose=verbose)
    
    # Summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    
    for name, res in results.items():
        print(f"\n{name.upper()}")
        print(f"  Overall R²: {res['val_metrics']['overall']['r2']:.4f}")
        print(f"  Overall RMSE: {res['val_metrics']['overall']['rmse']:.4f}")
    
    # Best model
    best = max(results, key=lambda m: results[m]['val_metrics']['overall']['r2'])
    print(f"\n🏆 Best Model: {best} (R² = {results[best]['val_metrics']['overall']['r2']:.4f})")
    
    # Feature importance for best model
    print("\n4. Extracting feature importance...")
    imp_df = get_feature_importance(
        results[best]['model'],
        best,
        data['feature_names'],
        data['target_names']
    )
    
    if not imp_df.empty:
        # Aggregate across targets
        avg_importance = imp_df.groupby('feature')['importance'].mean().sort_values(ascending=False)
        print(f"\nTop Features ({best}):")
        for feat, imp in avg_importance.head(5).items():
            print(f"  {feat}: {imp:.3f}")
    
    # Save
    if save:
        print("\n5. Saving models...")
        saved_paths = save_models(results, data)
    else:
        saved_paths = {}
    
    # Plot
    if plot:
        print("\n6. Generating plots...")
        plot_comparison(results, data)
    
    print("\n✅ Training complete!")
    
    return {
        'results': results,
        'data': data,
        'saved_paths': saved_paths,
        'feature_importance': imp_df
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train F1 ML models")
    parser.add_argument(
        '--data', '-d', type=str, default=None,
        help="Path to training data CSV"
    )
    parser.add_argument(
        '--no-save', action='store_true',
        help="Don't save models"
    )
    parser.add_argument(
        '--no-plot', action='store_true',
        help="Don't generate plots"
    )
    parser.add_argument(
        '--test', action='store_true',
        help="Quick test mode (fewer samples)"
    )
    
    args = parser.parse_args()
    
    run_training_pipeline(
        data_path=args.data,
        save=not args.no_save,
        plot=not args.no_plot
    )
