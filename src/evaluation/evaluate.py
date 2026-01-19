"""
Model Evaluation Module for F1 Track Time Prediction

Computes:
- Quantitative metrics (RMSE, MAE, MAPE, R²)
- Physical consistency checks
- Error distribution analysis
- Markdown report generation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore')


class ModelEvaluator:
    """
    Comprehensive model evaluator for F1 lap time prediction.
    """
    
    def __init__(self, output_dir: str = "outputs"):
        """Initialize evaluator with output directory."""
        self.output_dir = Path(output_dir)
        self.figures_dir = self.output_dir / "figures"
        self.reports_dir = self.output_dir / "reports"
        
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        self.results: Dict[str, Dict] = {}
    
    def evaluate_model(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        model_name: str
    ) -> Dict[str, Any]:
        """
        Evaluate a model's performance.
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target values
            feature_names: List of feature names
            model_name: Name of the model
            
        Returns:
            Dict with evaluation results
        """
        print(f"\n[Evaluating {model_name}]")
        
        # Make predictions
        y_pred = model.predict(X)
        
        # Compute metrics
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        # MAPE
        mask = y != 0
        mape = np.mean(np.abs((y[mask] - y_pred[mask]) / y[mask])) * 100
        
        # Max error
        max_error = np.max(np.abs(y - y_pred))
        
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  R²:   {r2:.4f}")
        print(f"  MAPE: {mape:.2f}%")
        
        # Physical consistency
        n_total = len(y_pred)
        n_valid = np.sum((y_pred >= 100) & (y_pred <= 140))
        validity = n_valid / n_total * 100
        print(f"  Validity: {validity:.1f}%")
        
        # Store results
        result = {
            'model_name': model_name,
            'predictions': y_pred,
            'metrics': {
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'mape': mape,
                'max_error': max_error
            },
            'consistency': {
                'total': n_total,
                'valid': n_valid,
                'validity': validity,
                'min_pred': np.min(y_pred),
                'max_pred': np.max(y_pred),
                'mean_pred': np.mean(y_pred),
                'std_pred': np.std(y_pred)
            }
        }
        
        self.results[model_name] = result
        return result
    
    def plot_error_distribution(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str
    ) -> None:
        """Plot error distribution histogram."""
        errors = y_pred - y_true
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Error histogram
        axes[0].hist(errors, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0].axvline(0, color='red', linestyle='--', linewidth=2)
        axes[0].set_xlabel('Prediction Error (s)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'{model_name}: Error Distribution')
        
        # Predicted vs Actual scatter
        axes[1].scatter(y_true, y_pred, alpha=0.1, s=5, color='steelblue')
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        axes[1].set_xlabel('Actual Lap Time (s)')
        axes[1].set_ylabel('Predicted Lap Time (s)')
        axes[1].set_title(f'{model_name}: Predicted vs Actual')
        
        plt.tight_layout()
        
        save_path = self.figures_dir / f"{model_name}_error_dist.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Error distribution plot saved to {save_path}")
        plt.close()
    
    def plot_residuals(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        feature_names: List[str],
        model_name: str
    ) -> None:
        """Plot residual analysis."""
        residuals = y_pred - y_true
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Residuals vs predicted
        axes[0].scatter(y_pred, residuals, alpha=0.1, s=5, color='steelblue')
        axes[0].axhline(0, color='red', linestyle='--', linewidth=2)
        axes[0].set_xlabel('Predicted Lap Time (s)')
        axes[0].set_ylabel('Residual (s)')
        axes[0].set_title(f'{model_name}: Residuals vs Predicted')
        
        # Residuals histogram
        axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        axes[1].axvline(0, color='red', linestyle='--', linewidth=2)
        axes[1].set_xlabel('Residual (s)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'{model_name}: Residual Distribution')
        
        plt.tight_layout()
        
        save_path = self.figures_dir / f"{model_name}_residuals.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Residuals plot saved to {save_path}")
        plt.close()
    
    def generate_report(self, model_name: str) -> None:
        """Generate markdown evaluation report."""
        if model_name not in self.results:
            print(f"  No results found for {model_name}")
            return
        
        result = self.results[model_name]
        metrics = result['metrics']
        consistency = result['consistency']
        
        report = f"""# Model Evaluation Report: {model_name}

## Performance Metrics

| Metric | Value |
|--------|-------|
| RMSE | {metrics['rmse']:.4f} s |
| MAE | {metrics['mae']:.4f} s |
| R² | {metrics['r2']:.4f} |
| MAPE | {metrics['mape']:.2f}% |
| Max Error | {metrics['max_error']:.4f} s |

## Physical Consistency Check

| Check | Result |
|-------|--------|
| Total Predictions | {consistency['total']:,} |
| Valid Predictions | {consistency['valid']:,} |
| Validity Rate | {consistency['validity']:.2f}% |

## Prediction Statistics

| Statistic | Value |
|-----------|-------|
| Min Prediction | {consistency['min_pred']:.2f} s |
| Max Prediction | {consistency['max_pred']:.2f} s |
| Mean Prediction | {consistency['mean_pred']:.2f} s |
| Std Prediction | {consistency['std_pred']:.2f} s |

## Interpretation

- **RMSE of {metrics['rmse']:.3f}s** means average prediction error is about {metrics['rmse']:.1f} seconds
- **R² of {metrics['r2']:.3f}** indicates the model explains {metrics['r2']*100:.1f}% of variance
- **{consistency['validity']:.1f}%** of predictions are physically plausible

## Plots

- Error distribution: `figures/{model_name}_error_dist.png`
- Residuals: `figures/{model_name}_residuals.png`
"""
        
        report_path = self.reports_dir / f"{model_name}_evaluation.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"  Report saved to {report_path}")


def main():
    """Demo evaluation."""
    print("Model Evaluation module ready.")


if __name__ == "__main__":
    main()
