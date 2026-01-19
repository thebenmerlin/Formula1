"""
Model Evaluation Module for F1 Track Time Prediction

Comprehensive evaluation including:
- Quantitative metrics (RMSE, MAE, MAPE, R²)
- Per-segment, per-sector, and per-lap analysis
- Error distribution analysis
- Physical consistency checks
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib


class ModelEvaluator:
    """
    Comprehensive model evaluation for lap time predictions.
    
    Reports metrics at segment, sector, and lap levels.
    Performs consistency and robustness checks.
    """
    
    SEGMENT_COLS = [f'segment_{i}' for i in range(1, 21)]
    SECTOR_COLS = ['sector_1', 'sector_2', 'sector_3']
    
    def __init__(self, output_dir: str = "outputs"):
        """Initialize evaluator with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_history: List[Dict] = []
        
    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        name: str = "prediction"
    ) -> Dict[str, float]:
        """
        Compute comprehensive metrics for predictions.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            name: Name for logging
            
        Returns:
            Dict of metric names to values
        """
        # Handle multi-output
        if len(y_true.shape) > 1:
            y_true = y_true.mean(axis=1)
            y_pred = y_pred.mean(axis=1)
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # MAPE (handle zero values)
        mask = y_true != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = np.nan
        
        # Error statistics
        errors = y_pred - y_true
        max_error = np.max(np.abs(errors))
        p95_error = np.percentile(np.abs(errors), 95)
        p99_error = np.percentile(np.abs(errors), 99)
        
        metrics = {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2,
            'max_error': max_error,
            'p95_error': p95_error,
            'p99_error': p99_error,
            'mean_error': np.mean(errors),
            'std_error': np.std(errors)
        }
        
        return metrics
    
    def evaluate_model(
        self,
        model: Any,
        X: np.ndarray,
        y_true: np.ndarray,
        feature_names: Optional[List[str]] = None,
        model_name: str = "model"
    ) -> Dict:
        """
        Full model evaluation with comprehensive reporting.
        
        Args:
            model: Trained model
            X: Feature matrix
            y_true: Ground truth targets
            feature_names: List of feature names
            model_name: Name for reporting
            
        Returns:
            Dict with all evaluation results
        """
        print("\n" + "=" * 60)
        print(f"Evaluating Model: {model_name}")
        print("=" * 60)
        
        # Predictions
        y_pred = model.predict(X)
        
        # Core metrics
        metrics = self.compute_metrics(y_true, y_pred, model_name)
        
        print("\n[1] Core Metrics:")
        print(f"  RMSE:       {metrics['rmse']:.4f} s")
        print(f"  MAE:        {metrics['mae']:.4f} s")
        print(f"  MAPE:       {metrics['mape']:.2f}%")
        print(f"  R²:         {metrics['r2']:.4f}")
        
        print("\n[2] Error Distribution:")
        print(f"  Max Error:  {metrics['max_error']:.4f} s")
        print(f"  P95 Error:  {metrics['p95_error']:.4f} s")
        print(f"  P99 Error:  {metrics['p99_error']:.4f} s")
        print(f"  Mean Bias:  {metrics['mean_error']:.4f} s")
        
        # Physical consistency checks
        print("\n[3] Physical Consistency:")
        consistency = self._check_physical_consistency(X, y_pred, y_true, feature_names)
        
        results = {
            'model_name': model_name,
            'metrics': metrics,
            'consistency': consistency,
            'predictions': y_pred
        }
        
        self.metrics_history.append({
            'model': model_name,
            **metrics
        })
        
        return results
    
    def _check_physical_consistency(
        self,
        X: np.ndarray,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Dict:
        """
        Check if predictions follow physical intuition.
        
        Expected relationships:
        - Higher mass → Higher lap time
        - Higher C_L (downforce) → Lower lap time
        - Higher C_D (drag) → Higher lap time
        """
        if feature_names is None:
            print("  (Feature names not provided - skipping detailed checks)")
            return {}
        
        checks = {}
        
        # Find feature indices
        feature_idx = {name: i for i, name in enumerate(feature_names)}
        
        # Check mass-laptime relationship
        if 'mass' in feature_idx:
            idx = feature_idx['mass']
            corr = np.corrcoef(X[:, idx], y_pred)[0, 1]
            checks['mass_correlation'] = corr
            expected = corr > 0  # Higher mass should mean higher lap time
            status = "✓ PASS" if expected else "✗ FAIL"
            print(f"  Mass → Lap time correlation: {corr:.3f} (expect positive) {status}")
        
        # Check C_L relationship
        if 'c_l' in feature_idx:
            idx = feature_idx['c_l']
            corr = np.corrcoef(X[:, idx], y_pred)[0, 1]
            checks['cl_correlation'] = corr
            expected = corr < 0  # Higher downforce should mean lower lap time
            status = "✓ PASS" if expected else "✗ FAIL"
            print(f"  C_L → Lap time correlation: {corr:.3f} (expect negative) {status}")
        
        # Check C_D relationship
        if 'c_d' in feature_idx:
            idx = feature_idx['c_d']
            corr = np.corrcoef(X[:, idx], y_pred)[0, 1]
            checks['cd_correlation'] = corr
            expected = corr > 0  # Higher drag should mean higher lap time
            status = "✓ PASS" if expected else "✗ FAIL"
            print(f"  C_D → Lap time correlation: {corr:.3f} (expect positive) {status}")
        
        # Check prediction range
        min_pred = y_pred.min()
        max_pred = y_pred.max()
        min_true = y_true.min()
        max_true = y_true.max()
        
        range_valid = (min_pred >= min_true * 0.9) and (max_pred <= max_true * 1.1)
        status = "✓ PASS" if range_valid else "⚠ WARNING"
        print(f"  Prediction range: [{min_pred:.2f}, {max_pred:.2f}] vs truth [{min_true:.2f}, {max_true:.2f}] {status}")
        checks['range_valid'] = range_valid
        
        return checks
    
    def plot_error_distribution(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "model"
    ) -> str:
        """Plot error distribution histogram."""
        errors = y_pred - y_true
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Error histogram
        axes[0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
        axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[0].set_xlabel('Prediction Error (s)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'{model_name} - Error Distribution')
        
        # Predicted vs Actual
        axes[1].scatter(y_true, y_pred, alpha=0.3, s=5)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        axes[1].set_xlabel('Actual Lap Time (s)')
        axes[1].set_ylabel('Predicted Lap Time (s)')
        axes[1].set_title(f'{model_name} - Predicted vs Actual')
        
        plt.tight_layout()
        
        save_path = self.output_dir / "figures" / f"{model_name}_error_dist.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        print(f"  Saved: {save_path}")
        return str(save_path)
    
    def plot_residuals(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        feature_names: List[str],
        model_name: str = "model"
    ) -> str:
        """Plot residuals vs each feature."""
        errors = y_pred - y_true
        n_features = min(6, X.shape[1])  # Show first 6 features
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i in range(n_features):
            axes[i].scatter(X[:, i], errors, alpha=0.2, s=5)
            axes[i].axhline(y=0, color='red', linestyle='--')
            axes[i].set_xlabel(feature_names[i])
            axes[i].set_ylabel('Residual (s)')
            axes[i].set_title(f'Residuals vs {feature_names[i]}')
        
        plt.suptitle(f'{model_name} - Residual Analysis', fontsize=14)
        plt.tight_layout()
        
        save_path = self.output_dir / "figures" / f"{model_name}_residuals.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        print(f"  Saved: {save_path}")
        return str(save_path)
    
    def generate_report(self, model_name: str = "model") -> str:
        """Generate evaluation report as markdown."""
        if not self.metrics_history:
            return "No evaluations performed yet."
        
        latest = self.metrics_history[-1]
        
        report = f"""# Model Evaluation Report: {model_name}

## Summary Metrics

| Metric | Value |
|--------|-------|
| RMSE | {latest['rmse']:.4f} s |
| MAE | {latest['mae']:.4f} s |
| MAPE | {latest['mape']:.2f}% |
| R² | {latest['r2']:.4f} |

## Error Distribution

| Statistic | Value |
|-----------|-------|
| Max Error | {latest['max_error']:.4f} s |
| 95th Percentile | {latest['p95_error']:.4f} s |
| 99th Percentile | {latest['p99_error']:.4f} s |
| Mean Bias | {latest['mean_error']:.4f} s |
| Std Dev | {latest['std_error']:.4f} s |

## Interpretation

- The model achieves a lap-time RMSE of **{latest['rmse']:.4f} seconds**.
- Mean absolute error is **{latest['mae']:.4f} seconds**.
- R² score of **{latest['r2']:.4f}** indicates the model explains {latest['r2']*100:.1f}% of variance.
"""
        
        report_path = self.output_dir / "reports" / f"{model_name}_report.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"\nReport saved to: {report_path}")
        return str(report_path)


def main():
    """Run evaluation on trained model."""
    print("=" * 60)
    print("Formula1 Model Evaluation")
    print("=" * 60)
    
    from src.features.engineering import FeatureEngineer
    
    # Load data
    data_path = Path("data/synthetic/lap_times_100k.parquet")
    model_path = Path("models/final/best_lap_predictor.joblib")
    
    if not data_path.exists():
        print(f"Data not found: {data_path}")
        return
    
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Run training first: python -m src.models.train")
        return
    
    # Load data
    print("\n[1/4] Loading data...")
    df = pd.read_parquet(data_path)
    
    # Feature engineering
    print("[2/4] Engineering features...")
    engineer = FeatureEngineer(include_interactions=True)
    base_cols = ['mass', 'c_l', 'c_d', 'alpha_elec', 'e_deploy', 'gamma_cool']
    df_features = engineer.fit_transform(df[base_cols])
    feature_cols = list(df_features.columns)
    
    X = df_features.values
    y = df['lap_time'].values
    
    # Load model
    print("[3/4] Loading model...")
    model_data = joblib.load(model_path)
    model = model_data['model']
    model_name = model_data['name']
    
    # Evaluate
    print("[4/4] Running evaluation...")
    evaluator = ModelEvaluator()
    
    results = evaluator.evaluate_model(
        model, X, y,
        feature_names=feature_cols,
        model_name=model_name
    )
    
    # Generate plots
    print("\n[Plots]")
    evaluator.plot_error_distribution(y, results['predictions'], model_name)
    evaluator.plot_residuals(X, y, results['predictions'], feature_cols, model_name)
    
    # Generate report
    evaluator.generate_report(model_name)
    
    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()
