"""
Sensitivity Analysis Module for F1 Track Time Prediction

Performs:
- Feature importance analysis
- Partial dependence plots
- Monotonicity checks
- Out-of-distribution testing
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Any
import warnings

warnings.filterwarnings('ignore')


class SensitivityAnalyzer:
    """
    Comprehensive sensitivity analysis for F1 lap time models.
    """
    
    def __init__(self, output_dir: str = "outputs"):
        """Initialize analyzer with output directory."""
        self.output_dir = Path(output_dir)
        self.figures_dir = self.output_dir / "figures"
        self.reports_dir = self.output_dir / "reports"
        
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
    
    def compute_feature_importance(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: List[str]
    ) -> pd.DataFrame:
        """
        Extract feature importance from trained model.
        """
        # Handle MultiOutputRegressor
        if hasattr(model, 'estimators_'):
            importances = np.mean([
                getattr(est, 'feature_importances_', np.zeros(len(feature_names)))
                for est in model.estimators_
            ], axis=0)
        elif hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
        else:
            importances = np.zeros(len(feature_names))
        
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        df = df.sort_values('importance', ascending=False).reset_index(drop=True)
        df['rank'] = range(1, len(df) + 1)
        
        return df
    
    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        model_name: str = "model",
        top_n: int = 15
    ) -> None:
        """Plot feature importance bar chart."""
        df = importance_df.head(top_n)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(df['feature'], df['importance'], color='steelblue')
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')
        ax.set_title(f'{model_name}: Top {top_n} Feature Importances')
        ax.invert_yaxis()
        
        plt.tight_layout()
        
        save_path = self.figures_dir / f"{model_name}_feature_importance.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Feature importance plot saved to {save_path}")
        plt.close()
    
    def check_monotonicity(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: List[str]
    ) -> pd.DataFrame:
        """
        Check if model predictions have expected monotonic relationships.
        """
        # Use sample for speed
        sample_idx = np.random.choice(len(X), min(1000, len(X)), replace=False)
        X_sample = X[sample_idx]
        
        expected = {
            'mass': 'positive',
            'c_d': 'positive',
            'c_l': 'negative',
            'power_to_weight': 'negative',
            'aero_efficiency': 'negative',
        }
        
        results = []
        for i, feat_name in enumerate(feature_names):
            # Compute correlation
            feat_values = np.linspace(X_sample[:, i].min(), X_sample[:, i].max(), 20)
            
            pd_values = []
            for val in feat_values:
                X_mod = X_sample.copy()
                X_mod[:, i] = val
                preds = model.predict(X_mod)
                pd_values.append(np.mean(preds))
            
            corr = np.corrcoef(feat_values, pd_values)[0, 1]
            
            if corr > 0.1:
                direction = 'positive'
            elif corr < -0.1:
                direction = 'negative'
            else:
                direction = 'neutral'
            
            exp = expected.get(feat_name, 'unknown')
            consistent = (exp == 'unknown') or (exp == direction)
            
            results.append({
                'feature': feat_name,
                'correlation': corr,
                'actual': direction,
                'expected': exp,
                'consistent': consistent
            })
        
        return pd.DataFrame(results)
    
    def plot_partial_dependence(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: List[str],
        model_name: str = "model",
        top_n: int = 6
    ) -> None:
        """Plot partial dependence for top features."""
        # Use sample
        sample_idx = np.random.choice(len(X), min(1000, len(X)), replace=False)
        X_sample = X[sample_idx]
        
        # Get top features by variance
        variances = np.var(X_sample, axis=0)
        top_indices = np.argsort(variances)[-top_n:][::-1]
        
        n_cols = 3
        n_rows = (top_n + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        axes = axes.flatten()
        
        for i, feat_idx in enumerate(top_indices):
            feat_name = feature_names[feat_idx]
            feat_values = np.linspace(X_sample[:, feat_idx].min(), X_sample[:, feat_idx].max(), 30)
            
            pd_values = []
            for val in feat_values:
                X_mod = X_sample.copy()
                X_mod[:, feat_idx] = val
                preds = model.predict(X_mod)
                pd_values.append(np.mean(preds))
            
            axes[i].plot(feat_values, pd_values, linewidth=2, color='steelblue')
            axes[i].set_xlabel(feat_name)
            axes[i].set_ylabel('Lap Time (s)')
            axes[i].set_title(f'PDP: {feat_name}')
            axes[i].grid(True, alpha=0.3)
        
        for i in range(top_n, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        save_path = self.figures_dir / f"{model_name}_partial_dependence.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Partial dependence plot saved to {save_path}")
        plt.close()
    
    def test_out_of_distribution(
        self,
        model: Any,
        feature_names: List[str],
        param_bounds: Dict[str, tuple]
    ) -> pd.DataFrame:
        """
        Test model on out-of-distribution inputs.
        """
        results = []
        
        # Generate OOD samples
        n_samples = 100
        
        for scale in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
            # Generate samples at scaled bounds
            X_test = []
            for feat in feature_names:
                if feat in param_bounds:
                    low, high = param_bounds[feat]
                    center = (low + high) / 2
                    scaled_low = center - (center - low) * scale
                    scaled_high = center + (high - center) * scale
                    X_test.append(np.random.uniform(scaled_low, scaled_high, n_samples))
                else:
                    X_test.append(np.random.normal(0, 1, n_samples))
            
            X_test = np.column_stack(X_test)
            preds = model.predict(X_test)
            
            results.append({
                'scale': scale,
                'mean_pred': np.mean(preds),
                'std_pred': np.std(preds),
                'min_pred': np.min(preds),
                'max_pred': np.max(preds)
            })
        
        return pd.DataFrame(results)
    
    def generate_sensitivity_report(
        self,
        importance_df: pd.DataFrame,
        monotonicity_df: pd.DataFrame,
        ood_df: pd.DataFrame,
        model_name: str
    ) -> None:
        """Generate markdown sensitivity report."""
        
        report = f"""# Sensitivity Analysis: {model_name}

## Feature Importance (Top 10)

| Rank | Feature | Importance |
|------|---------|------------|
"""
        for _, row in importance_df.head(10).iterrows():
            report += f"| {row['rank']} | {row['feature']} | {row['importance']:.4f} |\n"
        
        report += """

## Monotonicity Check

| Feature | Correlation | Direction | Expected | Consistent |
|---------|-------------|-----------|----------|------------|
"""
        for _, row in monotonicity_df.head(10).iterrows():
            check = "✓" if row['consistent'] else "✗"
            report += f"| {row['feature']} | {row['correlation']:.3f} | {row['actual']} | {row['expected']} | {check} |\n"
        
        n_consistent = monotonicity_df['consistent'].sum()
        n_total = len(monotonicity_df)
        
        report += f"""

## Out-of-Distribution Behavior

| Scale | Mean Pred | Std Pred | Min | Max |
|-------|-----------|----------|-----|-----|
"""
        for _, row in ood_df.iterrows():
            report += f"| {row['scale']:.1f} | {row['mean_pred']:.2f} | {row['std_pred']:.2f} | {row['min_pred']:.2f} | {row['max_pred']:.2f} |\n"
        
        report += f"""

## Summary

- **Top Feature**: {importance_df.iloc[0]['feature']}
- **Physics Consistency**: {n_consistent}/{n_total} features consistent

## Plots

- Feature importance: `figures/{model_name}_feature_importance.png`
- Partial dependence: `figures/{model_name}_partial_dependence.png`
"""
        
        report_path = self.reports_dir / f"{model_name}_sensitivity.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"  Sensitivity report saved to {report_path}")


def main():
    """Demo sensitivity analysis."""
    print("Sensitivity Analysis module ready.")


if __name__ == "__main__":
    main()
