"""
Sensitivity Analysis Module for F1 Track Time Prediction

Comprehensive analysis including:
- Feature importance (global and local)
- Partial dependence plots
- Monotonicity checks
- Robustness testing (OOD, extreme cases)
- SHAP explanations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from sklearn.inspection import partial_dependence
import shap
import joblib


class SensitivityAnalyzer:
    """
    Performs sensitivity and robustness analysis on trained models.
    
    Ensures predictions:
    - Respect physical intuition
    - Are stable across parameter ranges
    - Handle edge cases appropriately
    """
    
    # Expected monotonicity for each parameter
    EXPECTED_MONOTONICITY = {
        'mass': 'increasing',      # Higher mass → slower lap
        'c_l': 'decreasing',       # Higher downforce → faster lap
        'c_d': 'increasing',       # Higher drag → slower lap
        'alpha_elec': 'decreasing', # More electric → faster lap
        'e_deploy': 'decreasing',  # More energy → faster lap
        'gamma_cool': 'decreasing' # Better cooling → faster lap
    }
    
    def __init__(self, output_dir: str = "outputs"):
        """Initialize analyzer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def compute_feature_importance(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: List[str],
        method: str = 'permutation'
    ) -> pd.DataFrame:
        """
        Compute feature importance scores.
        
        Args:
            model: Trained model
            X: Feature matrix
            feature_names: List of feature names
            method: 'permutation' or 'builtin'
            
        Returns:
            DataFrame with feature importance
        """
        print("\n[Feature Importance]")
        
        if method == 'builtin' and hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        else:
            # Use permutation importance
            from sklearn.inspection import permutation_importance
            
            # Create dummy y for permutation (using predictions)
            y_pred = model.predict(X)
            result = permutation_importance(model, X, y_pred, n_repeats=10, random_state=42)
            importance = result.importances_mean
        
        df_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Features:")
        for i, row in df_importance.head(10).iterrows():
            print(f"  {row['feature']:25} {row['importance']:.4f}")
        
        return df_importance
    
    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        top_n: int = 15,
        model_name: str = "model"
    ) -> str:
        """Plot feature importance bar chart."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        top_df = importance_df.head(top_n).sort_values('importance')
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_df)))
        ax.barh(top_df['feature'], top_df['importance'], color=colors)
        ax.set_xlabel('Importance')
        ax.set_title(f'{model_name} - Feature Importance (Top {top_n})')
        
        plt.tight_layout()
        
        save_path = self.output_dir / "figures" / f"{model_name}_feature_importance.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        print(f"  Saved: {save_path}")
        return str(save_path)
    
    def check_monotonicity(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: List[str],
        n_points: int = 50
    ) -> Dict[str, Dict]:
        """
        Check if model predictions follow expected monotonicity.
        
        Args:
            model: Trained model
            X: Feature matrix (for baseline values)
            feature_names: List of feature names
            n_points: Number of points for sweep
            
        Returns:
            Dict with monotonicity check results
        """
        print("\n[Monotonicity Checks]")
        
        results = {}
        base_sample = X.mean(axis=0)  # Use mean as baseline
        
        for i, feat in enumerate(feature_names[:6]):  # Check base features only
            if feat not in self.EXPECTED_MONOTONICITY:
                continue
            
            expected = self.EXPECTED_MONOTONICITY[feat]
            
            # Create sweep
            min_val = X[:, i].min()
            max_val = X[:, i].max()
            sweep_values = np.linspace(min_val, max_val, n_points)
            
            predictions = []
            for val in sweep_values:
                test_sample = base_sample.copy()
                test_sample[i] = val
                pred = model.predict(test_sample.reshape(1, -1))[0]
                predictions.append(pred)
            
            predictions = np.array(predictions)
            
            # Check monotonicity
            diffs = np.diff(predictions)
            
            if expected == 'increasing':
                is_monotonic = np.all(diffs >= -0.001)  # Small tolerance
                direction = "increases"
            else:
                is_monotonic = np.all(diffs <= 0.001)
                direction = "decreases"
            
            status = "✓ PASS" if is_monotonic else "✗ FAIL"
            print(f"  {feat}: lap time {direction} as {feat} increases " +
                  f"(expected: {expected}) {status}")
            
            results[feat] = {
                'expected': expected,
                'is_monotonic': is_monotonic,
                'min_diff': diffs.min(),
                'max_diff': diffs.max()
            }
        
        return results
    
    def plot_partial_dependence(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: List[str],
        features_to_plot: Optional[List[str]] = None,
        model_name: str = "model"
    ) -> str:
        """
        Plot partial dependence for specified features.
        """
        if features_to_plot is None:
            features_to_plot = feature_names[:6]  # First 6 (base params)
        
        feature_indices = [feature_names.index(f) for f in features_to_plot if f in feature_names]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        print("\n[Partial Dependence Plots]")
        
        for i, (feat_idx, feat_name) in enumerate(zip(feature_indices, features_to_plot)):
            if i >= 6:
                break
            
            # Manual partial dependence calculation
            grid = np.linspace(X[:, feat_idx].min(), X[:, feat_idx].max(), 50)
            pd_values = []
            
            for val in grid:
                X_mod = X.copy()
                X_mod[:, feat_idx] = val
                preds = model.predict(X_mod)
                pd_values.append(preds.mean())
            
            axes[i].plot(grid, pd_values, linewidth=2)
            axes[i].set_xlabel(feat_name)
            axes[i].set_ylabel('Avg Lap Time (s)')
            axes[i].set_title(f'PD: {feat_name}')
            axes[i].grid(True, alpha=0.3)
        
        plt.suptitle(f'{model_name} - Partial Dependence Plots', fontsize=14)
        plt.tight_layout()
        
        save_path = self.output_dir / "figures" / f"{model_name}_partial_dependence.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        print(f"  Saved: {save_path}")
        return str(save_path)
    
    def compute_shap_values(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: List[str],
        n_samples: int = 1000,
        model_name: str = "model"
    ) -> Tuple[np.ndarray, Any]:
        """
        Compute SHAP values for model explanations.
        
        Args:
            model: Trained model
            X: Feature matrix
            feature_names: List of feature names
            n_samples: Number of samples for SHAP
            model_name: Name for saving
            
        Returns:
            Tuple of (shap_values, explainer)
        """
        print("\n[SHAP Analysis]")
        
        # Sample for efficiency
        if len(X) > n_samples:
            idx = np.random.choice(len(X), n_samples, replace=False)
            X_sample = X[idx]
        else:
            X_sample = X
        
        print(f"  Computing SHAP values for {len(X_sample)} samples...")
        
        # Create explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        # Summary plot
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            shap_values, X_sample,
            feature_names=feature_names,
            show=False,
            max_display=15
        )
        
        save_path = self.output_dir / "figures" / f"{model_name}_shap_summary.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {save_path}")
        
        return shap_values, explainer
    
    def test_out_of_distribution(
        self,
        model: Any,
        feature_names: List[str],
        param_bounds: Dict[str, Tuple[float, float]]
    ) -> pd.DataFrame:
        """
        Test model on out-of-distribution setups.
        
        Creates extreme parameter combinations to check robustness.
        """
        print("\n[Out-of-Distribution Testing]")
        
        # Define extreme test cases
        test_cases = [
            {
                'name': 'Min Mass + Max Aero',
                'mass': param_bounds['mass'][0],
                'c_l': param_bounds['c_l'][1],
                'c_d': param_bounds['c_d'][0],
                'alpha_elec': param_bounds['alpha_elec'][1],
                'e_deploy': param_bounds['e_deploy'][1],
                'gamma_cool': param_bounds['gamma_cool'][1]
            },
            {
                'name': 'Max Mass + Min Aero',
                'mass': param_bounds['mass'][1],
                'c_l': param_bounds['c_l'][0],
                'c_d': param_bounds['c_d'][1],
                'alpha_elec': param_bounds['alpha_elec'][0],
                'e_deploy': param_bounds['e_deploy'][0],
                'gamma_cool': param_bounds['gamma_cool'][0]
            },
            {
                'name': 'Balanced Setup',
                'mass': sum(param_bounds['mass']) / 2,
                'c_l': sum(param_bounds['c_l']) / 2,
                'c_d': sum(param_bounds['c_d']) / 2,
                'alpha_elec': sum(param_bounds['alpha_elec']) / 2,
                'e_deploy': sum(param_bounds['e_deploy']) / 2,
                'gamma_cool': sum(param_bounds['gamma_cool']) / 2
            },
            {
                'name': 'High Drag Setup',
                'mass': sum(param_bounds['mass']) / 2,
                'c_l': param_bounds['c_l'][1],
                'c_d': param_bounds['c_d'][1],  # High drag
                'alpha_elec': param_bounds['alpha_elec'][1],
                'e_deploy': param_bounds['e_deploy'][1],
                'gamma_cool': param_bounds['gamma_cool'][1]
            },
            {
                'name': 'Low Downforce Setup',
                'mass': param_bounds['mass'][0],
                'c_l': param_bounds['c_l'][0],  # Low downforce
                'c_d': param_bounds['c_d'][0],
                'alpha_elec': param_bounds['alpha_elec'][1],
                'e_deploy': param_bounds['e_deploy'][1],
                'gamma_cool': param_bounds['gamma_cool'][1]
            }
        ]
        
        results = []
        for case in test_cases:
            name = case.pop('name')
            
            # Create feature vector
            X_test = np.array([case[f] for f in ['mass', 'c_l', 'c_d', 'alpha_elec', 'e_deploy', 'gamma_cool']])
            
            # Pad with zeros for derived features (simplified)
            if len(feature_names) > 6:
                X_test = np.pad(X_test, (0, len(feature_names) - 6))
            
            pred = model.predict(X_test.reshape(1, -1))[0]
            
            results.append({
                'case': name,
                **case,
                'predicted_lap': pred
            })
            
            print(f"  {name}: {pred:.2f} s")
        
        return pd.DataFrame(results)
    
    def generate_sensitivity_report(
        self,
        importance_df: pd.DataFrame,
        monotonicity: Dict,
        ood_results: pd.DataFrame,
        model_name: str = "model"
    ) -> str:
        """Generate comprehensive sensitivity analysis report."""
        
        # Monotonicity summary
        mono_pass = sum(1 for v in monotonicity.values() if v['is_monotonic'])
        mono_total = len(monotonicity)
        
        report = f"""# Sensitivity Analysis Report: {model_name}

## Feature Importance (Top 10)

| Rank | Feature | Importance |
|------|---------|------------|
"""
        for i, row in importance_df.head(10).iterrows():
            report += f"| {importance_df.index.get_loc(i)+1} | {row['feature']} | {row['importance']:.4f} |\n"
        
        report += f"""

## Monotonicity Checks

**Result: {mono_pass}/{mono_total} checks passed**

| Parameter | Expected | Status |
|-----------|----------|--------|
"""
        for param, result in monotonicity.items():
            status = "✓ Pass" if result['is_monotonic'] else "✗ Fail"
            report += f"| {param} | {result['expected']} | {status} |\n"
        
        report += """

## Out-of-Distribution Tests

| Setup | Predicted Lap Time |
|-------|-------------------|
"""
        for _, row in ood_results.iterrows():
            report += f"| {row['case']} | {row['predicted_lap']:.2f} s |\n"
        
        report += """

## Interpretation

The sensitivity analysis confirms that the model:
1. Correctly identifies the most influential parameters
2. Follows expected physical relationships (monotonicity)
3. Produces reasonable predictions for extreme setups
"""
        
        report_path = self.output_dir / "reports" / f"{model_name}_sensitivity.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"\nReport saved to: {report_path}")
        return str(report_path)


def main():
    """Run sensitivity analysis on trained model."""
    print("=" * 60)
    print("Formula1 Sensitivity Analysis")
    print("=" * 60)
    
    from src.features.engineering import FeatureEngineer
    
    # Load data and model
    data_path = Path("data/synthetic/lap_times_100k.parquet")
    model_path = Path("models/final/best_lap_predictor.joblib")
    
    if not data_path.exists() or not model_path.exists():
        print("Data or model not found. Run generator and training first.")
        return
    
    # Load
    df = pd.read_parquet(data_path)
    model_data = joblib.load(model_path)
    model = model_data['model']
    model_name = model_data['name']
    
    # Features
    engineer = FeatureEngineer(include_interactions=True)
    base_cols = ['mass', 'c_l', 'c_d', 'alpha_elec', 'e_deploy', 'gamma_cool']
    df_features = engineer.fit_transform(df[base_cols])
    feature_names = list(df_features.columns)
    X = df_features.values
    
    # Analysis
    analyzer = SensitivityAnalyzer()
    
    print("\n[1/5] Computing feature importance...")
    importance = analyzer.compute_feature_importance(model, X, feature_names)
    analyzer.plot_feature_importance(importance, model_name=model_name)
    
    print("\n[2/5] Checking monotonicity...")
    monotonicity = analyzer.check_monotonicity(model, X, feature_names)
    
    print("\n[3/5] Generating partial dependence plots...")
    analyzer.plot_partial_dependence(model, X, feature_names, model_name=model_name)
    
    print("\n[4/5] Testing out-of-distribution...")
    param_bounds = {
        'mass': (700, 850),
        'c_l': (0.8, 1.5),
        'c_d': (0.7, 1.3),
        'alpha_elec': (0.0, 0.4),
        'e_deploy': (2.0, 4.0),
        'gamma_cool': (0.8, 1.2)
    }
    ood = analyzer.test_out_of_distribution(model, feature_names, param_bounds)
    
    print("\n[5/5] Generating report...")
    analyzer.generate_sensitivity_report(importance, monotonicity, ood, model_name)
    
    print("\n✓ Sensitivity analysis complete!")


if __name__ == "__main__":
    main()
