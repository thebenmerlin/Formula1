"""
Design Comparison and Trade-off Analysis

Tools for comparing multiple vehicle designs and analyzing
Pareto-optimal trade-offs between lap time, energy, and thermal risk.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import ConvexHull
from typing import Dict, List, Optional, Tuple
import os
import sys
import joblib

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.config import (
    MODELS_DIR, OUTPUTS_DIR, TARGET_NAMES,
    DESIGN_RANGES, REFERENCE
)
from src.data.physics_model import PhysicsModel
from src.models.features import FeaturePipeline


def is_pareto_efficient(costs: np.ndarray) -> np.ndarray:
    """
    Find Pareto-efficient points (non-dominated solutions).
    
    A point is Pareto-efficient if no other point is better in ALL objectives.
    
    Args:
        costs: Array of shape (n_points, n_objectives)
               Lower values are better (minimization)
               
    Returns:
        Boolean array indicating Pareto-efficient points
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    
    for i, c in enumerate(costs):
        if is_efficient[i]:
            # Check if any other point dominates this one
            is_efficient[is_efficient] = np.any(
                costs[is_efficient] < c, axis=1
            ) | np.all(costs[is_efficient] <= c, axis=1)
            is_efficient[i] = True  # Keep self
    
    return is_efficient


class DesignComparator:
    """
    Compares multiple vehicle designs on performance metrics.
    """
    
    def __init__(self, model: object = None, pipeline: FeaturePipeline = None):
        """
        Initialize comparator.
        
        Args:
            model: Trained ML model (optional, uses physics if not provided)
            pipeline: Feature pipeline (required if model provided)
        """
        self.model = model
        self.pipeline = pipeline
    
    @classmethod
    def load(cls, model_name: str = 'random_forest') -> 'DesignComparator':
        """Load comparator with trained model."""
        model = joblib.load(os.path.join(MODELS_DIR, f'{model_name}_model.joblib'))
        pipeline = FeaturePipeline.load(os.path.join(MODELS_DIR, 'feature_pipeline.joblib'))
        return cls(model=model, pipeline=pipeline)
    
    def evaluate_design(self, design: Dict[str, float]) -> Dict[str, float]:
        """
        Evaluate a single design using physics model.
        
        Args:
            design: Design parameter dict
            
        Returns:
            Dict with lap_time, energy_used, thermal_risk
        """
        physics = PhysicsModel(design)
        return physics.get_outputs()
    
    def evaluate_designs(
        self,
        designs: List[Dict[str, float]],
        names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Evaluate multiple designs.
        
        Args:
            designs: List of design parameter dicts
            names: Optional list of design names
            
        Returns:
            DataFrame with designs and their metrics
        """
        if names is None:
            names = [f'Design_{i}' for i in range(len(designs))]
        
        results = []
        for name, design in zip(names, designs):
            metrics = self.evaluate_design(design)
            result = {'name': name}
            result.update(design)
            result.update(metrics)
            results.append(result)
        
        return pd.DataFrame(results)
    
    def find_pareto_frontier(
        self,
        designs: List[Dict[str, float]],
        names: Optional[List[str]] = None,
        objectives: List[str] = ['lap_time', 'energy_used', 'thermal_risk']
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Find Pareto-optimal designs.
        
        Args:
            designs: List of design dicts
            names: Optional design names
            objectives: Objectives to minimize
            
        Returns:
            Tuple of (all designs DataFrame, Pareto-optimal designs DataFrame)
        """
        # Evaluate all designs
        df = self.evaluate_designs(designs, names)
        
        # Extract objective values
        costs = df[objectives].values
        
        # Find Pareto-efficient points
        is_pareto = is_pareto_efficient(costs)
        
        df['is_pareto'] = is_pareto
        pareto_df = df[df['is_pareto']].copy()
        
        return df, pareto_df
    
    def grid_search(
        self,
        n_per_dim: int = 5,
        fixed_params: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Exhaustive grid search over design space.
        
        Args:
            n_per_dim: Number of points per dimension
            fixed_params: Parameters to fix (not varied)
            
        Returns:
            DataFrame with all evaluated designs
        """
        bounds = DESIGN_RANGES.get_bounds()
        param_names = DESIGN_RANGES.get_param_names()
        
        # Determine which params to vary
        if fixed_params:
            vary_params = [p for p in param_names if p not in fixed_params]
        else:
            vary_params = param_names
            fixed_params = {}
        
        # Create grids
        grids = []
        for param in vary_params:
            low, high = bounds[param]
            grids.append(np.linspace(low, high, n_per_dim))
        
        # Generate all combinations
        mesh = np.meshgrid(*grids, indexing='ij')
        combinations = np.column_stack([m.ravel() for m in mesh])
        
        print(f"Evaluating {len(combinations)} design combinations...")
        
        # Evaluate each
        designs = []
        for combo in combinations:
            design = fixed_params.copy()
            for i, param in enumerate(vary_params):
                design[param] = combo[i]
            designs.append(design)
        
        return self.evaluate_designs(designs)
    
    def plot_tradeoffs(
        self,
        df: pd.DataFrame,
        output_dir: str = OUTPUTS_DIR,
        highlight_pareto: bool = True
    ):
        """
        Create trade-off visualization plots.
        
        Args:
            df: DataFrame with evaluated designs
            output_dir: Directory for saving plots
            highlight_pareto: Whether to highlight Pareto-optimal points
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 2D trade-off plots
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        pairs = [
            ('lap_time', 'energy_used'),
            ('lap_time', 'thermal_risk'),
            ('energy_used', 'thermal_risk')
        ]
        
        for ax, (x_col, y_col) in zip(axes, pairs):
            if highlight_pareto and 'is_pareto' in df.columns:
                # Non-Pareto points
                non_pareto = df[~df['is_pareto']]
                ax.scatter(non_pareto[x_col], non_pareto[y_col], 
                          alpha=0.3, s=20, c='gray', label='Dominated')
                
                # Pareto points
                pareto = df[df['is_pareto']]
                ax.scatter(pareto[x_col], pareto[y_col], 
                          s=80, c='red', edgecolors='black', label='Pareto-optimal')
            else:
                ax.scatter(df[x_col], df[y_col], alpha=0.5, s=30)
            
            ax.set_xlabel(x_col.replace('_', ' ').title())
            ax.set_ylabel(y_col.replace('_', ' ').title())
            ax.grid(True, alpha=0.3)
            
            if highlight_pareto and 'is_pareto' in df.columns:
                ax.legend()
        
        plt.suptitle('Design Trade-offs', fontsize=14)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, 'tradeoff_analysis.png'), dpi=150)
        plt.close()
        
        # 3D trade-off plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        colors = df['lap_time']
        scatter = ax.scatter(
            df['energy_used'], 
            df['thermal_risk'], 
            df['lap_time'],
            c=colors, cmap='viridis', s=30, alpha=0.6
        )
        
        ax.set_xlabel('Energy Used (MJ)')
        ax.set_ylabel('Thermal Risk')
        ax.set_zlabel('Lap Time (s)')
        ax.set_title('3D Trade-off Space')
        
        plt.colorbar(scatter, label='Lap Time (s)', shrink=0.6)
        fig.savefig(os.path.join(output_dir, 'tradeoff_3d.png'), dpi=150)
        plt.close()
        
        print(f"Trade-off plots saved to: {output_dir}")


def run_comparison_analysis(output_dir: str = OUTPUTS_DIR):
    """
    Run full comparison analysis with example designs.
    """
    print("=" * 60)
    print("Design Comparison Analysis")
    print("=" * 60)
    
    comparator = DesignComparator()
    
    # Example designs
    designs = [
        {'m': 770, 'C_L': 1.0, 'C_D': 1.0, 'alpha_elec': 0.47, 'E_deploy': 7.0, 'gamma_cool': 1.0},
        {'m': 755, 'C_L': 0.85, 'C_D': 0.88, 'alpha_elec': 0.52, 'E_deploy': 8.0, 'gamma_cool': 0.9},
        {'m': 785, 'C_L': 1.28, 'C_D': 1.18, 'alpha_elec': 0.42, 'E_deploy': 6.0, 'gamma_cool': 1.15},
        {'m': 745, 'C_L': 1.15, 'C_D': 1.05, 'alpha_elec': 0.58, 'E_deploy': 8.8, 'gamma_cool': 1.25},
        {'m': 780, 'C_L': 0.95, 'C_D': 0.95, 'alpha_elec': 0.40, 'E_deploy': 5.5, 'gamma_cool': 1.3},
    ]
    names = ['Baseline', 'Low Drag', 'High DF', 'Aggressive', 'Conservative']
    
    # Evaluate
    print("\nEvaluating example designs...")
    df = comparator.evaluate_designs(designs, names)
    
    print("\nResults:")
    print(df[['name', 'lap_time', 'energy_used', 'thermal_risk']].to_string(index=False))
    
    # Find Pareto frontier
    print("\nFinding Pareto-optimal designs...")
    all_df, pareto_df = comparator.find_pareto_frontier(designs, names)
    
    print(f"\nPareto-optimal designs ({len(pareto_df)}):")
    print(pareto_df[['name', 'lap_time', 'energy_used', 'thermal_risk']].to_string(index=False))
    
    # Plot
    comparator.plot_tradeoffs(all_df, output_dir)
    
    # Save results
    results_path = os.path.join(output_dir, 'comparison_results.csv')
    all_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
    
    return all_df, pareto_df


if __name__ == "__main__":
    run_comparison_analysis()
