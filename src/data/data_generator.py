"""
Synthetic Data Generator

Generates physics-informed training data for ML models by:
1. Sampling design parameter vectors using Latin Hypercube Sampling
2. Running physics model on each design
3. Injecting noise to simulate unmodeled effects
4. Saving to CSV for ML training

This separates the physics model from the ML training pipeline,
ensuring clean data flow and reproducibility.
"""

import numpy as np
import pandas as pd
from scipy.stats import qmc
from typing import Dict, List, Tuple, Optional
import os
import sys
import argparse
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.config import (
    DESIGN_RANGES, ML_CONFIG, DATA_DIR,
    TARGET_NAMES, PHYSICS
)
from src.data.physics_model import PhysicsModel


class DesignSampler:
    """
    Generates design parameter samples using Latin Hypercube Sampling.
    
    LHS ensures better coverage of the design space compared to random sampling,
    which is important for training accurate surrogate models.
    """
    
    def __init__(self, seed: int = ML_CONFIG.random_seed):
        """
        Initialize sampler with random seed.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self.bounds = DESIGN_RANGES.get_bounds()
        self.param_names = DESIGN_RANGES.get_param_names()
        self.n_params = len(self.param_names)
    
    def sample_lhs(self, n_samples: int) -> pd.DataFrame:
        """
        Generate samples using Latin Hypercube Sampling.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            DataFrame with design parameter samples
        """
        # Create LHS sampler
        sampler = qmc.LatinHypercube(d=self.n_params, seed=self.seed)
        
        # Generate samples in [0, 1]^d
        unit_samples = sampler.random(n=n_samples)
        
        # Scale to actual parameter bounds
        l_bounds = [self.bounds[p][0] for p in self.param_names]
        u_bounds = [self.bounds[p][1] for p in self.param_names]
        
        scaled_samples = qmc.scale(unit_samples, l_bounds, u_bounds)
        
        # Create DataFrame
        df = pd.DataFrame(scaled_samples, columns=self.param_names)
        
        return df
    
    def sample_random(self, n_samples: int) -> pd.DataFrame:
        """
        Generate samples using uniform random sampling.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            DataFrame with design parameter samples
        """
        np.random.seed(self.seed)
        
        data = {}
        for param in self.param_names:
            low, high = self.bounds[param]
            data[param] = np.random.uniform(low, high, n_samples)
        
        return pd.DataFrame(data)
    
    def sample_grid(self, n_per_dim: int = 5) -> pd.DataFrame:
        """
        Generate samples on a regular grid (for sensitivity analysis).
        
        Args:
            n_per_dim: Number of points per dimension
            
        Returns:
            DataFrame with design parameter samples
        """
        grids = []
        for param in self.param_names:
            low, high = self.bounds[param]
            grids.append(np.linspace(low, high, n_per_dim))
        
        # Create meshgrid
        mesh = np.meshgrid(*grids, indexing='ij')
        
        # Flatten to samples
        samples = np.column_stack([m.ravel() for m in mesh])
        
        return pd.DataFrame(samples, columns=self.param_names)


class SyntheticDataGenerator:
    """
    Generates complete training dataset with physics calculations and noise.
    """
    
    def __init__(
        self,
        n_samples: int = ML_CONFIG.n_samples,
        noise_fraction: float = ML_CONFIG.noise_fraction,
        seed: int = ML_CONFIG.random_seed
    ):
        """
        Initialize data generator.
        
        Args:
            n_samples: Number of samples to generate
            noise_fraction: Fraction of output value to add as noise (e.g., 0.03 = 3%)
            seed: Random seed for reproducibility
        """
        self.n_samples = n_samples
        self.noise_fraction = noise_fraction
        self.seed = seed
        self.sampler = DesignSampler(seed=seed)
    
    def _add_noise(self, values: np.ndarray, fraction: float) -> np.ndarray:
        """
        Add Gaussian noise to output values.
        
        Noise magnitude is proportional to the value itself.
        
        Args:
            values: Array of values
            fraction: Noise fraction (e.g., 0.03 for 3%)
            
        Returns:
            Values with added noise
        """
        np.random.seed(self.seed + 1)  # Different seed for noise
        noise = np.random.normal(0, fraction, len(values)) * values
        return values + noise
    
    def generate(
        self,
        method: str = 'lhs',
        add_noise: bool = True,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Generate complete synthetic dataset.
        
        Args:
            method: Sampling method ('lhs', 'random', 'grid')
            add_noise: Whether to add noise to outputs
            verbose: Print progress information
            
        Returns:
            DataFrame with design parameters and outputs
        """
        if verbose:
            print(f"Generating {self.n_samples} synthetic samples...")
            print(f"  Method: {method}")
            print(f"  Noise: {self.noise_fraction * 100:.1f}%")
        
        # Sample design parameters
        if method == 'lhs':
            designs = self.sampler.sample_lhs(self.n_samples)
        elif method == 'random':
            designs = self.sampler.sample_random(self.n_samples)
        elif method == 'grid':
            designs = self.sampler.sample_grid()
            self.n_samples = len(designs)
        else:
            raise ValueError(f"Unknown sampling method: {method}")
        
        # Compute outputs for each design
        outputs = {target: [] for target in TARGET_NAMES}
        
        for i in range(len(designs)):
            if verbose and (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{len(designs)} samples...")
            
            # Get design vector
            design_vec = designs.iloc[i].to_dict()
            
            # Run physics model
            model = PhysicsModel(design_vec)
            results = model.get_outputs()
            
            # Store outputs
            for target in TARGET_NAMES:
                outputs[target].append(results[target])
        
        # Convert to arrays
        for target in TARGET_NAMES:
            outputs[target] = np.array(outputs[target])
        
        # Add noise if requested
        if add_noise:
            if verbose:
                print(f"  Adding {self.noise_fraction * 100:.1f}% Gaussian noise...")
            
            # Different noise levels for different outputs
            outputs['lap_time'] = self._add_noise(outputs['lap_time'], self.noise_fraction)
            outputs['energy_used'] = self._add_noise(outputs['energy_used'], self.noise_fraction)
            
            # Thermal risk needs to stay in [0, 1]
            np.random.seed(self.seed + 2)
            thermal_noise = np.random.normal(0, self.noise_fraction * 0.5, self.n_samples)
            outputs['thermal_risk'] = np.clip(
                outputs['thermal_risk'] + thermal_noise * outputs['thermal_risk'],
                0, 1
            )
        
        # Combine into single DataFrame
        df = designs.copy()
        for target in TARGET_NAMES:
            df[target] = outputs[target]
        
        if verbose:
            print(f"  Generated {len(df)} samples")
            print(f"\nOutput Statistics:")
            for target in TARGET_NAMES:
                print(f"  {target}: mean={df[target].mean():.3f}, "
                      f"std={df[target].std():.3f}, "
                      f"range=[{df[target].min():.3f}, {df[target].max():.3f}]")
        
        return df
    
    def generate_and_save(
        self,
        filename: Optional[str] = None,
        method: str = 'lhs',
        add_noise: bool = True
    ) -> str:
        """
        Generate dataset and save to CSV.
        
        Args:
            filename: Output filename (default: auto-generated with timestamp)
            method: Sampling method
            add_noise: Whether to add noise
            
        Returns:
            Path to saved file
        """
        # Generate data
        df = self.generate(method=method, add_noise=add_noise)
        
        # Create filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"synthetic_data_{self.n_samples}_{timestamp}.csv"
        
        # Ensure data directory exists
        os.makedirs(DATA_DIR, exist_ok=True)
        
        # Save to CSV
        filepath = os.path.join(DATA_DIR, filename)
        df.to_csv(filepath, index=False)
        
        print(f"\nSaved dataset to: {filepath}")
        
        return filepath


def generate_standard_dataset() -> str:
    """
    Generate the standard training dataset with default parameters.
    
    Returns:
        Path to saved dataset
    """
    generator = SyntheticDataGenerator(
        n_samples=ML_CONFIG.n_samples,
        noise_fraction=ML_CONFIG.noise_fraction,
        seed=ML_CONFIG.random_seed
    )
    
    return generator.generate_and_save(
        filename="training_data.csv",
        method='lhs',
        add_noise=True
    )


def generate_test_dataset(n_samples: int = 500) -> str:
    """
    Generate a smaller test dataset with different seed.
    
    Args:
        n_samples: Number of test samples
        
    Returns:
        Path to saved dataset
    """
    generator = SyntheticDataGenerator(
        n_samples=n_samples,
        noise_fraction=ML_CONFIG.noise_fraction,
        seed=ML_CONFIG.random_seed + 100  # Different seed
    )
    
    return generator.generate_and_save(
        filename="test_data.csv",
        method='lhs',
        add_noise=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic F1 training data")
    parser.add_argument(
        '--samples', '-n', type=int, default=ML_CONFIG.n_samples,
        help=f"Number of samples to generate (default: {ML_CONFIG.n_samples})"
    )
    parser.add_argument(
        '--noise', type=float, default=ML_CONFIG.noise_fraction,
        help=f"Noise fraction (default: {ML_CONFIG.noise_fraction})"
    )
    parser.add_argument(
        '--method', choices=['lhs', 'random', 'grid'], default='lhs',
        help="Sampling method (default: lhs)"
    )
    parser.add_argument(
        '--output', '-o', type=str, default=None,
        help="Output filename (default: auto-generated)"
    )
    parser.add_argument(
        '--test', action='store_true',
        help="Generate small test dataset (500 samples)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("F1 ML Design System - Synthetic Data Generator")
    print("=" * 60)
    
    if args.test:
        print("\nGenerating test dataset (500 samples)...")
        generate_test_dataset()
    else:
        generator = SyntheticDataGenerator(
            n_samples=args.samples,
            noise_fraction=args.noise,
            seed=ML_CONFIG.random_seed
        )
        generator.generate_and_save(
            filename=args.output,
            method=args.method,
            add_noise=True
        )
    
    print("\nDone!")
