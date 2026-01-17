"""
Feature Engineering Pipeline

Transforms raw design parameters into ML-ready features.
Handles derived features, normalization, and train/test splits.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, List, Optional
import os
import sys
import joblib

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.config import (
    DESIGN_RANGES, ML_CONFIG, TARGET_NAMES,
    DATA_DIR, MODELS_DIR
)


# Base features (direct from design vector)
BASE_FEATURES = DESIGN_RANGES.get_param_names()


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived features from base design parameters.
    
    Derived features capture interactions that may be informative:
    - L/D ratio (aero efficiency)
    - Power-to-weight ratio proxy
    - Energy density proxy
    
    Args:
        df: DataFrame with base design parameters
        
    Returns:
        DataFrame with added derived features
    """
    df = df.copy()
    
    # Aero efficiency: L/D ratio (higher = more efficient)
    df['L_D_ratio'] = df['C_L'] / df['C_D']
    
    # Power-to-weight proxy (normalized)
    # Higher alpha_elec means more electric power fraction
    df['power_weight_proxy'] = df['alpha_elec'] / (df['m'] / 800)  # Normalized by max mass
    
    # Energy intensity: deployable energy normalized by mass
    df['energy_per_mass'] = df['E_deploy'] / (df['m'] / 100)  # MJ per 100kg
    
    # Cooling-adjusted energy: interaction term
    df['cooling_energy_ratio'] = df['gamma_cool'] * df['E_deploy']
    
    # Aero-mass interaction
    df['aero_mass_ratio'] = (df['C_L'] * df['C_D']) / (df['m'] / 740)
    
    return df


# All features including derived
ALL_FEATURES = BASE_FEATURES + [
    'L_D_ratio', 'power_weight_proxy', 'energy_per_mass',
    'cooling_energy_ratio', 'aero_mass_ratio'
]


class FeaturePipeline:
    """
    Complete feature engineering pipeline for training and inference.
    
    Handles:
    - Derived feature computation
    - Train/val/test splitting
    - Feature scaling (StandardScaler)
    - Saving/loading of fitted transformers
    """
    
    def __init__(
        self,
        use_derived: bool = True,
        scaler: Optional[StandardScaler] = None
    ):
        """
        Initialize feature pipeline.
        
        Args:
            use_derived: Whether to compute derived features
            scaler: Pre-fitted scaler (for inference mode)
        """
        self.use_derived = use_derived
        self.scaler = scaler if scaler else StandardScaler()
        self.is_fitted = scaler is not None
        
        self.feature_names = ALL_FEATURES if use_derived else BASE_FEATURES
    
    def transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform raw features to ML-ready features.
        
        Args:
            df: DataFrame with design parameters
            
        Returns:
            DataFrame with transformed features
        """
        if self.use_derived:
            df = compute_derived_features(df)
        
        return df[self.feature_names]
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        scale: bool = True
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Fit scaler and transform features.
        
        Args:
            df: DataFrame with design parameters
            scale: Whether to apply standard scaling
            
        Returns:
            Tuple of (transformed array, feature names)
        """
        features_df = self.transform_features(df)
        
        if scale:
            X = self.scaler.fit_transform(features_df)
            self.is_fitted = True
        else:
            X = features_df.values
        
        return X, self.feature_names
    
    def transform(
        self,
        df: pd.DataFrame,
        scale: bool = True
    ) -> np.ndarray:
        """
        Transform features using fitted scaler.
        
        Args:
            df: DataFrame with design parameters
            scale: Whether to apply standard scaling
            
        Returns:
            Transformed feature array
        """
        features_df = self.transform_features(df)
        
        if scale:
            if not self.is_fitted:
                raise ValueError("Scaler not fitted. Call fit_transform first.")
            X = self.scaler.transform(features_df)
        else:
            X = features_df.values
        
        return X
    
    def save(self, filepath: str):
        """Save fitted pipeline to disk."""
        state = {
            'scaler': self.scaler,
            'use_derived': self.use_derived,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }
        joblib.dump(state, filepath)
        print(f"Saved feature pipeline to: {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'FeaturePipeline':
        """Load fitted pipeline from disk."""
        state = joblib.load(filepath)
        pipeline = cls(
            use_derived=state['use_derived'],
            scaler=state['scaler']
        )
        pipeline.is_fitted = state['is_fitted']
        pipeline.feature_names = state['feature_names']
        return pipeline


def prepare_data(
    data_path: Optional[str] = None,
    use_derived: bool = True,
    scale: bool = True,
    random_state: int = ML_CONFIG.random_seed
) -> Dict:
    """
    Load data and prepare train/val/test splits.
    
    Args:
        data_path: Path to CSV file (default: latest in data dir)
        use_derived: Whether to use derived features
        scale: Whether to scale features
        random_state: Random seed for splitting
        
    Returns:
        Dict with X_train, X_val, X_test, y_train, y_val, y_test,
             pipeline, feature_names, target_names
    """
    # Find data file if not specified
    if data_path is None:
        csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")
        # Use most recent
        data_path = os.path.join(DATA_DIR, sorted(csv_files)[-1])
    
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"  Loaded {len(df)} samples")
    
    # Initialize pipeline
    pipeline = FeaturePipeline(use_derived=use_derived)
    
    # Extract targets
    y = df[TARGET_NAMES].values
    
    # Transform features
    X, feature_names = pipeline.fit_transform(df, scale=scale)
    
    print(f"  Features: {len(feature_names)}")
    print(f"  Targets: {len(TARGET_NAMES)}")
    
    # Split: train / (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=(ML_CONFIG.val_ratio + ML_CONFIG.test_ratio),
        random_state=random_state
    )
    
    # Split remaining: val / test
    val_fraction = ML_CONFIG.val_ratio / (ML_CONFIG.val_ratio + ML_CONFIG.test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - val_fraction),
        random_state=random_state
    )
    
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'pipeline': pipeline,
        'feature_names': feature_names,
        'target_names': TARGET_NAMES,
        'raw_df': df
    }


if __name__ == "__main__":
    print("Testing Feature Pipeline")
    print("=" * 60)
    
    # Load and prepare data
    data = prepare_data()
    
    print("\nFeature Names:")
    for i, name in enumerate(data['feature_names']):
        print(f"  {i+1}. {name}")
    
    print("\nTarget Statistics (train set):")
    for i, target in enumerate(data['target_names']):
        values = data['y_train'][:, i]
        print(f"  {target}: mean={values.mean():.3f}, std={values.std():.3f}")
