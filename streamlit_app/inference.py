"""
Inference Module for F1 Analytics Dashboard

Handles:
- Model loading from frozen joblib
- Feature engineering
- Prediction for single setups
- Input validation
- Out-of-distribution detection
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.engineering import FeatureEngineer


@dataclass
class Setup:
    """Vehicle setup parameters."""
    mass: float           # Total mass (kg) [700-850]
    c_l: float           # Aero load coefficient [0.8-1.5]
    c_d: float           # Aero drag coefficient [0.7-1.3]
    alpha_elec: float    # Electric power fraction [0.0-0.4]
    e_deploy: float      # Deployable energy (MJ) [2.0-4.0]
    gamma_cool: float    # Cooling aggressiveness [0.8-1.2]
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'mass': self.mass,
            'c_l': self.c_l,
            'c_d': self.c_d,
            'alpha_elec': self.alpha_elec,
            'e_deploy': self.e_deploy,
            'gamma_cool': self.gamma_cool
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([self.to_dict()])


# Parameter bounds for validation
PARAM_BOUNDS = {
    'mass': (700.0, 850.0),
    'c_l': (0.8, 1.5),
    'c_d': (0.7, 1.3),
    'alpha_elec': (0.0, 0.4),
    'e_deploy': (2.0, 4.0),
    'gamma_cool': (0.8, 1.2)
}

# Default setup
DEFAULT_SETUP = Setup(
    mass=780.0,
    c_l=1.2,
    c_d=1.0,
    alpha_elec=0.15,
    e_deploy=3.0,
    gamma_cool=1.0
)


class F1Predictor:
    """
    F1 lap time predictor using frozen trained models.
    """
    
    def __init__(self, model_path: str = "models/final/best_lap_predictor.joblib"):
        """
        Initialize predictor with frozen model.
        
        Args:
            model_path: Path to saved model joblib
        """
        self.model_path = Path(model_path)
        self.model = None
        self.model_name = None
        self.scaler = None
        self.val_rmse = None
        self.feature_engineer = FeatureEngineer(include_interactions=True)
        self.feature_names = None
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load model from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        data = joblib.load(self.model_path)
        self.model = data['model']
        self.model_name = data['name']
        self.scaler = data.get('scaler')
        self.val_rmse = data.get('val_rmse', 0.55)
    
    def validate_setup(self, setup: Setup) -> Tuple[bool, list]:
        """
        Validate setup parameters are within bounds.
        
        Returns:
            Tuple of (is_valid, list of warnings)
        """
        warnings = []
        params = setup.to_dict()
        
        for param, value in params.items():
            low, high = PARAM_BOUNDS[param]
            if value < low or value > high:
                warnings.append(f"{param}={value:.2f} out of range [{low}, {high}]")
        
        is_valid = len(warnings) == 0
        return is_valid, warnings
    
    def is_out_of_distribution(self, setup: Setup, threshold: float = 0.2) -> Tuple[bool, float]:
        """
        Check if setup is significantly outside training distribution.
        
        Args:
            setup: Vehicle setup
            threshold: Fraction outside bounds to flag as OOD
            
        Returns:
            Tuple of (is_ood, severity_score)
        """
        params = setup.to_dict()
        severity = 0.0
        
        for param, value in params.items():
            low, high = PARAM_BOUNDS[param]
            range_size = high - low
            
            if value < low:
                severity += abs(value - low) / range_size
            elif value > high:
                severity += abs(value - high) / range_size
        
        return severity > threshold, severity
    
    def predict(self, setup: Setup) -> Dict[str, Any]:
        """
        Predict lap time for a single setup.
        
        Args:
            setup: Vehicle setup parameters
            
        Returns:
            Dict with predictions and metadata
        """
        # Convert to DataFrame
        df = setup.to_dataframe()
        
        # Engineer features
        df_features = self.feature_engineer.fit_transform(df)
        self.feature_names = list(df_features.columns)
        
        X = df_features.values
        
        # Predict
        lap_time = self.model.predict(X)[0]
        
        # Estimate sectors (approximate split based on Spa characteristics)
        # Sector 1: ~35%, Sector 2: ~40%, Sector 3: ~25%
        sector_1 = lap_time * 0.35
        sector_2 = lap_time * 0.40
        sector_3 = lap_time * 0.25
        
        # Estimate 20 segments (proportional distribution)
        segment_weights = [
            0.022, 0.051, 0.029, 0.026, 0.117,  # S1 segments 1-5
            0.017, 0.015, 0.026, 0.020, 0.058,  # S1-S2 segments 6-10
            0.044, 0.029, 0.051, 0.022, 0.041,  # S2 segments 11-15
            0.066, 0.051, 0.044, 0.029, 0.045   # S3 segments 16-20
        ]
        segment_times = [lap_time * w for w in segment_weights]
        
        # Validation
        is_valid, warnings = self.validate_setup(setup)
        is_ood, ood_severity = self.is_out_of_distribution(setup)
        
        return {
            'lap_time': lap_time,
            'sector_1': sector_1,
            'sector_2': sector_2,
            'sector_3': sector_3,
            'segment_times': segment_times,
            'is_valid': is_valid,
            'warnings': warnings,
            'is_ood': is_ood,
            'ood_severity': ood_severity,
            'model_name': self.model_name,
            'uncertainty': self.val_rmse
        }
    
    def predict_batch(self, setups: list) -> list:
        """Predict for multiple setups."""
        return [self.predict(s) for s in setups]
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Extract feature importance from model."""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'estimators_'):
            importances = np.mean([
                getattr(est, 'feature_importances_', np.zeros(len(self.feature_names)))
                for est in self.model.estimators_
            ], axis=0)
        else:
            importances = np.zeros(len(self.feature_names))
        
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        })
        return df.sort_values('importance', ascending=False)
    
    def compute_partial_dependence(
        self,
        param_name: str,
        base_setup: Setup,
        n_points: int = 20
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute partial dependence for a single parameter.
        
        Args:
            param_name: Parameter to vary
            base_setup: Base setup to modify
            n_points: Number of evaluation points
            
        Returns:
            Tuple of (parameter_values, predicted_lap_times)
        """
        low, high = PARAM_BOUNDS[param_name]
        param_values = np.linspace(low, high, n_points)
        lap_times = []
        
        for val in param_values:
            modified_setup = Setup(**{**base_setup.to_dict(), param_name: val})
            result = self.predict(modified_setup)
            lap_times.append(result['lap_time'])
        
        return param_values, np.array(lap_times)


def get_predictor() -> F1Predictor:
    """Factory function to get predictor instance."""
    return F1Predictor()
