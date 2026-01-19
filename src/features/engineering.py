"""
Feature Engineering Module for F1 Track Time Prediction

Derives physically meaningful features from the 6-parameter design vector:
- Power-to-weight ratio
- Aero efficiency ratio (C_L / C_D)
- Electric energy density per lap
- Cooling-adjusted power availability
- Segment-type-weighted features

All feature creation follows engineering logic, not blind expansion.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional


class FeatureEngineer:
    """
    Creates physics-informed derived features from the design parameter vector.
    
    Base features (6):
        m, C_L, C_D, alpha_elec, E_deploy, gamma_cool
        
    Derived features (engineering logic):
        - Power-to-weight ratio
        - Aero efficiency
        - Energy density metrics
        - Composite performance indicators
    """
    
    # Feature names for reference
    BASE_FEATURES = ['mass', 'c_l', 'c_d', 'alpha_elec', 'e_deploy', 'gamma_cool']
    
    # Baseline values for normalization
    BASELINE = {
        'ice_power_kw': 560,
        'elec_power_kw': 120,
        'ref_mass': 750,
        'track_length_km': 7.004
    }
    
    def __init__(self, include_interactions: bool = True, include_polynomials: bool = False):
        """
        Initialize feature engineer.
        
        Args:
            include_interactions: Include interaction features (ratios, products)
            include_polynomials: Include polynomial features (squared terms)
        """
        self.include_interactions = include_interactions
        self.include_polynomials = include_polynomials
        self.feature_names_: Optional[List[str]] = None
        
    def fit(self, X: pd.DataFrame) -> 'FeatureEngineer':
        """Fit the feature engineer (learns nothing, just validates)."""
        # Validate all base features present
        missing = set(self.BASE_FEATURES) - set(X.columns)
        if missing:
            raise ValueError(f"Missing base features: {missing}")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform base features into engineered feature set.
        
        Args:
            X: DataFrame with base features
            
        Returns:
            DataFrame with base + derived features
        """
        df = X.copy()
        
        # ==============================================
        # 1. POWER METRICS
        # ==============================================
        
        # Total available power (kW)
        df['total_power_kw'] = (
            self.BASELINE['ice_power_kw'] * df['gamma_cool'] + 
            self.BASELINE['elec_power_kw'] * df['alpha_elec']
        )
        
        # Power-to-weight ratio (kW/kg) - KEY METRIC
        df['power_to_weight'] = df['total_power_kw'] / df['mass']
        
        # ICE power adjusted for cooling
        df['ice_power_adj'] = self.BASELINE['ice_power_kw'] * df['gamma_cool']
        
        # ==============================================
        # 2. AERO METRICS
        # ==============================================
        
        # Aero efficiency ratio (lift/drag) - KEY METRIC
        df['aero_efficiency'] = df['c_l'] / df['c_d']
        
        # Net aero effect (downforce benefit minus drag penalty)
        # Higher C_L is good, higher C_D is bad
        df['aero_net'] = df['c_l'] - 0.5 * df['c_d']
        
        # Aero balance indicator
        df['aero_balance'] = df['c_l'] * (1 - df['c_d'])
        
        # ==============================================
        # 3. ENERGY METRICS
        # ==============================================
        
        # Energy per km (MJ/km)
        df['energy_per_km'] = df['e_deploy'] / self.BASELINE['track_length_km']
        
        # Energy deployment rate (kJ/segment, 20 segments)
        df['energy_per_segment_kj'] = (df['e_deploy'] * 1000) / 20
        
        # Electric contribution potential
        df['electric_contribution'] = df['alpha_elec'] * df['e_deploy']
        
        # ==============================================
        # 4. COMPOSITE PERFORMANCE METRICS
        # ==============================================
        
        # Straight-line performance proxy
        # High power, low drag = fast straights
        df['straight_performance'] = df['power_to_weight'] / df['c_d']
        
        # Corner performance proxy
        # High downforce, low mass = fast corners
        df['corner_performance'] = df['c_l'] / (df['mass'] / self.BASELINE['ref_mass'])
        
        # Overall balance (neither straight nor corner dominant)
        df['performance_balance'] = (
            df['straight_performance'] / df['straight_performance'].mean() +
            df['corner_performance'] / df['corner_performance'].mean()
        ) / 2
        
        # Mass efficiency (lighter is better, normalized)
        df['mass_efficiency'] = self.BASELINE['ref_mass'] / df['mass']
        
        # Thermal efficiency (cooling vs power tradeoff)
        df['thermal_efficiency'] = df['gamma_cool'] * df['power_to_weight']
        
        # ==============================================
        # 5. INTERACTION FEATURES (if enabled)
        # ==============================================
        
        if self.include_interactions:
            # Mass-aero interaction
            df['mass_x_cl'] = df['mass'] * df['c_l']
            df['mass_x_cd'] = df['mass'] * df['c_d']
            
            # Power-aero interaction
            df['power_x_cd'] = df['total_power_kw'] * df['c_d']
            
            # Energy-power interaction
            df['energy_x_alpha'] = df['e_deploy'] * df['alpha_elec']
            
            # Cooling-power interaction
            df['cooling_x_power'] = df['gamma_cool'] * df['total_power_kw']
        
        # ==============================================
        # 6. POLYNOMIAL FEATURES (if enabled)
        # ==============================================
        
        if self.include_polynomials:
            for feat in self.BASE_FEATURES:
                df[f'{feat}_sq'] = df[feat] ** 2
        
        # Store feature names
        self.feature_names_ = list(df.columns)
        
        return df
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names after transformation."""
        if self.feature_names_ is None:
            raise ValueError("Must call fit_transform first")
        return self.feature_names_
    
    def get_derived_feature_names(self) -> List[str]:
        """Get only the derived (non-base) feature names."""
        if self.feature_names_ is None:
            raise ValueError("Must call fit_transform first")
        return [f for f in self.feature_names_ if f not in self.BASE_FEATURES]


def create_features(
    df: pd.DataFrame,
    include_interactions: bool = True,
    include_polynomials: bool = False
) -> Tuple[pd.DataFrame, FeatureEngineer]:
    """
    Convenience function to create engineered features.
    
    Args:
        df: DataFrame with base features
        include_interactions: Include interaction terms
        include_polynomials: Include squared terms
        
    Returns:
        Tuple of (transformed DataFrame, fitted FeatureEngineer)
    """
    engineer = FeatureEngineer(
        include_interactions=include_interactions,
        include_polynomials=include_polynomials
    )
    df_transformed = engineer.fit_transform(df)
    return df_transformed, engineer


def main():
    """Demo feature engineering on sample data."""
    print("=" * 60)
    print("Feature Engineering Demo")
    print("=" * 60)
    
    # Create sample data
    np.random.seed(42)
    n = 5
    sample_data = pd.DataFrame({
        'mass': np.random.uniform(700, 850, n),
        'c_l': np.random.uniform(0.8, 1.5, n),
        'c_d': np.random.uniform(0.7, 1.3, n),
        'alpha_elec': np.random.uniform(0.0, 0.4, n),
        'e_deploy': np.random.uniform(2.0, 4.0, n),
        'gamma_cool': np.random.uniform(0.8, 1.2, n)
    })
    
    print("\nBase features (6):")
    print(sample_data)
    
    # Engineer features
    engineer = FeatureEngineer(include_interactions=True, include_polynomials=False)
    df_eng = engineer.fit_transform(sample_data)
    
    print(f"\nTotal features after engineering: {len(df_eng.columns)}")
    print(f"Derived features: {len(engineer.get_derived_feature_names())}")
    
    print("\nDerived feature names:")
    for i, name in enumerate(engineer.get_derived_feature_names(), 1):
        print(f"  {i}. {name}")
    
    print("\nSample of engineered features:")
    print(df_eng[['power_to_weight', 'aero_efficiency', 'straight_performance', 'corner_performance']].round(4))


if __name__ == "__main__":
    main()
