"""
Central configuration for F1 ML Design System.

All physics constants, design parameter ranges, and model hyperparameters
are defined here for reproducibility and easy modification.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple

# =============================================================================
# PHYSICS CONSTANTS
# =============================================================================

@dataclass(frozen=True)
class PhysicsConstants:
    """Immutable physics constants used throughout the system."""
    
    # Environmental
    g: float = 9.81                    # Gravitational acceleration (m/s²)
    rho_air: float = 1.225             # Air density at sea level (kg/m³)
    
    # Tyre and grip
    mu_tyre: float = 1.5               # Tyre friction coefficient (dry F1)
    mu_braking: float = 1.8            # Peak braking friction (with aero)
    
    # Powertrain (2026 regulations inspired)
    P_total_max: float = 746e3         # Max total power: ~1000 hp (W)
    P_ice_max: float = 485e3           # ICE max power (W)
    P_mgu_k_max: float = 350e3         # MGU-K max power (W)
    P_mgu_h_max: float = 350e3         # MGU-H max power (W)
    
    # Energy recovery
    eta_regen: float = 0.90            # Regeneration efficiency
    eta_deploy: float = 0.95           # Deployment efficiency
    
    # Thermal
    T_ambient: float = 25.0            # Ambient temperature (°C)
    T_battery_max: float = 60.0        # Max safe battery temp (°C)
    thermal_capacity: float = 50e3     # Battery thermal capacity (J/°C)

PHYSICS = PhysicsConstants()

# =============================================================================
# DESIGN PARAMETER RANGES
# =============================================================================

@dataclass
class DesignParameterRanges:
    """Valid ranges for all design parameters."""
    
    # Mass (kg)
    m_min: float = 740.0
    m_max: float = 800.0
    
    # Aero load coefficient (normalized)
    C_L_min: float = 0.8
    C_L_max: float = 1.3
    
    # Aero drag coefficient (normalized)
    C_D_min: float = 0.85
    C_D_max: float = 1.2
    
    # Electric power fraction
    alpha_elec_min: float = 0.35
    alpha_elec_max: float = 0.60
    
    # Max deployable energy per lap (MJ)
    E_deploy_min: float = 5.0
    E_deploy_max: float = 9.0
    
    # Cooling aggressiveness factor
    gamma_cool_min: float = 0.7
    gamma_cool_max: float = 1.3
    
    def get_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Return bounds as dictionary for sampling."""
        return {
            'm': (self.m_min, self.m_max),
            'C_L': (self.C_L_min, self.C_L_max),
            'C_D': (self.C_D_min, self.C_D_max),
            'alpha_elec': (self.alpha_elec_min, self.alpha_elec_max),
            'E_deploy': (self.E_deploy_min, self.E_deploy_max),
            'gamma_cool': (self.gamma_cool_min, self.gamma_cool_max),
        }
    
    def get_param_names(self) -> list:
        """Return ordered list of parameter names."""
        return ['m', 'C_L', 'C_D', 'alpha_elec', 'E_deploy', 'gamma_cool']

DESIGN_RANGES = DesignParameterRanges()

# =============================================================================
# REFERENCE VEHICLE (Baseline for normalization)
# =============================================================================

@dataclass
class ReferenceVehicle:
    """Reference/baseline vehicle for normalization and comparison."""
    
    m: float = 770.0           # Mass (kg)
    C_L: float = 1.0           # Baseline aero load
    C_D: float = 1.0           # Baseline aero drag
    alpha_elec: float = 0.47   # Electric fraction
    E_deploy: float = 7.0      # Energy deploy (MJ)
    gamma_cool: float = 1.0    # Cooling factor
    
    # Reference aero values (for denormalization)
    A_ref: float = 1.5         # Reference frontal area (m²)
    C_L_ref: float = 3.5       # Reference downforce coefficient
    C_D_ref: float = 1.0       # Reference drag coefficient

REFERENCE = ReferenceVehicle()

# =============================================================================
# ML MODEL CONFIGURATION
# =============================================================================

@dataclass
class MLConfig:
    """Configuration for ML model training."""
    
    # Data generation
    n_samples: int = 10000
    noise_fraction: float = 0.03    # 3% Gaussian noise
    random_seed: int = 42
    
    # Train/val/test split
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Gradient Boosting hyperparameters
    gb_n_estimators: int = 200
    gb_max_depth: int = 6
    gb_learning_rate: float = 0.1
    gb_min_samples_leaf: int = 10
    
    # MLP hyperparameters
    mlp_hidden_layers: Tuple[int, ...] = (64, 64)
    mlp_max_iter: int = 500
    mlp_alpha: float = 0.001
    
    # Ridge regression
    ridge_alpha: float = 1.0

ML_CONFIG = MLConfig()

# =============================================================================
# OUTPUT TARGETS
# =============================================================================

TARGET_NAMES = ['lap_time', 'energy_used', 'thermal_risk']
TARGET_UNITS = {
    'lap_time': 's',
    'energy_used': 'MJ',
    'thermal_risk': ''
}

# =============================================================================
# FILE PATHS
# =============================================================================

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, 'outputs')

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)
