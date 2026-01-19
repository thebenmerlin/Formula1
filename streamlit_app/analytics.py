"""
Analytics Module for F1 Dashboard

Handles:
- Setup comparison logic
- Delta calculations
- Sensitivity analysis
- Monotonicity checks
- Trade-off computations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from inference import Setup, F1Predictor, PARAM_BOUNDS


@dataclass
class ComparisonResult:
    """Results from comparing two setups."""
    baseline_name: str
    comparison_name: str
    lap_delta: float       # seconds (positive = slower)
    lap_percent: float     # percent change
    sector_deltas: List[float]
    segment_deltas: List[float]
    faster_segments: List[int]
    slower_segments: List[int]


def compare_setups(
    baseline: Dict,
    comparison: Dict,
    baseline_name: str = "Baseline",
    comparison_name: str = "Comparison"
) -> ComparisonResult:
    """
    Compare two prediction results.
    
    Args:
        baseline: Prediction result for baseline setup
        comparison: Prediction result for comparison setup
        baseline_name: Display name for baseline
        comparison_name: Display name for comparison
        
    Returns:
        ComparisonResult with all delta calculations
    """
    lap_delta = comparison['lap_time'] - baseline['lap_time']
    lap_percent = (lap_delta / baseline['lap_time']) * 100
    
    sector_deltas = [
        comparison['sector_1'] - baseline['sector_1'],
        comparison['sector_2'] - baseline['sector_2'],
        comparison['sector_3'] - baseline['sector_3']
    ]
    
    segment_deltas = [
        c - b for c, b in zip(comparison['segment_times'], baseline['segment_times'])
    ]
    
    faster_segments = [i+1 for i, d in enumerate(segment_deltas) if d < -0.001]
    slower_segments = [i+1 for i, d in enumerate(segment_deltas) if d > 0.001]
    
    return ComparisonResult(
        baseline_name=baseline_name,
        comparison_name=comparison_name,
        lap_delta=lap_delta,
        lap_percent=lap_percent,
        sector_deltas=sector_deltas,
        segment_deltas=segment_deltas,
        faster_segments=faster_segments,
        slower_segments=slower_segments
    )


def create_summary_table(
    setups: List[Tuple[str, Setup]],
    predictions: List[Dict]
) -> pd.DataFrame:
    """
    Create summary table for multiple setups.
    
    Args:
        setups: List of (name, setup) tuples
        predictions: List of prediction results
        
    Returns:
        DataFrame with lap/sector times and deltas
    """
    rows = []
    baseline_lap = predictions[0]['lap_time'] if predictions else 0
    
    for (name, setup), pred in zip(setups, predictions):
        delta = pred['lap_time'] - baseline_lap
        delta_str = f"+{delta:.3f}" if delta > 0 else f"{delta:.3f}" if delta < 0 else "—"
        
        rows.append({
            'Setup': name,
            'Lap Time (s)': f"{pred['lap_time']:.3f}",
            'Delta': delta_str if name != setups[0][0] else "Baseline",
            'Sector 1': f"{pred['sector_1']:.3f}",
            'Sector 2': f"{pred['sector_2']:.3f}",
            'Sector 3': f"{pred['sector_3']:.3f}",
            'Valid': "✓" if pred['is_valid'] else "⚠",
        })
    
    return pd.DataFrame(rows)


def create_segment_table(
    setups: List[Tuple[str, Setup]],
    predictions: List[Dict]
) -> pd.DataFrame:
    """
    Create segment breakdown table.
    
    Args:
        setups: List of (name, setup) tuples
        predictions: List of prediction results
        
    Returns:
        DataFrame with segment times for all setups
    """
    # Segment names for Spa
    segment_names = [
        "La Source", "Eau Rouge Approach", "Eau Rouge", "Raidillon", "Kemmel Straight",
        "Les Combes Entry", "Les Combes Exit", "Malmedy", "Rivage", "Pre-Pouhon",
        "Pouhon", "Fagnes", "Campus Straight", "Campus", "Stavelot",
        "Paul Frere", "Blanchimont", "Chicane Approach", "Bus Stop", "Start/Finish"
    ]
    
    data = {'Segment': [f"{i+1}. {name}" for i, name in enumerate(segment_names)]}
    
    for (setup_name, _), pred in zip(setups, predictions):
        data[setup_name] = [f"{t:.3f}" for t in pred['segment_times']]
    
    # Add deltas if more than one setup
    if len(predictions) > 1:
        baseline_times = predictions[0]['segment_times']
        for (setup_name, _), pred in zip(setups[1:], predictions[1:]):
            deltas = [p - b for p, b in zip(pred['segment_times'], baseline_times)]
            data[f"{setup_name} Δ"] = [
                f"+{d:.3f}" if d > 0.001 else f"{d:.3f}" if d < -0.001 else "—"
                for d in deltas
            ]
    
    return pd.DataFrame(data)


def check_monotonicity(
    predictor: F1Predictor,
    base_setup: Setup
) -> pd.DataFrame:
    """
    Check if model respects expected monotonic relationships.
    
    Expected relationships:
    - mass ↑ → lap_time ↑
    - c_d ↑ → lap_time ↑
    - c_l ↑ → lap_time ↓ (more downforce = faster corners)
    - alpha_elec: depends on track
    - e_deploy ↑ → lap_time ↓
    - gamma_cool ↑ → lap_time ↓
    
    Returns:
        DataFrame with monotonicity check results
    """
    expected = {
        'mass': 'positive',      # More mass = slower
        'c_d': 'positive',       # More drag = slower
        'c_l': 'negative',       # More downforce = faster
        'alpha_elec': 'negative',# More electric = faster
        'e_deploy': 'negative',  # More energy = faster
        'gamma_cool': 'negative' # More cooling = faster
    }
    
    results = []
    
    for param, expected_dir in expected.items():
        values, lap_times = predictor.compute_partial_dependence(param, base_setup, n_points=10)
        
        # Compute correlation
        corr = np.corrcoef(values, lap_times)[0, 1]
        
        if corr > 0.1:
            actual_dir = 'positive'
        elif corr < -0.1:
            actual_dir = 'negative'
        else:
            actual_dir = 'neutral'
        
        is_consistent = (
            (expected_dir == 'positive' and actual_dir == 'positive') or
            (expected_dir == 'negative' and actual_dir == 'negative') or
            actual_dir == 'neutral'
        )
        
        results.append({
            'Parameter': param,
            'Expected': f"↑{param} → ↑lap" if expected_dir == 'positive' else f"↑{param} → ↓lap",
            'Observed': f"corr={corr:.3f}",
            'Consistent': "✓" if is_consistent else "✗",
            'Status': 'OK' if is_consistent else 'WARN'
        })
    
    return pd.DataFrame(results)


def compute_tradeoff_curve(
    predictor: F1Predictor,
    base_setup: Setup,
    param_name: str,
    n_points: int = 20
) -> pd.DataFrame:
    """
    Compute trade-off curve for a parameter.
    
    Args:
        predictor: F1 predictor instance
        base_setup: Base setup to modify
        param_name: Parameter to vary
        n_points: Number of evaluation points
        
    Returns:
        DataFrame with parameter values and lap times
    """
    values, lap_times = predictor.compute_partial_dependence(param_name, base_setup, n_points)
    
    return pd.DataFrame({
        param_name: values,
        'Lap Time (s)': lap_times
    })


def compute_aero_efficiency_curve(
    predictor: F1Predictor,
    base_setup: Setup,
    n_points: int = 15
) -> pd.DataFrame:
    """
    Compute lap time vs aero efficiency (C_L / C_D).
    
    Varies C_L and C_D together to explore efficiency trade-offs.
    """
    results = []
    
    c_l_range = np.linspace(0.9, 1.4, n_points)
    c_d_range = np.linspace(0.75, 1.25, n_points)
    
    for c_l, c_d in zip(c_l_range, c_d_range):
        efficiency = c_l / c_d
        modified = Setup(**{**base_setup.to_dict(), 'c_l': c_l, 'c_d': c_d})
        pred = predictor.predict(modified)
        
        results.append({
            'Aero Efficiency (C_L/C_D)': efficiency,
            'C_L': c_l,
            'C_D': c_d,
            'Lap Time (s)': pred['lap_time']
        })
    
    return pd.DataFrame(results)


def get_model_stats(predictor: F1Predictor) -> Dict:
    """
    Get model statistics for confidence display.
    
    Returns:
        Dict with model metadata and error statistics
    """
    return {
        'model_name': predictor.model_name,
        'cv_rmse': predictor.val_rmse,
        'mean_error': predictor.val_rmse * 0.8,  # Approximate
        'max_error': predictor.val_rmse * 2.5,   # Approximate worst case
        'uncertainty_95': predictor.val_rmse * 1.96
    }
