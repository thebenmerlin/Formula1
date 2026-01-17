"""
Segment Predictor for Streamlit Animation

Generates segment-by-segment performance predictions for visualizing
vehicle designs as points moving around the track.

This module produces precomputed data for animation - NO live ML inference.
"""

import numpy as np
import pandas as pd
import json
import os
import sys
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import joblib

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.config import MODELS_DIR, OUTPUTS_DIR, TARGET_NAMES
from src.data.track import SPA_TRACK, TrackSegment
from src.data.physics_model import PhysicsModel
from src.models.features import FeaturePipeline


@dataclass
class SegmentPrediction:
    """Prediction for a single track segment."""
    segment_id: int
    segment_name: str
    cumulative_distance: float  # meters from start
    segment_length: float       # meters
    time_in_segment: float      # seconds
    cumulative_time: float      # seconds from start
    entry_speed: float          # km/h
    exit_speed: float           # km/h
    avg_speed: float            # km/h
    energy_deployed: float      # MJ
    cumulative_energy: float    # MJ
    thermal_state: float        # 0-1
    sector: int
    is_corner: bool


@dataclass  
class LapPrediction:
    """Complete lap prediction for a design."""
    design_name: str
    design_params: Dict[str, float]
    total_lap_time: float       # seconds
    total_energy_used: float    # MJ
    peak_thermal_risk: float    # 0-1
    segments: List[SegmentPrediction]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'design_name': self.design_name,
            'design_params': self.design_params,
            'total_lap_time': self.total_lap_time,
            'total_energy_used': self.total_energy_used,
            'peak_thermal_risk': self.peak_thermal_risk,
            'segments': [asdict(s) for s in self.segments]
        }


class SegmentPredictor:
    """
    Generates segment-wise predictions using physics model.
    
    Uses the physics model directly for segment breakdown,
    with ML predictions for overall lap metrics comparison.
    """
    
    def __init__(self, model: Any = None, pipeline: FeaturePipeline = None):
        """
        Initialize predictor.
        
        Args:
            model: Trained ML model (optional, for comparison)
            pipeline: Feature pipeline (required if model provided)
        """
        self.model = model
        self.pipeline = pipeline
        self.track = SPA_TRACK
    
    @classmethod
    def load(cls, model_name: str = 'random_forest') -> 'SegmentPredictor':
        """
        Load predictor with trained model.
        
        Args:
            model_name: Name of saved model to load
            
        Returns:
            Initialized SegmentPredictor
        """
        model_path = os.path.join(MODELS_DIR, f'{model_name}_model.joblib')
        pipeline_path = os.path.join(MODELS_DIR, 'feature_pipeline.joblib')
        
        model = joblib.load(model_path)
        pipeline = FeaturePipeline.load(pipeline_path)
        
        return cls(model=model, pipeline=pipeline)
    
    def predict_lap(self, design: Dict[str, float], name: str = 'Design') -> LapPrediction:
        """
        Generate segment-by-segment predictions for a design.
        
        Args:
            design: Design parameter dict
            name: Name for this design
            
        Returns:
            LapPrediction with all segment data
        """
        # Use physics model for detailed segment breakdown
        physics = PhysicsModel(design)
        lap_time, energy_used, thermal_risk, segment_results = physics.simulate_lap()
        
        # Build segment predictions
        segments = []
        cumulative_time = 0.0
        cumulative_energy = 0.0
        
        for seg_result in segment_results:
            segment = self.track.segments[seg_result.segment_id]
            
            cumulative_time += seg_result.time
            cumulative_energy += seg_result.energy_deployed
            
            seg_pred = SegmentPrediction(
                segment_id=seg_result.segment_id,
                segment_name=seg_result.segment_name,
                cumulative_distance=self.track.get_cumulative_distance(seg_result.segment_id),
                segment_length=seg_result.distance,
                time_in_segment=seg_result.time,
                cumulative_time=cumulative_time,
                entry_speed=seg_result.entry_speed * 3.6,  # Convert to km/h
                exit_speed=seg_result.exit_speed * 3.6,
                avg_speed=(seg_result.entry_speed + seg_result.exit_speed) / 2 * 3.6,
                energy_deployed=seg_result.energy_deployed,
                cumulative_energy=cumulative_energy,
                thermal_state=seg_result.thermal_increase,
                sector=segment.sector,
                is_corner=segment.is_corner
            )
            segments.append(seg_pred)
        
        return LapPrediction(
            design_name=name,
            design_params=design,
            total_lap_time=lap_time,
            total_energy_used=energy_used,
            peak_thermal_risk=thermal_risk,
            segments=segments
        )
    
    def predict_multiple(
        self,
        designs: List[Dict[str, float]],
        names: Optional[List[str]] = None
    ) -> List[LapPrediction]:
        """
        Generate predictions for multiple designs.
        
        Args:
            designs: List of design parameter dicts
            names: Optional list of design names
            
        Returns:
            List of LapPrediction objects
        """
        if names is None:
            names = [f'Design_{i}' for i in range(len(designs))]
        
        predictions = []
        for design, name in zip(designs, names):
            pred = self.predict_lap(design, name)
            predictions.append(pred)
        
        return predictions
    
    def export_for_streamlit(
        self,
        predictions: List[LapPrediction],
        output_path: Optional[str] = None
    ) -> str:
        """
        Export predictions in Streamlit-ready format.
        
        Creates JSON file with:
        - Track geometry (for map rendering)
        - Segment predictions per design (for animation)
        - Summary metrics (for display)
        
        Args:
            predictions: List of LapPrediction objects
            output_path: Output file path (default: outputs/streamlit_data.json)
            
        Returns:
            Path to saved file
        """
        if output_path is None:
            os.makedirs(OUTPUTS_DIR, exist_ok=True)
            output_path = os.path.join(OUTPUTS_DIR, 'streamlit_data.json')
        
        # Build track geometry
        track_data = self.track.to_dict()
        
        # Add cumulative distances for animation
        track_data['cumulative_distances'] = [
            self.track.get_cumulative_distance(i) 
            for i in range(len(self.track.segments))
        ]
        track_data['cumulative_distances'].append(self.track.total_length)
        
        # Summary table
        summary = []
        for pred in predictions:
            summary.append({
                'name': pred.design_name,
                'lap_time': round(pred.total_lap_time, 2),
                'energy_used': round(pred.total_energy_used, 2),
                'thermal_risk': round(pred.peak_thermal_risk, 3),
                'params': pred.design_params
            })
        
        # Full data structure
        output = {
            'track': track_data,
            'predictions': [p.to_dict() for p in predictions],
            'summary': summary,
            'metadata': {
                'n_designs': len(predictions),
                'n_segments': len(self.track.segments),
                'total_track_length': self.track.total_length
            }
        }
        
        # Save
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Exported Streamlit data to: {output_path}")
        
        return output_path


def generate_example_predictions() -> str:
    """
    Generate predictions for example design configurations.
    
    Returns:
        Path to saved Streamlit data file
    """
    predictor = SegmentPredictor()
    
    # Example designs representing different philosophies
    designs = [
        # Balanced baseline
        {
            'm': 770, 'C_L': 1.0, 'C_D': 1.0,
            'alpha_elec': 0.47, 'E_deploy': 7.0, 'gamma_cool': 1.0
        },
        # Low drag (Monza setup)
        {
            'm': 755, 'C_L': 0.85, 'C_D': 0.88,
            'alpha_elec': 0.52, 'E_deploy': 8.0, 'gamma_cool': 0.9
        },
        # High downforce (Monaco setup)
        {
            'm': 785, 'C_L': 1.28, 'C_D': 1.18,
            'alpha_elec': 0.42, 'E_deploy': 6.0, 'gamma_cool': 1.15
        },
        # Aggressive electric
        {
            'm': 745, 'C_L': 1.15, 'C_D': 1.05,
            'alpha_elec': 0.58, 'E_deploy': 8.8, 'gamma_cool': 1.25
        },
        # Conservative thermal
        {
            'm': 780, 'C_L': 0.95, 'C_D': 0.95,
            'alpha_elec': 0.40, 'E_deploy': 5.5, 'gamma_cool': 1.3
        }
    ]
    
    names = [
        'Balanced Baseline',
        'Low Drag (Monza)',
        'High Downforce (Monaco)',
        'Aggressive Electric',
        'Conservative Thermal'
    ]
    
    print("Generating segment predictions for example designs...")
    predictions = predictor.predict_multiple(designs, names)
    
    # Print summary
    print("\nDesign Summary:")
    print("-" * 70)
    for pred in predictions:
        print(f"{pred.design_name:25s} | {pred.total_lap_time:6.2f}s | "
              f"{pred.total_energy_used:5.2f} MJ | Thermal: {pred.peak_thermal_risk:.3f}")
    
    # Export for Streamlit
    return predictor.export_for_streamlit(predictions)


if __name__ == "__main__":
    generate_example_predictions()
