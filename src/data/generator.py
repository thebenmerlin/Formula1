"""
Physics-Informed Synthetic Data Generator for F1 Track Time Prediction

Generates large-scale synthetic datasets based on:
- Straight-line speed limits (power + drag)
- Cornering speed limits (aero load + mass)
- Energy deployment constraints
- Controlled noise injection
"""

import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class TrackSegment:
    """Represents a single track segment with physics properties."""
    id: int
    name: str
    segment_type: str  # 'straight' or 'corner'
    length_m: float
    apex_radius_m: Optional[float]
    elevation_change_m: float


@dataclass
class PhysicsConstants:
    """Physical constants for simulation."""
    air_density: float  # kg/m³
    gravity: float  # m/s²
    rolling_resistance: float
    tire_grip: float


@dataclass
class VehicleBaseline:
    """Baseline vehicle parameters."""
    ice_power_kw: float
    electric_power_kw: float
    frontal_area_m2: float


class SpaTrackGenerator:
    """
    Physics-informed synthetic data generator for Spa-Francorchamps.
    
    Generates lap times based on the 6-parameter design vector:
    x = [m, C_L, C_D, alpha_elec, E_deploy, gamma_cool]
    """
    
    def __init__(self, config_path: str = "configs/track_spa.yaml"):
        """Initialize generator with track configuration."""
        self.config = self._load_config(config_path)
        self.segments = self._parse_segments()
        self.physics = self._parse_physics()
        self.baseline = self._parse_baseline()
        self.sector_mapping = self._parse_sectors()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load track configuration from YAML."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _parse_segments(self) -> List[TrackSegment]:
        """Parse segment definitions from config."""
        segments = []
        for seg in self.config['segments']:
            segments.append(TrackSegment(
                id=seg['id'],
                name=seg['name'],
                segment_type=seg['type'],
                length_m=seg['length_m'],
                apex_radius_m=seg.get('apex_radius_m'),
                elevation_change_m=seg['elevation_change_m']
            ))
        return segments
    
    def _parse_physics(self) -> PhysicsConstants:
        """Parse physics constants from config."""
        p = self.config['physics']
        return PhysicsConstants(
            air_density=p['air_density_kg_m3'],
            gravity=p['gravity_m_s2'],
            rolling_resistance=p['rolling_resistance'],
            tire_grip=p['tire_grip_coefficient']
        )
    
    def _parse_baseline(self) -> VehicleBaseline:
        """Parse baseline vehicle parameters."""
        v = self.config['vehicle_baseline']
        return VehicleBaseline(
            ice_power_kw=v['ice_power_kw'],
            electric_power_kw=v['electric_power_kw'],
            frontal_area_m2=v['frontal_area_m2']
        )
    
    def _parse_sectors(self) -> Dict[int, int]:
        """Create segment-to-sector mapping."""
        mapping = {}
        for sector_name, sector_data in self.config['sectors'].items():
            sector_num = int(sector_name.split('_')[1])
            for seg_id in sector_data['segments']:
                mapping[seg_id] = sector_num
        return mapping
    
    def compute_segment_time(
        self,
        segment: TrackSegment,
        mass: float,
        c_l: float,
        c_d: float,
        alpha_elec: float,
        e_deploy: float,
        gamma_cool: float,
        energy_remaining: float
    ) -> Tuple[float, float]:
        """
        Compute time for a single segment based on vehicle setup.
        
        Physics principles:
        - Straights: Limited by power and drag
        - Corners: Limited by downforce, grip, and mass
        
        Returns:
            Tuple of (segment_time, energy_used)
        """
        rho = self.physics.air_density
        g = self.physics.gravity
        mu = self.physics.tire_grip
        frontal_area = self.baseline.frontal_area_m2
        
        # Total available power (with cooling factor)
        ice_power = self.baseline.ice_power_kw * 1000 * gamma_cool  # Convert to W
        elec_power = self.baseline.electric_power_kw * 1000 * alpha_elec
        
        # Energy deployment rate (MJ per lap distributed across 20 segments)
        energy_per_segment = (e_deploy * 1e6) / 20  # Joules per segment
        can_deploy = energy_remaining >= energy_per_segment * 0.5
        
        if segment.segment_type == 'straight':
            # Straight-line physics: Power limited
            total_power = ice_power + (elec_power if can_deploy else 0)
            
            # Approximate top speed from power balance with drag
            # P = F_drag * v = 0.5 * rho * Cd * A * v^3
            # v_max = (2P / (rho * Cd * A))^(1/3)
            drag_coefficient = c_d * 0.9  # Normalized to physical value
            v_max = (2 * total_power / (rho * drag_coefficient * frontal_area)) ** (1/3)
            v_max = min(v_max, 100)  # Cap at ~360 km/h
            
            # Average speed considering acceleration zones
            # Simplified: assume 85% of max speed on average
            v_avg = v_max * 0.85
            
            # Elevation effect
            elevation_factor = 1 + 0.002 * segment.elevation_change_m
            v_avg *= elevation_factor
            
            time = segment.length_m / v_avg
            energy_used = energy_per_segment * 0.3 if can_deploy else 0
            
        else:
            # Corner physics: Grip limited
            radius = segment.apex_radius_m if segment.apex_radius_m else 50
            
            # Downforce contribution to grip
            # F_downforce = 0.5 * rho * C_L * A * v^2
            # Max lateral acceleration: a_lat = (mu * (m*g + F_down)) / m
            # At corner: v^2/r = a_lat
            # Solving for v gives corner speed limit
            
            lift_coefficient = c_l * 3.5  # Scale normalized to physical
            
            # Iterative solve for corner speed (simplified)
            # v = sqrt(r * g * (mu + (0.5 * rho * C_L * A * v^2) / (m * g)))
            # Using approximation:
            base_speed = np.sqrt(radius * g * mu)
            
            # Downforce boost factor
            aero_factor = 1 + (lift_coefficient * frontal_area * rho) / (2 * mass)
            aero_factor = min(aero_factor, 1.6)  # Cap boost
            
            v_corner = base_speed * np.sqrt(aero_factor)
            
            # Mass penalty
            mass_factor = 750 / mass  # Reference mass 750kg
            v_corner *= (mass_factor ** 0.3)
            
            # Cap corner speed
            v_corner = min(v_corner, 80)  # ~290 km/h max in corners
            
            time = segment.length_m / v_corner
            energy_used = energy_per_segment * 0.5 if can_deploy else 0
        
        return time, energy_used
    
    def compute_lap_time(
        self,
        mass: float,
        c_l: float,
        c_d: float,
        alpha_elec: float,
        e_deploy: float,
        gamma_cool: float,
        noise_level: float = 0.0
    ) -> Dict:
        """
        Compute full lap time breakdown for a given setup.
        
        Args:
            mass: Total mass (kg)
            c_l: Normalized aero load coefficient
            c_d: Normalized aero drag coefficient
            alpha_elec: Electric power fraction (0-1)
            e_deploy: Max deployable energy (MJ)
            gamma_cool: Cooling aggressiveness factor
            noise_level: Fraction of noise to inject (0-1)
            
        Returns:
            Dict with segment_times, sector_times, and lap_time
        """
        energy_remaining = e_deploy * 1e6  # Convert to Joules
        
        segment_times = []
        sector_times = {1: 0.0, 2: 0.0, 3: 0.0}
        
        for segment in self.segments:
            time, energy_used = self.compute_segment_time(
                segment, mass, c_l, c_d, alpha_elec, e_deploy, gamma_cool, energy_remaining
            )
            
            # Inject controlled noise
            if noise_level > 0:
                noise = np.random.normal(0, time * noise_level)
                time = max(time + noise, time * 0.8)  # Floor at 80% nominal
            
            segment_times.append(time)
            energy_remaining -= energy_used
            
            # Accumulate sector times
            sector = self.sector_mapping[segment.id]
            sector_times[sector] += time
        
        lap_time = sum(segment_times)
        
        return {
            'segment_times': segment_times,
            'sector_1': sector_times[1],
            'sector_2': sector_times[2],
            'sector_3': sector_times[3],
            'lap_time': lap_time
        }
    
    def generate_dataset(
        self,
        n_samples: int = 100000,
        noise_level: float = 0.02,
        random_seed: int = 42,
        param_bounds: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Generate a large synthetic dataset of vehicle setups and lap times.
        
        Args:
            n_samples: Number of samples to generate
            noise_level: Noise injection level (fraction)
            random_seed: Random seed for reproducibility
            param_bounds: Optional custom parameter bounds
            
        Returns:
            DataFrame with design parameters and predicted times
        """
        np.random.seed(random_seed)
        
        # Default parameter bounds
        if param_bounds is None:
            param_bounds = {
                'mass': (700, 850),
                'c_l': (0.8, 1.5),
                'c_d': (0.7, 1.3),
                'alpha_elec': (0.0, 0.4),
                'e_deploy': (2.0, 4.0),
                'gamma_cool': (0.8, 1.2)
            }
        
        # Generate random samples using Latin Hypercube-like distribution
        samples = []
        
        for _ in tqdm(range(n_samples), desc="Generating samples"):
            # Sample design parameters
            mass = np.random.uniform(*param_bounds['mass'])
            c_l = np.random.uniform(*param_bounds['c_l'])
            c_d = np.random.uniform(*param_bounds['c_d'])
            alpha_elec = np.random.uniform(*param_bounds['alpha_elec'])
            e_deploy = np.random.uniform(*param_bounds['e_deploy'])
            gamma_cool = np.random.uniform(*param_bounds['gamma_cool'])
            
            # Compute lap time breakdown
            result = self.compute_lap_time(
                mass, c_l, c_d, alpha_elec, e_deploy, gamma_cool, noise_level
            )
            
            # Build sample record
            sample = {
                'mass': mass,
                'c_l': c_l,
                'c_d': c_d,
                'alpha_elec': alpha_elec,
                'e_deploy': e_deploy,
                'gamma_cool': gamma_cool,
                'sector_1': result['sector_1'],
                'sector_2': result['sector_2'],
                'sector_3': result['sector_3'],
                'lap_time': result['lap_time']
            }
            
            # Add individual segment times
            for i, seg_time in enumerate(result['segment_times'], 1):
                sample[f'segment_{i}'] = seg_time
            
            samples.append(sample)
        
        return pd.DataFrame(samples)


def main():
    """Generate and save synthetic dataset."""
    print("=" * 60)
    print("Formula1 Synthetic Data Generator")
    print("Spa-Francorchamps Track Time Prediction")
    print("=" * 60)
    
    # Initialize generator
    generator = SpaTrackGenerator()
    
    print(f"\nTrack: {generator.config['track']['name']}")
    print(f"Total segments: {len(generator.segments)}")
    print(f"Total length: {generator.config['track']['total_length_km']} km")
    
    # Generate dataset
    print("\n[1/3] Generating synthetic dataset...")
    df = generator.generate_dataset(
        n_samples=100000,
        noise_level=0.02,
        random_seed=42
    )
    
    # Save dataset
    output_path = Path("data/synthetic/lap_times_100k.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[2/3] Saving to {output_path}...")
    df.to_parquet(output_path, index=False)
    
    # Also save CSV for inspection
    csv_path = output_path.with_suffix('.csv')
    df.head(1000).to_csv(csv_path, index=False)
    print(f"Sample CSV saved to {csv_path}")
    
    # Summary statistics
    print("\n[3/3] Dataset Summary:")
    print("-" * 40)
    print(f"Total samples: {len(df):,}")
    print(f"Features: {len([c for c in df.columns if c not in ['lap_time', 'sector_1', 'sector_2', 'sector_3'] and not c.startswith('segment_')])}")
    print(f"Segment outputs: 20")
    print(f"Sector outputs: 3")
    print(f"\nLap time statistics:")
    print(f"  Mean: {df['lap_time'].mean():.3f} s")
    print(f"  Std:  {df['lap_time'].std():.3f} s")
    print(f"  Min:  {df['lap_time'].min():.3f} s")
    print(f"  Max:  {df['lap_time'].max():.3f} s")
    
    print("\n✓ Data generation complete!")
    return df


if __name__ == "__main__":
    main()
