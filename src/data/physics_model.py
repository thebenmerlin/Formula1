"""
Physics-Informed Performance Model

This module implements simplified physics calculations for F1 vehicle performance.
These are NOT high-fidelity simulations, but physics-informed heuristics suitable
for early-stage design exploration and ML training data generation.

Models Include:
1. Longitudinal dynamics (acceleration, braking)
2. Cornering (grip-limited speed)
3. Energy deployment and regeneration
4. Thermal accumulation

All assumptions are explicitly documented for transparency.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.config import PHYSICS, REFERENCE, DESIGN_RANGES
from src.data.track import TrackSegment, SPA_TRACK


@dataclass
class VehicleState:
    """
    Instantaneous state of the vehicle during simulation.
    
    Used to track progress through a lap segment.
    """
    speed: float = 0.0              # m/s
    position: float = 0.0           # m from segment start
    energy_used: float = 0.0        # MJ
    energy_remaining: float = 0.0   # MJ
    thermal_state: float = 0.0      # normalized 0-1
    time: float = 0.0               # seconds


@dataclass
class SegmentResult:
    """
    Performance results for a single track segment.
    """
    segment_id: int
    segment_name: str
    time: float                     # seconds
    entry_speed: float              # m/s
    exit_speed: float               # m/s
    min_speed: float                # m/s
    max_speed: float                # m/s
    energy_deployed: float          # MJ
    energy_regenerated: float       # MJ
    thermal_increase: float         # delta thermal state
    distance: float                 # meters


class PhysicsModel:
    """
    Physics-informed vehicle performance model.
    
    This model calculates approximate lap performance based on vehicle
    design parameters and track characteristics.
    
    Key Simplifications:
    - Point-mass vehicle (no weight transfer, no suspension)
    - Grip-circle cornering (combined lat/long limit)
    - Linear thermal model
    - Constant efficiency factors
    """
    
    def __init__(self, design_vector: Dict[str, float]):
        """
        Initialize physics model with design parameters.
        
        Args:
            design_vector: Dict with keys m, C_L, C_D, alpha_elec, E_deploy, gamma_cool
        """
        self.m = design_vector['m']
        self.C_L = design_vector['C_L']
        self.C_D = design_vector['C_D']
        self.alpha_elec = design_vector['alpha_elec']
        self.E_deploy = design_vector['E_deploy']
        self.gamma_cool = design_vector['gamma_cool']
        
        # Derived quantities
        self._compute_derived_params()
    
    def _compute_derived_params(self):
        """Compute derived vehicle parameters."""
        # Aero forces (reference values scaled by coefficients)
        self.A = REFERENCE.A_ref  # Frontal area
        
        # Power available
        self.P_total = PHYSICS.P_total_max
        self.P_electric_max = self.alpha_elec * self.P_total
        self.P_ice = (1 - self.alpha_elec) * self.P_total
        
        # L/D ratio for efficiency metric
        if self.C_D > 0:
            self.L_D_ratio = self.C_L / self.C_D
        else:
            self.L_D_ratio = 1.0
    
    def _aero_downforce(self, speed: float) -> float:
        """
        Calculate aerodynamic downforce at given speed.
        
        F_down = 0.5 * rho * v² * A * C_L * C_L_ref
        
        Args:
            speed: Vehicle speed in m/s
            
        Returns:
            Downforce in Newtons
        """
        return (0.5 * PHYSICS.rho_air * speed**2 * self.A * 
                self.C_L * REFERENCE.C_L_ref)
    
    def _aero_drag(self, speed: float) -> float:
        """
        Calculate aerodynamic drag at given speed.
        
        F_drag = 0.5 * rho * v² * A * C_D * C_D_ref
        
        Args:
            speed: Vehicle speed in m/s
            
        Returns:
            Drag force in Newtons
        """
        return (0.5 * PHYSICS.rho_air * speed**2 * self.A * 
                self.C_D * REFERENCE.C_D_ref)
    
    def _effective_grip(self, speed: float) -> float:
        """
        Calculate effective grip (tyre + aero).
        
        At high speed, aero downforce significantly increases grip.
        
        Args:
            speed: Vehicle speed in m/s
            
        Returns:
            Effective friction coefficient
        """
        # Base mechanical grip
        F_weight = self.m * PHYSICS.g
        
        # Aero-enhanced normal force
        F_downforce = self._aero_downforce(speed)
        F_normal = F_weight + F_downforce
        
        # Effective mu (slightly decreases at very high loads)
        mu_eff = PHYSICS.mu_tyre * (F_weight + 0.9 * F_downforce) / F_normal
        
        return mu_eff
    
    def max_cornering_speed(self, curvature: float) -> float:
        """
        Calculate maximum cornering speed for given curvature.
        
        Uses iterative solution since grip depends on speed (aero).
        
        v_max where: m * v² / r = mu_eff(v) * F_normal(v)
        
        Args:
            curvature: Track curvature in 1/m
            
        Returns:
            Maximum sustainable cornering speed in m/s
        """
        if curvature == 0:
            return 350 / 3.6  # ~97 m/s = 350 km/h theoretical max
        
        radius = 1.0 / curvature
        
        # Initial guess based on mechanical grip only
        v_guess = np.sqrt(PHYSICS.mu_tyre * PHYSICS.g * radius)
        
        # Iterate to find speed where lateral force = available grip
        for _ in range(10):
            F_centripetal_required = self.m * v_guess**2 / radius
            
            F_weight = self.m * PHYSICS.g
            F_downforce = self._aero_downforce(v_guess)
            F_normal = F_weight + F_downforce
            
            F_grip_available = PHYSICS.mu_tyre * F_normal
            
            # Update speed estimate
            v_new = np.sqrt(F_grip_available * radius / self.m)
            
            if abs(v_new - v_guess) < 0.1:
                break
            v_guess = 0.5 * (v_guess + v_new)  # Damped update
        
        return v_guess
    
    def max_acceleration(self, speed: float, gradient: float = 0) -> float:
        """
        Calculate maximum acceleration at given speed.
        
        Limited by:
        - Available power (P = F * v)
        - Traction (grip limit)
        - Gradient resistance
        
        Args:
            speed: Current speed in m/s
            gradient: Road gradient (positive = uphill)
            
        Returns:
            Maximum acceleration in m/s²
        """
        if speed < 1.0:
            speed = 1.0  # Avoid division by zero
        
        # Power-limited acceleration
        F_drive_power = self.P_total / speed
        
        # Traction-limited
        F_weight = self.m * PHYSICS.g
        F_downforce = self._aero_downforce(speed)
        F_traction_limit = PHYSICS.mu_tyre * (F_weight + F_downforce) * 0.7  # Rear bias
        
        F_drive = min(F_drive_power, F_traction_limit)
        
        # Resistance forces
        F_drag = self._aero_drag(speed)
        F_gradient = self.m * PHYSICS.g * np.sin(np.arctan(gradient))
        F_rolling = 0.015 * self.m * PHYSICS.g  # Rolling resistance
        
        F_net = F_drive - F_drag - F_gradient - F_rolling
        
        return F_net / self.m
    
    def max_deceleration(self, speed: float, gradient: float = 0) -> float:
        """
        Calculate maximum braking deceleration (negative).
        
        Args:
            speed: Current speed in m/s
            gradient: Road gradient
            
        Returns:
            Maximum deceleration in m/s² (negative value)
        """
        # Brake force limited by grip
        F_weight = self.m * PHYSICS.g
        F_downforce = self._aero_downforce(speed)
        F_brake_limit = PHYSICS.mu_braking * (F_weight + F_downforce)
        
        # Aero drag helps braking
        F_drag = self._aero_drag(speed)
        
        # Gradient effect
        F_gradient = self.m * PHYSICS.g * np.sin(np.arctan(gradient))
        
        # Total deceleration
        F_total = F_brake_limit + F_drag - F_gradient
        
        return -F_total / self.m  # Negative for deceleration
    
    def simulate_segment(
        self,
        segment: TrackSegment,
        entry_speed: float,
        energy_remaining: float,
        thermal_state: float
    ) -> SegmentResult:
        """
        Simulate vehicle performance through a track segment.
        
        Uses a simplified approach:
        1. Calculate min speed (corner apex or straight V_max)
        2. Determine braking/accelerating phases
        3. Integrate time using average speeds
        4. Track energy and thermal state
        
        Args:
            segment: Track segment to simulate
            entry_speed: Speed at segment entry (m/s)
            energy_remaining: Available electric energy (MJ)
            thermal_state: Current thermal state (0-1)
            
        Returns:
            SegmentResult with all performance data
        """
        gradient = segment.elevation_change / segment.length
        
        # Target speed for this segment
        if segment.is_corner:
            v_apex = self.max_cornering_speed(segment.curvature)
        else:
            # Straight - accelerate to max speed
            v_apex = 350 / 3.6  # Theoretical max
        
        # Speed envelope through segment
        v_entry = entry_speed
        v_min = min(v_entry, v_apex)
        
        # Calculate exit speed
        if segment.is_corner:
            # Exit at or below apex speed
            v_exit = v_apex * 0.95  # Slight loss exiting
        else:
            # Accelerate on straight
            avg_accel = self.max_acceleration((v_entry + v_apex) / 2, gradient)
            # v² = u² + 2as
            v_exit_sq = v_entry**2 + 2 * avg_accel * segment.length
            v_exit = np.sqrt(max(v_exit_sq, v_entry**2))
            v_exit = min(v_exit, v_apex)
        
        v_max = max(v_entry, v_exit, v_apex)
        v_avg = (v_entry + v_exit + v_min) / 3
        
        # Segment time (simplified)
        if v_avg > 1:
            seg_time = segment.length / v_avg
        else:
            seg_time = segment.length / 10  # Fallback
        
        # Energy model
        if segment.is_corner:
            # Less deployment in corners
            deploy_factor = 0.3 + 0.7 * (1 - segment.curvature * 100)
        else:
            # Full deployment on straights
            deploy_factor = 1.0
        
        # Energy deployed this segment
        P_electric = self.P_electric_max * deploy_factor
        E_deployed = (P_electric * seg_time) / 1e6  # Convert to MJ
        
        # Cap by remaining energy
        E_deployed = min(E_deployed, energy_remaining)
        
        # Regeneration under braking
        if v_exit < v_entry:
            # Regenerating
            delta_KE = 0.5 * self.m * (v_entry**2 - v_exit**2)
            E_regen = delta_KE * PHYSICS.eta_regen / 1e6
            E_regen = min(E_regen, 0.3)  # Cap regen per segment
        else:
            E_regen = 0.0
        
        # Thermal model
        # Heat generation from battery discharge
        Q_discharge = E_deployed * (1 - PHYSICS.eta_deploy)  # Waste heat
        
        # Cooling effect
        Q_cooling = self.gamma_cool * 0.1 * seg_time  # Simplified linear cooling
        
        # Net thermal change
        delta_thermal = (Q_discharge - Q_cooling) / 2.0  # Normalized
        delta_thermal = np.clip(delta_thermal, -0.1, 0.1)  # Bound rate
        
        return SegmentResult(
            segment_id=segment.segment_id,
            segment_name=segment.name,
            time=seg_time,
            entry_speed=v_entry,
            exit_speed=v_exit,
            min_speed=v_min,
            max_speed=v_max,
            energy_deployed=E_deployed,
            energy_regenerated=E_regen,
            thermal_increase=delta_thermal,
            distance=segment.length
        )
    
    def simulate_lap(self) -> Tuple[float, float, float, List[SegmentResult]]:
        """
        Simulate a complete lap of Spa-Francorchamps.
        
        Returns:
            Tuple of (lap_time, energy_used, thermal_risk, segment_results)
        """
        # Initialize state
        energy_remaining = self.E_deploy  # MJ
        thermal_state = 0.2  # Start slightly warm
        
        # Start from grid (low speed)
        current_speed = 50 / 3.6  # 50 km/h
        
        total_time = 0.0
        total_energy = 0.0
        max_thermal = thermal_state
        
        segment_results = []
        
        for segment in SPA_TRACK.segments:
            result = self.simulate_segment(
                segment=segment,
                entry_speed=current_speed,
                energy_remaining=energy_remaining,
                thermal_state=thermal_state
            )
            
            segment_results.append(result)
            
            # Update state
            total_time += result.time
            energy_deployed = result.energy_deployed - result.energy_regenerated
            total_energy += result.energy_deployed
            energy_remaining = max(0, energy_remaining - energy_deployed + result.energy_regenerated)
            thermal_state = np.clip(thermal_state + result.thermal_increase, 0, 1)
            max_thermal = max(max_thermal, thermal_state)
            
            # Update speed for next segment
            current_speed = result.exit_speed
        
        return total_time, total_energy, max_thermal, segment_results
    
    def get_outputs(self) -> Dict[str, float]:
        """
        Calculate all output targets for the current design.
        
        Returns:
            Dict with lap_time, energy_used, thermal_risk
        """
        lap_time, energy_used, thermal_risk, _ = self.simulate_lap()
        
        return {
            'lap_time': lap_time,
            'energy_used': energy_used,
            'thermal_risk': thermal_risk
        }


def compute_baseline_lap() -> Dict[str, float]:
    """
    Compute lap performance for the reference/baseline vehicle.
    
    Returns:
        Dict with baseline lap_time, energy_used, thermal_risk
    """
    design = {
        'm': REFERENCE.m,
        'C_L': REFERENCE.C_L,
        'C_D': REFERENCE.C_D,
        'alpha_elec': REFERENCE.alpha_elec,
        'E_deploy': REFERENCE.E_deploy,
        'gamma_cool': REFERENCE.gamma_cool
    }
    
    model = PhysicsModel(design)
    return model.get_outputs()


if __name__ == "__main__":
    # Test with baseline vehicle
    print("Testing Physics Model with Baseline Vehicle")
    print("=" * 50)
    
    baseline = compute_baseline_lap()
    print(f"Baseline Lap Time: {baseline['lap_time']:.2f} s")
    print(f"Energy Used: {baseline['energy_used']:.2f} MJ")
    print(f"Thermal Risk: {baseline['thermal_risk']:.3f}")
    
    # Test with lighter, high-downforce setup
    print("\nTesting High-Downforce Setup")
    print("-" * 50)
    
    fast_design = {
        'm': 750,
        'C_L': 1.25,
        'C_D': 1.1,
        'alpha_elec': 0.50,
        'E_deploy': 8.0,
        'gamma_cool': 1.1
    }
    
    fast_model = PhysicsModel(fast_design)
    lap_time, energy, thermal, segments = fast_model.simulate_lap()
    
    print(f"Lap Time: {lap_time:.2f} s (Δ{lap_time - baseline['lap_time']:+.2f}s)")
    print(f"Energy Used: {energy:.2f} MJ")
    print(f"Thermal Risk: {thermal:.3f}")
    
    print("\nSegment Breakdown (first 5):")
    for seg in segments[:5]:
        print(f"  {seg.segment_name:20s}: {seg.time:.2f}s | "
              f"V={seg.min_speed*3.6:.0f}-{seg.max_speed*3.6:.0f} km/h | "
              f"E={seg.energy_deployed:.2f} MJ")
