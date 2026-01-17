"""
Spa-Francorchamps Track Definition

This module defines the Spa-Francorchamps circuit as a series of segments,
each with physical properties needed for lap time simulation.

Track Data Sources:
- Official FIA circuit data
- Approximate curvatures derived from track maps
- Elevation data from topographic analysis

Total Length: ~7.004 km
Segments: 20 (mix of corners and straights)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
import json


@dataclass
class TrackSegment:
    """
    A single track segment with physical properties.
    
    Attributes:
        segment_id: Unique identifier (0-indexed)
        name: Human-readable segment name
        length: Segment length in meters
        curvature: Average curvature in 1/m (0 for straights)
        elevation_change: Net elevation change in meters (+ve uphill)
        sector: Sector number (1, 2, or 3)
        segment_type: 'straight', 'corner_slow', 'corner_medium', 'corner_fast'
    """
    segment_id: int
    name: str
    length: float           # meters
    curvature: float        # 1/m (radius = 1/curvature for corners)
    elevation_change: float # meters
    sector: int
    segment_type: str
    
    @property
    def radius(self) -> float:
        """Corner radius in meters (inf for straights)."""
        if self.curvature == 0:
            return float('inf')
        return 1.0 / self.curvature
    
    @property
    def is_corner(self) -> bool:
        """True if this is a corner segment."""
        return self.curvature > 0


def create_spa_track() -> List[TrackSegment]:
    """
    Create the Spa-Francorchamps track definition.
    
    Returns:
        List of TrackSegment objects representing the full circuit.
    
    Notes:
        - Track is divided into 20 segments for simulation granularity
        - Curvatures are approximate and normalized for the physics model
        - Elevation changes are approximate based on track profile
    """
    segments = [
        # SECTOR 1: La Source to Eau Rouge exit
        TrackSegment(0, "La Source Hairpin", 120, 0.025, -5, 1, "corner_slow"),
        TrackSegment(1, "Pit Straight Exit", 350, 0.0, -30, 1, "straight"),
        TrackSegment(2, "Eau Rouge Entry", 150, 0.012, -10, 1, "corner_fast"),
        TrackSegment(3, "Eau Rouge Apex", 120, 0.015, 25, 1, "corner_fast"),
        TrackSegment(4, "Raidillon", 180, 0.010, 35, 1, "corner_fast"),
        
        # SECTOR 2: Kemmel to Campus
        TrackSegment(5, "Kemmel Straight", 750, 0.0, 10, 2, "straight"),
        TrackSegment(6, "Les Combes 1", 150, 0.014, -5, 2, "corner_medium"),
        TrackSegment(7, "Les Combes 2", 130, 0.016, -3, 2, "corner_medium"),
        TrackSegment(8, "Malmedy", 200, 0.008, -8, 2, "corner_fast"),
        TrackSegment(9, "Rivage", 180, 0.018, -20, 2, "corner_medium"),
        TrackSegment(10, "Rivage Exit", 250, 0.0, -15, 2, "straight"),
        
        # SECTOR 2 continued: Pouhon area
        TrackSegment(11, "Pouhon Entry", 200, 0.006, -5, 2, "corner_fast"),
        TrackSegment(12, "Pouhon Apex", 250, 0.007, 0, 2, "corner_fast"),
        TrackSegment(13, "Fagnes", 350, 0.0, 5, 2, "straight"),
        
        # SECTOR 3: Campus to finish
        TrackSegment(14, "Campus Entry", 180, 0.010, 0, 3, "corner_medium"),
        TrackSegment(15, "Campus Exit", 200, 0.008, 3, 3, "corner_medium"),
        TrackSegment(16, "Stavelot", 220, 0.012, -10, 3, "corner_medium"),
        TrackSegment(17, "Blanchimont", 400, 0.004, -5, 3, "corner_fast"),
        TrackSegment(18, "Bus Stop Chicane", 250, 0.020, 0, 3, "corner_slow"),
        TrackSegment(19, "Start/Finish Straight", 834, 0.0, 5, 3, "straight"),
    ]
    
    return segments


class SpaTrack:
    """
    Spa-Francorchamps circuit wrapper with utility methods.
    
    This class provides convenient access to track data and
    precomputed properties for the physics simulation.
    """
    
    def __init__(self):
        self.segments = create_spa_track()
        self._precompute_properties()
    
    def _precompute_properties(self):
        """Precompute aggregate track properties."""
        self.total_length = sum(s.length for s in self.segments)
        self.total_elevation_gain = sum(
            s.elevation_change for s in self.segments if s.elevation_change > 0
        )
        self.total_elevation_loss = abs(sum(
            s.elevation_change for s in self.segments if s.elevation_change < 0
        ))
        self.n_segments = len(self.segments)
        
        # Cumulative distances for position tracking
        self._cumulative_distances = [0.0]
        for seg in self.segments:
            self._cumulative_distances.append(
                self._cumulative_distances[-1] + seg.length
            )
    
    @property
    def corner_segments(self) -> List[TrackSegment]:
        """Return only corner segments."""
        return [s for s in self.segments if s.is_corner]
    
    @property
    def straight_segments(self) -> List[TrackSegment]:
        """Return only straight segments."""
        return [s for s in self.segments if not s.is_corner]
    
    def get_segment_at_distance(self, distance: float) -> Tuple[TrackSegment, float]:
        """
        Get the segment at a given distance from start.
        
        Args:
            distance: Distance from start line in meters
            
        Returns:
            Tuple of (segment, distance_into_segment)
        """
        # Wrap around for multiple laps
        distance = distance % self.total_length
        
        for i, seg in enumerate(self.segments):
            seg_start = self._cumulative_distances[i]
            seg_end = self._cumulative_distances[i + 1]
            if seg_start <= distance < seg_end:
                return seg, distance - seg_start
        
        # Edge case: exactly at finish
        return self.segments[-1], self.segments[-1].length
    
    def get_cumulative_distance(self, segment_id: int) -> float:
        """Get cumulative distance to start of segment."""
        return self._cumulative_distances[segment_id]
    
    def get_sector_segments(self, sector: int) -> List[TrackSegment]:
        """Get all segments in a given sector."""
        return [s for s in self.segments if s.sector == sector]
    
    def to_dict(self) -> dict:
        """Export track data as dictionary (for JSON serialization)."""
        return {
            'name': 'Spa-Francorchamps',
            'total_length': self.total_length,
            'n_segments': self.n_segments,
            'segments': [
                {
                    'id': s.segment_id,
                    'name': s.name,
                    'length': s.length,
                    'curvature': s.curvature,
                    'elevation_change': s.elevation_change,
                    'sector': s.sector,
                    'type': s.segment_type,
                }
                for s in self.segments
            ]
        }
    
    def to_json(self, filepath: str):
        """Export track data to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def summary(self) -> str:
        """Return a human-readable summary of the track."""
        return f"""
Spa-Francorchamps Circuit
=========================
Total Length: {self.total_length:.0f} m ({self.total_length/1000:.3f} km)
Segments: {self.n_segments}
Corners: {len(self.corner_segments)}
Straights: {len(self.straight_segments)}
Elevation Gain: {self.total_elevation_gain:.0f} m
Elevation Loss: {self.total_elevation_loss:.0f} m

Sector Breakdown:
  Sector 1: {len(self.get_sector_segments(1))} segments
  Sector 2: {len(self.get_sector_segments(2))} segments
  Sector 3: {len(self.get_sector_segments(3))} segments
"""


# Singleton instance for easy access
SPA_TRACK = SpaTrack()


if __name__ == "__main__":
    # Print track summary when run directly
    print(SPA_TRACK.summary())
    
    print("\nSegment Details:")
    print("-" * 80)
    for seg in SPA_TRACK.segments:
        radius_str = f"{seg.radius:.0f}m" if seg.is_corner else "∞"
        print(f"  {seg.segment_id:2d}. {seg.name:25s} | {seg.length:4.0f}m | "
              f"R={radius_str:>6s} | ΔH={seg.elevation_change:+4.0f}m | S{seg.sector}")
