"""
Spa-Francorchamps Track Visualization

Accurate 2D representation of the Spa circuit for Streamlit visualization.
Based on real GPS coordinates, normalized to 0-1 range.

The circuit flows:
1. Start/Finish straight → La Source hairpin
2. Down through Eau Rouge → Raidillon (climbing)
3. Kemmel straight → Les Combes
4. Through Malmedy, Rivage, Pouhon
5. Stavelot, Blanchimont → Bus Stop
6. Back to Start/Finish
"""

import numpy as np
from typing import List, Tuple, Dict

# Accurate Spa-Francorchamps coordinates (normalized 0-1)
# Based on real GPS track data, oriented with start/finish at right side
# Track direction: clockwise as seen from above

SPA_TRACK_POINTS = [
    # === START/FINISH STRAIGHT ===
    (0.92, 0.25),   # Start/Finish line
    (0.88, 0.24),
    (0.84, 0.23),
    
    # === LA SOURCE HAIRPIN ===
    (0.80, 0.22),   # Braking zone
    (0.77, 0.21),
    (0.75, 0.21),   # Apex
    (0.74, 0.22),
    (0.74, 0.24),   # Exit
    (0.75, 0.26),
    
    # === DOWN TO EAU ROUGE ===
    (0.76, 0.28),
    (0.75, 0.31),
    (0.73, 0.34),
    (0.70, 0.37),
    (0.66, 0.40),
    (0.62, 0.42),   # Bottom of hill
    
    # === EAU ROUGE (Left-right flick) ===
    (0.58, 0.43),
    (0.55, 0.43),   # Eau Rouge left
    (0.52, 0.42),   # Compression point
    
    # === RAIDILLON (Climbing right) ===
    (0.49, 0.40),
    (0.47, 0.37),   # Climbing up
    (0.45, 0.34),
    (0.43, 0.31),
    (0.42, 0.28),   # Top of Raidillon
    
    # === KEMMEL STRAIGHT ===
    (0.40, 0.25),
    (0.37, 0.22),
    (0.33, 0.19),
    (0.28, 0.16),
    (0.23, 0.14),
    (0.18, 0.12),   # End of Kemmel
    
    # === LES COMBES ===
    (0.14, 0.11),   # Braking
    (0.11, 0.11),
    (0.09, 0.12),   # Turn 1
    (0.08, 0.14),
    (0.08, 0.16),   # Turn 2
    (0.09, 0.18),
    
    # === MALMEDY ===
    (0.10, 0.21),
    (0.10, 0.24),
    (0.09, 0.27),   # Malmedy apex
    
    # === RIVAGE ===
    (0.08, 0.30),
    (0.08, 0.33),
    (0.09, 0.36),   # Rivage hairpin
    (0.11, 0.38),
    (0.13, 0.39),
    
    # === THROUGH THE FOREST ===
    (0.16, 0.40),
    (0.19, 0.42),
    (0.22, 0.45),
    
    # === POUHON (Double apex left) ===
    (0.25, 0.48),
    (0.27, 0.51),
    (0.28, 0.54),   # Pouhon entry
    (0.28, 0.57),
    (0.27, 0.60),   # Pouhon apex 1
    (0.26, 0.63),
    (0.26, 0.66),   # Pouhon apex 2
    (0.27, 0.69),
    
    # === FAGNES / LES FAGNES ===
    (0.29, 0.72),
    (0.32, 0.74),
    (0.36, 0.76),
    
    # === CAMPUS (Chicane) ===
    (0.40, 0.77),
    (0.44, 0.77),
    (0.47, 0.76),   # Campus
    (0.50, 0.75),
    
    # === STAVELOT ===
    (0.54, 0.74),
    (0.58, 0.72),
    (0.61, 0.70),   # Stavelot entry
    (0.63, 0.67),
    (0.64, 0.64),   # Stavelot apex
    (0.65, 0.61),
    
    # === CURVE / PAUL FRERE ===
    (0.66, 0.58),
    (0.68, 0.55),
    (0.71, 0.52),
    
    # === BLANCHIMONT (Fast left) ===
    (0.74, 0.49),
    (0.77, 0.46),   # Blanchimont entry
    (0.80, 0.43),
    (0.82, 0.40),   # Blanchimont apex
    (0.84, 0.37),
    (0.85, 0.34),
    
    # === BUS STOP CHICANE ===
    (0.86, 0.32),   # Braking zone
    (0.87, 0.30),
    (0.89, 0.29),   # Bus stop 1
    (0.91, 0.28),
    (0.92, 0.27),   # Bus stop 2
    (0.92, 0.26),
    
    # === BACK TO START/FINISH ===
    (0.92, 0.25),   # Complete lap
]

# Segment boundaries (indices into SPA_TRACK_POINTS that mark segment transitions)
# Maps to the 20 segments defined in track.py
SEGMENT_INDICES = [
    0,    # 0: La Source Hairpin start
    9,    # 1: Pit Straight Exit / Down to Eau Rouge
    15,   # 2: Eau Rouge Entry
    18,   # 3: Eau Rouge Apex
    24,   # 4: Raidillon
    30,   # 5: Kemmel Straight
    36,   # 6: Les Combes 1
    42,   # 7: Les Combes 2
    45,   # 8: Malmedy
    52,   # 9: Rivage
    56,   # 10: Rivage Exit
    64,   # 11: Pouhon Entry
    72,   # 12: Pouhon Apex
    76,   # 13: Fagnes
    80,   # 14: Campus Entry
    84,   # 15: Campus Exit  
    90,   # 16: Stavelot
    100,  # 17: Blanchimont
    106,  # 18: Bus Stop Chicane
    110,  # 19: Start/Finish Straight (end = start)
]

# Segment names for labeling
SEGMENT_NAMES = [
    "La Source",
    "Pit Exit",
    "Eau Rouge",
    "Raidillon",
    "Raidillon",
    "Kemmel",
    "Les Combes",
    "Les Combes",
    "Malmedy",
    "Rivage",
    "Rivage Exit",
    "Pouhon",
    "Pouhon",
    "Fagnes",
    "Campus",
    "Campus",
    "Stavelot",
    "Blanchimont",
    "Bus Stop",
    "Start/Finish"
]

# Key corner labels and positions
CORNER_LABELS = [
    {"name": "La Source", "pos": (0.75, 0.18), "anchor": "bottom"},
    {"name": "Eau Rouge", "pos": (0.55, 0.46), "anchor": "top"},
    {"name": "Raidillon", "pos": (0.40, 0.28), "anchor": "right"},
    {"name": "Les Combes", "pos": (0.06, 0.14), "anchor": "right"},
    {"name": "Rivage", "pos": (0.05, 0.36), "anchor": "right"},
    {"name": "Pouhon", "pos": (0.23, 0.60), "anchor": "right"},
    {"name": "Stavelot", "pos": (0.67, 0.64), "anchor": "left"},
    {"name": "Blanchimont", "pos": (0.84, 0.40), "anchor": "left"},
    {"name": "Bus Stop", "pos": (0.94, 0.28), "anchor": "left"},
]

# Sector boundaries (segment indices)
SECTOR_1_END = 5      # After Raidillon
SECTOR_2_END = 14     # After Fagnes
SECTOR_3_END = 20     # Finish


def get_track_coordinates() -> Tuple[List[float], List[float]]:
    """Get track x and y coordinates as separate lists."""
    x = [p[0] for p in SPA_TRACK_POINTS]
    y = [p[1] for p in SPA_TRACK_POINTS]
    return x, y


def get_position_at_progress(progress: float) -> Tuple[float, float]:
    """
    Get track position (x, y) at given lap progress (0-1).
    
    Uses linear interpolation between track points.
    """
    n_points = len(SPA_TRACK_POINTS)
    
    # Map progress to point index
    float_idx = progress * (n_points - 1)
    idx = int(float_idx)
    t = float_idx - idx
    
    if idx >= n_points - 1:
        return SPA_TRACK_POINTS[-1]
    
    # Interpolate between points
    x1, y1 = SPA_TRACK_POINTS[idx]
    x2, y2 = SPA_TRACK_POINTS[idx + 1]
    
    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)
    
    return (x, y)


def get_segment_position(segment_idx: int, segment_progress: float = 0.5) -> Tuple[float, float]:
    """
    Get position for a specific segment.
    
    Args:
        segment_idx: Segment index (0-19)
        segment_progress: Progress within segment (0-1)
        
    Returns:
        (x, y) position tuple
    """
    n_segments = len(SEGMENT_INDICES)
    n_points = len(SPA_TRACK_POINTS)
    
    if segment_idx >= n_segments:
        segment_idx = n_segments - 1
    
    # Calculate overall progress
    segment_fraction = segment_idx / n_segments
    next_segment_fraction = (segment_idx + 1) / n_segments
    
    progress = segment_fraction + segment_progress * (next_segment_fraction - segment_fraction)
    
    return get_position_at_progress(progress)
