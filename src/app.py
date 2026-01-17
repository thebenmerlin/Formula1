"""
F1 Design Comparison Dashboard

A Streamlit visualization layer for comparing multiple F1-style vehicle designs
at Spa-Francorchamps. Consumes precomputed ML predictions for animation.

NOT a race simulator. A design trade-space exploration tool.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import json
import os
import subprocess
import sys
from typing import Dict, List, Tuple, Optional

# Page configuration
st.set_page_config(
    page_title="F1 Design Comparison Dashboard",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "outputs", "streamlit_data.json")

# Design colors
DESIGN_COLORS = {
    0: "#FF1801",  # Ferrari Red
    1: "#00D2BE",  # Mercedes Teal
    2: "#0600EF",  # Red Bull Blue
    3: "#FF8700",  # McLaren Orange
    4: "#006F62",  # Aston Martin Green
}

# Spa track coordinates (normalized 0-1, approximate circuit shape)
SPA_TRACK_COORDS = [
    # Start/Finish
    (0.85, 0.15),  # Start
    # La Source
    (0.88, 0.18), (0.86, 0.22), (0.82, 0.20),
    # Down to Eau Rouge
    (0.75, 0.18), (0.68, 0.15),
    # Eau Rouge / Raidillon
    (0.62, 0.12), (0.58, 0.15), (0.55, 0.22),
    # Kemmel Straight
    (0.50, 0.30), (0.42, 0.40),
    # Les Combes
    (0.35, 0.48), (0.30, 0.52), (0.32, 0.56),
    # Malmedy, Rivage
    (0.28, 0.60), (0.25, 0.58), (0.22, 0.55),
    # Down through forest
    (0.18, 0.52), (0.15, 0.55),
    # Pouhon
    (0.12, 0.60), (0.10, 0.65), (0.12, 0.70),
    # Fagnes, Campus
    (0.18, 0.75), (0.25, 0.78), (0.32, 0.80),
    # Stavelot
    (0.40, 0.82), (0.48, 0.80),
    # Blanchimont
    (0.58, 0.75), (0.68, 0.65), (0.75, 0.55),
    # Bus Stop Chicane
    (0.80, 0.45), (0.82, 0.38), (0.80, 0.32), (0.83, 0.28),
    # Back to Start/Finish
    (0.85, 0.22), (0.85, 0.15),
]


@st.cache_data
def load_data(path: str = DATA_PATH) -> Optional[Dict]:
    """Load precomputed prediction data."""
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)


def create_track_figure(
    selected_designs: List[Dict],
    animation_progress: float = 1.0
) -> go.Figure:
    """
    Create Plotly figure with Spa track and design markers.
    
    Args:
        selected_designs: List of design prediction dicts
        animation_progress: 0-1 progress through the lap
        
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    
    # Draw track outline
    track_x = [p[0] for p in SPA_TRACK_COORDS]
    track_y = [p[1] for p in SPA_TRACK_COORDS]
    
    fig.add_trace(go.Scatter(
        x=track_x, y=track_y,
        mode='lines',
        line=dict(color='#444444', width=12),
        name='Track',
        hoverinfo='skip'
    ))
    
    # Track center line
    fig.add_trace(go.Scatter(
        x=track_x, y=track_y,
        mode='lines',
        line=dict(color='#888888', width=1, dash='dash'),
        name='Center Line',
        hoverinfo='skip'
    ))
    
    # Sector markers
    sector_positions = [
        (0.55, 0.22, "S1"),  # End of Sector 1 (after Raidillon)
        (0.18, 0.75, "S2"),  # End of Sector 2 (after Fagnes)
        (0.85, 0.15, "S3"),  # End of Sector 3 (Finish)
    ]
    
    for x, y, label in sector_positions:
        fig.add_annotation(
            x=x, y=y,
            text=label,
            showarrow=False,
            font=dict(size=10, color='white'),
            bgcolor='rgba(0,0,0,0.6)',
            borderpad=3
        )
    
    # Add design markers
    n_track_points = len(SPA_TRACK_COORDS)
    
    for i, design in enumerate(selected_designs):
        # Calculate position based on cumulative time
        total_time = design['total_lap_time']
        current_time = animation_progress * total_time
        
        # Find which segment we're in
        cumulative_time = 0
        segment_progress = 0
        segment_idx = 0
        
        for seg in design['segments']:
            if cumulative_time + seg['time_in_segment'] >= current_time:
                # We're in this segment
                time_into_segment = current_time - cumulative_time
                segment_progress = time_into_segment / seg['time_in_segment']
                break
            cumulative_time += seg['time_in_segment']
            segment_idx += 1
        
        # Map segment to track position
        segment_fraction = (segment_idx + segment_progress) / len(design['segments'])
        track_point_idx = int(segment_fraction * (n_track_points - 1))
        track_point_idx = min(track_point_idx, n_track_points - 2)
        
        # Interpolate between track points
        t = (segment_fraction * (n_track_points - 1)) - track_point_idx
        x1, y1 = SPA_TRACK_COORDS[track_point_idx]
        x2, y2 = SPA_TRACK_COORDS[min(track_point_idx + 1, n_track_points - 1)]
        
        pos_x = x1 + t * (x2 - x1)
        pos_y = y1 + t * (y2 - y1)
        
        color = DESIGN_COLORS.get(i, '#FFFFFF')
        
        fig.add_trace(go.Scatter(
            x=[pos_x], y=[pos_y],
            mode='markers+text',
            marker=dict(size=18, color=color, symbol='circle',
                       line=dict(color='white', width=2)),
            text=[design['design_name'][:3]],
            textposition='top center',
            textfont=dict(size=10, color='white'),
            name=design['design_name'],
            hovertemplate=(
                f"<b>{design['design_name']}</b><br>"
                f"Time: {current_time:.2f}s<br>"
                f"Segment: {segment_idx + 1}/20<br>"
                "<extra></extra>"
            )
        ))
    
    # Layout
    fig.update_layout(
        title=dict(
            text="Spa-Francorchamps Circuit",
            font=dict(size=18, color='white'),
            x=0.5
        ),
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.15,
            xanchor='center',
            x=0.5,
            font=dict(color='white')
        ),
        xaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False,
            range=[-0.05, 1.05]
        ),
        yaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False,
            range=[-0.05, 1.05], scaleanchor='x'
        ),
        plot_bgcolor='#1a1a1a',
        paper_bgcolor='#1a1a1a',
        margin=dict(l=20, r=20, t=50, b=50),
        height=500
    )
    
    return fig


def create_timing_grid(selected_designs: List[Dict], track_data: Dict) -> pd.DataFrame:
    """Create timing grid DataFrame with sector times."""
    data = []
    
    # Get sector boundaries from track data
    sector_1_segs = [s for s in track_data['segments'] if s['sector'] == 1]
    sector_2_segs = [s for s in track_data['segments'] if s['sector'] == 2]
    sector_3_segs = [s for s in track_data['segments'] if s['sector'] == 3]
    
    for design in selected_designs:
        sectors = {'S1': 0, 'S2': 0, 'S3': 0}
        
        for seg in design['segments']:
            if seg['sector'] == 1:
                sectors['S1'] += seg['time_in_segment']
            elif seg['sector'] == 2:
                sectors['S2'] += seg['time_in_segment']
            else:
                sectors['S3'] += seg['time_in_segment']
        
        data.append({
            'Design': design['design_name'],
            'Sector 1': f"{sectors['S1']:.3f}s",
            'Sector 2': f"{sectors['S2']:.3f}s",
            'Sector 3': f"{sectors['S3']:.3f}s",
            'Lap Time': f"{design['total_lap_time']:.3f}s",
            'Energy (MJ)': f"{design['total_energy_used']:.2f}",
            'Thermal Risk': f"{design['peak_thermal_risk']:.3f}"
        })
    
    return pd.DataFrame(data)


def create_energy_plot(selected_designs: List[Dict]) -> go.Figure:
    """Create cumulative energy plot vs distance."""
    fig = go.Figure()
    
    for i, design in enumerate(selected_designs):
        distances = [seg['cumulative_distance'] for seg in design['segments']]
        energies = [seg['cumulative_energy'] for seg in design['segments']]
        
        color = DESIGN_COLORS.get(i, '#FFFFFF')
        
        fig.add_trace(go.Scatter(
            x=distances, y=energies,
            mode='lines+markers',
            name=design['design_name'],
            line=dict(color=color, width=2),
            marker=dict(size=4)
        ))
    
    fig.update_layout(
        title="Cumulative Energy Deployment",
        xaxis_title="Distance (m)",
        yaxis_title="Energy (MJ)",
        template='plotly_dark',
        height=300,
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(orientation='h', y=-0.25)
    )
    
    return fig


def create_speed_plot(selected_designs: List[Dict]) -> go.Figure:
    """Create speed profile plot."""
    fig = go.Figure()
    
    for i, design in enumerate(selected_designs):
        distances = [seg['cumulative_distance'] for seg in design['segments']]
        speeds = [seg['avg_speed'] for seg in design['segments']]
        
        color = DESIGN_COLORS.get(i, '#FFFFFF')
        
        fig.add_trace(go.Scatter(
            x=distances, y=speeds,
            mode='lines',
            name=design['design_name'],
            line=dict(color=color, width=2),
            fill='tozeroy',
            fillcolor=color.replace(')', ', 0.1)').replace('rgb', 'rgba') if 'rgb' in color else f'{color}22'
        ))
    
    fig.update_layout(
        title="Speed Profile",
        xaxis_title="Distance (m)",
        yaxis_title="Speed (km/h)",
        template='plotly_dark',
        height=300,
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(orientation='h', y=-0.25)
    )
    
    return fig


def create_delta_plot(selected_designs: List[Dict], baseline_idx: int = 0) -> go.Figure:
    """Create lap time delta plot vs baseline."""
    if len(selected_designs) < 2:
        return None
    
    fig = go.Figure()
    
    baseline = selected_designs[baseline_idx]
    baseline_cumtime = [seg['cumulative_time'] for seg in baseline['segments']]
    baseline_distances = [seg['cumulative_distance'] for seg in baseline['segments']]
    
    for i, design in enumerate(selected_designs):
        if i == baseline_idx:
            continue
            
        design_cumtime = [seg['cumulative_time'] for seg in design['segments']]
        
        # Delta is positive when slower than baseline
        deltas = [dt - bt for dt, bt in zip(design_cumtime, baseline_cumtime)]
        distances = [seg['cumulative_distance'] for seg in design['segments']]
        
        color = DESIGN_COLORS.get(i, '#FFFFFF')
        
        fig.add_trace(go.Scatter(
            x=distances, y=deltas,
            mode='lines',
            name=f"Δ vs {baseline['design_name'][:8]}",
            line=dict(color=color, width=2)
        ))
    
    # Zero line
    fig.add_hline(y=0, line_dash='dash', line_color='white', opacity=0.5)
    
    fig.update_layout(
        title=f"Time Delta vs {baseline['design_name']}",
        xaxis_title="Distance (m)",
        yaxis_title="Delta (s, +ve = slower)",
        template='plotly_dark',
        height=300,
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(orientation='h', y=-0.25)
    )
    
    return fig


def regenerate_data(designs: List[Dict]) -> bool:
    """
    Regenerate predictions by calling segment_predictor.
    
    Args:
        designs: List of design parameter dicts
        
    Returns:
        True if successful
    """
    try:
        # Run the segment predictor module
        result = subprocess.run(
            [sys.executable, '-m', 'src.analysis.segment_predictor'],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=60
        )
        return result.returncode == 0
    except Exception as e:
        st.error(f"Failed to regenerate data: {e}")
        return False


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown("""
    <h1 style='text-align: center; color: #FF1801;'>
        🏎️ F1 Design Comparison Dashboard
    </h1>
    <p style='text-align: center; color: #888;'>
        ML-Predicted Performance at Spa-Francorchamps | 2026-Style Hybrid Constraints
    </p>
    """, unsafe_allow_html=True)
    
    # Load data
    data = load_data()
    
    if data is None:
        st.error("⚠️ No prediction data found. Run `python -m src.analysis.segment_predictor` first.")
        st.stop()
    
    track_data = data['track']
    predictions = data['predictions']
    summary = data['summary']
    
    # Sidebar
    st.sidebar.header("⚙️ Design Selection")
    
    design_names = [p['design_name'] for p in predictions]
    selected_names = st.sidebar.multiselect(
        "Select Designs to Compare (2-4)",
        options=design_names,
        default=design_names[:3],
        max_selections=4
    )
    
    if len(selected_names) < 2:
        st.warning("Please select at least 2 designs to compare.")
        st.stop()
    
    # Get selected designs
    selected_designs = [p for p in predictions if p['design_name'] in selected_names]
    
    # Animation control
    st.sidebar.markdown("---")
    st.sidebar.header("🎬 Animation")
    
    animation_progress = st.sidebar.slider(
        "Lap Progress",
        min_value=0.0,
        max_value=1.0,
        value=1.0,
        step=0.01,
        format="%.0f%%"
    )
    
    # Auto-animate checkbox
    auto_animate = st.sidebar.checkbox("Auto-animate", value=False)
    
    if auto_animate:
        st.sidebar.info("Animation updates on slider change")
    
    # Design parameters (read-only display)
    st.sidebar.markdown("---")
    st.sidebar.header("📐 Design Parameters")
    
    with st.sidebar.expander("View Parameters", expanded=False):
        for design in selected_designs:
            st.markdown(f"**{design['design_name']}**")
            params = design['design_params']
            cols = st.columns(2)
            cols[0].metric("Mass", f"{params['m']:.0f} kg")
            cols[1].metric("C_L", f"{params['C_L']:.2f}")
            cols = st.columns(2)
            cols[0].metric("C_D", f"{params['C_D']:.2f}")
            cols[1].metric("α_elec", f"{params['alpha_elec']:.2f}")
            st.markdown("---")
    
    # Regenerate button
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Regenerate Predictions", use_container_width=True):
        with st.spinner("Regenerating predictions..."):
            if regenerate_data([d['design_params'] for d in selected_designs]):
                st.sidebar.success("Data regenerated!")
                st.cache_data.clear()
                st.rerun()
            else:
                st.sidebar.error("Regeneration failed")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Track visualization
        track_fig = create_track_figure(selected_designs, animation_progress)
        st.plotly_chart(track_fig, use_container_width=True)
    
    with col2:
        # Quick stats
        st.markdown("### 📊 Quick Comparison")
        
        for i, design in enumerate(selected_designs):
            color = DESIGN_COLORS.get(i, '#FFFFFF')
            delta = design['total_lap_time'] - selected_designs[0]['total_lap_time']
            delta_str = f"+{delta:.3f}s" if delta > 0 else f"{delta:.3f}s"
            
            st.markdown(f"""
            <div style='background: linear-gradient(90deg, {color}33, transparent); 
                        padding: 10px; border-radius: 5px; margin-bottom: 10px;
                        border-left: 4px solid {color};'>
                <b style='color: {color};'>{design['design_name']}</b><br>
                <span style='font-size: 1.5em; color: white;'>{design['total_lap_time']:.3f}s</span>
                <span style='color: {"#4CAF50" if delta <= 0 else "#F44336"};'> ({delta_str})</span><br>
                <small style='color: #888;'>Energy: {design['total_energy_used']:.2f} MJ</small>
            </div>
            """, unsafe_allow_html=True)
    
    # Timing grid
    st.markdown("---")
    st.markdown("### ⏱️ Timing Grid")
    
    timing_df = create_timing_grid(selected_designs, track_data)
    
    # Style the dataframe
    st.dataframe(
        timing_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Design": st.column_config.TextColumn("Design", width="medium"),
            "Lap Time": st.column_config.TextColumn("Lap Time", width="small"),
        }
    )
    
    # Engineering plots
    st.markdown("---")
    st.markdown("### 📈 Engineering Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        energy_fig = create_energy_plot(selected_designs)
        st.plotly_chart(energy_fig, use_container_width=True)
    
    with col2:
        speed_fig = create_speed_plot(selected_designs)
        st.plotly_chart(speed_fig, use_container_width=True)
    
    # Delta plot
    delta_fig = create_delta_plot(selected_designs, baseline_idx=0)
    if delta_fig:
        st.plotly_chart(delta_fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <p style='text-align: center; color: #666; font-size: 0.8em;'>
        F1 ML Design Trade-Space Exploration System | 
        Physics-Informed Synthetic Data | ML Surrogate Models<br>
        <i>This is a design analysis tool, not a race simulator.</i>
    </p>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
