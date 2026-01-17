"""
F1 Design Comparison Dashboard

Professional engineering visualization for F1-style vehicle design comparison.
Styled after real F1 engineering workstations - light mode, data-focused, clean.

Consumes precomputed ML predictions only. NOT a race simulator.
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

# Page configuration - Professional light theme
st.set_page_config(
    page_title="F1 Design Analysis | Spa-Francorchamps",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling
st.markdown("""
<style>
    /* Light mode professional theme */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #1e1e1e 0%, #2d2d2d 100%);
        color: white;
        padding: 20px 30px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 24px;
        font-weight: 600;
        letter-spacing: -0.5px;
    }
    
    .main-header p {
        margin: 5px 0 0 0;
        color: #a0a0a0;
        font-size: 13px;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    .metric-card .label {
        font-size: 11px;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    
    .metric-card .value {
        font-size: 28px;
        font-weight: 700;
        color: #1a1a1a;
        font-family: 'Monaco', 'Consolas', monospace;
    }
    
    .metric-card .delta {
        font-size: 13px;
        font-weight: 500;
        margin-top: 4px;
    }
    
    .delta-positive { color: #dc3545; }
    .delta-negative { color: #28a745; }
    .delta-neutral { color: #666; }
    
    /* Section headers */
    .section-header {
        font-size: 14px;
        font-weight: 600;
        color: #333;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        padding-bottom: 8px;
        border-bottom: 2px solid #e0e0e0;
        margin: 20px 0 15px 0;
    }
    
    /* Data table styling */
    .timing-grid {
        background: white;
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #e0e0e0;
    }
    
    .timing-grid table {
        width: 100%;
        border-collapse: collapse;
    }
    
    .timing-grid th {
        background: #f5f5f5;
        padding: 12px 16px;
        text-align: left;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #666;
        border-bottom: 1px solid #e0e0e0;
    }
    
    .timing-grid td {
        padding: 14px 16px;
        font-size: 14px;
        font-family: 'Monaco', 'Consolas', monospace;
        border-bottom: 1px solid #f0f0f0;
    }
    
    .timing-grid tr:last-child td {
        border-bottom: none;
    }
    
    /* Design indicator */
    .design-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 2px;
        margin-right: 8px;
        vertical-align: middle;
    }
    
    /* Fastest time highlight */
    .fastest {
        color: #9b59b6;
        font-weight: 700;
    }
    
    /* Sidebar styling */
    .sidebar-section {
        background: white;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 16px;
        border: 1px solid #e0e0e0;
    }
    
    .sidebar-title {
        font-size: 12px;
        font-weight: 600;
        color: #333;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 12px;
    }
    
    /* Parameter display */
    .param-row {
        display: flex;
        justify-content: space-between;
        padding: 6px 0;
        border-bottom: 1px solid #f0f0f0;
        font-size: 13px;
    }
    
    .param-label {
        color: #666;
    }
    
    .param-value {
        font-weight: 600;
        font-family: 'Monaco', 'Consolas', monospace;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #c1c1c1;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Constants
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "outputs", "streamlit_data.json")

# Professional F1 team colors
DESIGN_COLORS = [
    "#E10600",  # Red (Ferrari-inspired)
    "#00A19C",  # Teal (Mercedes-inspired)  
    "#0600EF",  # Blue (Red Bull-inspired)
    "#FF8000",  # Orange (McLaren-inspired)
    "#006F62",  # Green (Aston Martin-inspired)
]

# Import accurate Spa track coordinates
try:
    from src.spa_track_viz import (
        SPA_TRACK_POINTS, CORNER_LABELS, 
        get_position_at_progress, get_track_coordinates
    )
except ModuleNotFoundError:
    from spa_track_viz import (
        SPA_TRACK_POINTS, CORNER_LABELS, 
        get_position_at_progress, get_track_coordinates
    )


def create_track_map(
    selected_designs: List[Dict],
    animation_progress: float = 1.0
) -> go.Figure:
    """
    Create accurate Spa-Francorchamps track map with design markers.
    
    Args:
        selected_designs: List of design prediction dicts
        animation_progress: 0-1 progress through the lap
        
    Returns:
        Plotly Figure with track and markers
    """
    fig = go.Figure()
    
    # Get track coordinates
    track_x, track_y = get_track_coordinates()
    
    # Draw track outline (thick gray)
    fig.add_trace(go.Scatter(
        x=track_x, y=track_y,
        mode='lines',
        line=dict(color='#d0d0d0', width=16),
        hoverinfo='skip',
        showlegend=False
    ))
    
    # Draw track racing line (thin dark)
    fig.add_trace(go.Scatter(
        x=track_x, y=track_y,
        mode='lines',
        line=dict(color='#666666', width=2),
        hoverinfo='skip',
        showlegend=False
    ))
    
    # Add corner labels
    for corner in CORNER_LABELS:
        fig.add_annotation(
            x=corner['pos'][0],
            y=corner['pos'][1],
            text=corner['name'],
            showarrow=False,
            font=dict(size=9, color='#666'),
            xanchor='left' if corner['anchor'] == 'left' else 'right' if corner['anchor'] == 'right' else 'center',
            yanchor='top' if corner['anchor'] == 'top' else 'bottom' if corner['anchor'] == 'bottom' else 'middle'
        )
    
    # Add sector markers
    sector_markers = [
        {'pos': (0.42, 0.28), 'label': 'S1'},  # After Raidillon
        {'pos': (0.29, 0.72), 'label': 'S2'},  # After Fagnes  
        {'pos': (0.92, 0.25), 'label': 'S3/F'},  # Finish
    ]
    
    for marker in sector_markers:
        fig.add_annotation(
            x=marker['pos'][0],
            y=marker['pos'][1],
            text=marker['label'],
            showarrow=False,
            font=dict(size=10, color='white', family='Arial Black'),
            bgcolor='#333',
            borderpad=3
        )
    
    # Add design markers based on animation progress
    for i, design in enumerate(selected_designs):
        # Calculate position based on cumulative time
        total_time = design['total_lap_time']
        current_time = animation_progress * total_time
        
        # Find which segment and progress we're in
        cumulative_time = 0
        segment_idx = 0
        segment_progress = 0
        
        for seg in design['segments']:
            if cumulative_time + seg['time_in_segment'] >= current_time:
                time_into_segment = current_time - cumulative_time
                segment_progress = time_into_segment / seg['time_in_segment']
                break
            cumulative_time += seg['time_in_segment']
            segment_idx += 1
        
        # Map segment to overall lap progress
        lap_progress = (segment_idx + segment_progress) / len(design['segments'])
        lap_progress = min(lap_progress, 0.999)  # Prevent overflow
        
        # Get position from track
        pos_x, pos_y = get_position_at_progress(lap_progress)
        
        color = DESIGN_COLORS[i % len(DESIGN_COLORS)]
        
        # Add marker
        fig.add_trace(go.Scatter(
            x=[pos_x], y=[pos_y],
            mode='markers',
            marker=dict(
                size=16, 
                color=color,
                symbol='circle',
                line=dict(color='white', width=2)
            ),
            name=design['design_name'],
            hovertemplate=(
                f"<b>{design['design_name']}</b><br>"
                f"Progress: {animation_progress*100:.0f}%<br>"
                f"Time: {current_time:.2f}s<br>"
                "<extra></extra>"
            )
        ))
    
    # Layout for professional look
    fig.update_layout(
        title=dict(
            text="Circuit: Spa-Francorchamps",
            font=dict(size=14, color='#333'),
            x=0.5
        ),
        xaxis=dict(
            showgrid=False, 
            zeroline=False, 
            showticklabels=False,
            range=[-0.05, 1.05],
            scaleanchor='y'
        ),
        yaxis=dict(
            showgrid=False, 
            zeroline=False, 
            showticklabels=False,
            range=[-0.02, 0.85]
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.1,
            xanchor='center',
            x=0.5,
            font=dict(size=10)
        ),
        margin=dict(l=10, r=10, t=40, b=40),
        height=450
    )
    
    return fig


@st.cache_data
def load_data(path: str = DATA_PATH) -> Optional[Dict]:
    """Load precomputed prediction data."""
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)


def format_time(seconds: float) -> str:
    """Format time in F1 style (mm:ss.sss)."""
    mins = int(seconds // 60)
    secs = seconds % 60
    if mins > 0:
        return f"{mins}:{secs:06.3f}"
    return f"{secs:.3f}"


def format_delta(delta: float) -> Tuple[str, str]:
    """Format delta time with sign and CSS class."""
    if abs(delta) < 0.001:
        return "—", "delta-neutral"
    elif delta > 0:
        return f"+{delta:.3f}", "delta-positive"
    else:
        return f"{delta:.3f}", "delta-negative"


def render_header():
    """Render professional header."""
    st.markdown("""
    <div class="main-header">
        <h1>📊 Vehicle Design Analysis</h1>
        <p>Circuit: Spa-Francorchamps | 7.004 km | 2026 Hybrid Regulations</p>
    </div>
    """, unsafe_allow_html=True)


def render_timing_grid(selected_designs: List[Dict], track_data: Dict) -> None:
    """Render professional F1-style timing grid."""
    
    # Calculate sector times
    data = []
    for i, design in enumerate(selected_designs):
        sectors = {'S1': 0, 'S2': 0, 'S3': 0}
        for seg in design['segments']:
            if seg['sector'] == 1:
                sectors['S1'] += seg['time_in_segment']
            elif seg['sector'] == 2:
                sectors['S2'] += seg['time_in_segment']
            else:
                sectors['S3'] += seg['time_in_segment']
        
        data.append({
            'idx': i,
            'name': design['design_name'],
            'S1': sectors['S1'],
            'S2': sectors['S2'],
            'S3': sectors['S3'],
            'lap': design['total_lap_time'],
            'energy': design['total_energy_used'],
        })
    
    # Find fastest times
    if data:
        min_s1 = min(d['S1'] for d in data)
        min_s2 = min(d['S2'] for d in data)
        min_s3 = min(d['S3'] for d in data)
        min_lap = min(d['lap'] for d in data)
    
    # Build HTML table
    html = '<div class="timing-grid"><table>'
    html += '''
    <tr>
        <th>Setup</th>
        <th>Sector 1</th>
        <th>Sector 2</th>
        <th>Sector 3</th>
        <th>Lap Time</th>
        <th>Gap</th>
        <th>Energy (MJ)</th>
    </tr>
    '''
    
    baseline_lap = data[0]['lap'] if data else 0
    
    for d in sorted(data, key=lambda x: x['lap']):
        color = DESIGN_COLORS[d['idx'] % len(DESIGN_COLORS)]
        gap = d['lap'] - baseline_lap
        gap_str, gap_class = format_delta(gap)
        
        # Highlight fastest sectors in purple
        s1_class = 'fastest' if abs(d['S1'] - min_s1) < 0.001 else ''
        s2_class = 'fastest' if abs(d['S2'] - min_s2) < 0.001 else ''
        s3_class = 'fastest' if abs(d['S3'] - min_s3) < 0.001 else ''
        lap_class = 'fastest' if abs(d['lap'] - min_lap) < 0.001 else ''
        
        html += f'''
        <tr>
            <td>
                <span class="design-indicator" style="background:{color}"></span>
                {d['name']}
            </td>
            <td class="{s1_class}">{d['S1']:.3f}</td>
            <td class="{s2_class}">{d['S2']:.3f}</td>
            <td class="{s3_class}">{d['S3']:.3f}</td>
            <td class="{lap_class}">{format_time(d['lap'])}</td>
            <td class="{gap_class}">{gap_str}</td>
            <td>{d['energy']:.2f}</td>
        </tr>
        '''
    
    html += '</table></div>'
    st.markdown(html, unsafe_allow_html=True)


def render_summary_metrics(selected_designs: List[Dict]) -> None:
    """Render summary metrics in professional card format."""
    if not selected_designs:
        return
    
    # Find fastest
    fastest = min(selected_designs, key=lambda x: x['total_lap_time'])
    most_efficient = min(selected_designs, key=lambda x: x['total_energy_used'])
    
    cols = st.columns(4)
    
    with cols[0]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Fastest Lap</div>
            <div class="value">{format_time(fastest['total_lap_time'])}</div>
            <div class="delta delta-neutral">{fastest['design_name']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[1]:
        spread = max(d['total_lap_time'] for d in selected_designs) - fastest['total_lap_time']
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Performance Spread</div>
            <div class="value">{spread:.3f}s</div>
            <div class="delta delta-neutral">Fastest to Slowest</div>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[2]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Most Efficient</div>
            <div class="value">{most_efficient['total_energy_used']:.2f} MJ</div>
            <div class="delta delta-neutral">{most_efficient['design_name']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[3]:
        avg_energy = np.mean([d['total_energy_used'] for d in selected_designs])
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Avg Energy Deploy</div>
            <div class="value">{avg_energy:.2f} MJ</div>
            <div class="delta delta-neutral">Across {len(selected_designs)} setups</div>
        </div>
        """, unsafe_allow_html=True)


def create_speed_trace(selected_designs: List[Dict]) -> go.Figure:
    """Create professional speed trace plot."""
    fig = go.Figure()
    
    for i, design in enumerate(selected_designs):
        distances = [0]
        speeds = [design['segments'][0]['entry_speed']]
        
        for seg in design['segments']:
            distances.append(seg['cumulative_distance'] + seg['segment_length'])
            speeds.append(seg['exit_speed'])
        
        color = DESIGN_COLORS[i % len(DESIGN_COLORS)]
        
        fig.add_trace(go.Scatter(
            x=distances,
            y=speeds,
            mode='lines',
            name=design['design_name'],
            line=dict(color=color, width=1.5),
        ))
    
    fig.update_layout(
        title=dict(text="Speed Trace", font=dict(size=14, color='#333')),
        xaxis=dict(
            title="Distance (m)",
            gridcolor='#e0e0e0',
            linecolor='#ccc',
            tickfont=dict(size=10)
        ),
        yaxis=dict(
            title="Speed (km/h)",
            gridcolor='#e0e0e0',
            linecolor='#ccc',
            tickfont=dict(size=10)
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='left',
            x=0,
            font=dict(size=10)
        ),
        margin=dict(l=50, r=20, t=50, b=50),
        height=300
    )
    
    return fig


def create_energy_trace(selected_designs: List[Dict]) -> go.Figure:
    """Create cumulative energy deployment plot."""
    fig = go.Figure()
    
    for i, design in enumerate(selected_designs):
        distances = [seg['cumulative_distance'] for seg in design['segments']]
        energies = [seg['cumulative_energy'] for seg in design['segments']]
        
        color = DESIGN_COLORS[i % len(DESIGN_COLORS)]
        
        fig.add_trace(go.Scatter(
            x=distances,
            y=energies,
            mode='lines',
            name=design['design_name'],
            line=dict(color=color, width=1.5),
        ))
    
    fig.update_layout(
        title=dict(text="Energy Deployment", font=dict(size=14, color='#333')),
        xaxis=dict(
            title="Distance (m)",
            gridcolor='#e0e0e0',
            linecolor='#ccc',
            tickfont=dict(size=10)
        ),
        yaxis=dict(
            title="Cumulative Energy (MJ)",
            gridcolor='#e0e0e0',
            linecolor='#ccc',
            tickfont=dict(size=10)
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='left',
            x=0,
            font=dict(size=10)
        ),
        margin=dict(l=50, r=20, t=50, b=50),
        height=300
    )
    
    return fig


def create_delta_trace(selected_designs: List[Dict]) -> go.Figure:
    """Create time delta plot vs baseline."""
    if len(selected_designs) < 2:
        return None
    
    fig = go.Figure()
    baseline = selected_designs[0]
    baseline_times = {seg['segment_id']: seg['cumulative_time'] for seg in baseline['segments']}
    
    for i, design in enumerate(selected_designs[1:], 1):
        distances = []
        deltas = []
        
        for seg in design['segments']:
            distances.append(seg['cumulative_distance'])
            baseline_time = baseline_times.get(seg['segment_id'], seg['cumulative_time'])
            deltas.append(seg['cumulative_time'] - baseline_time)
        
        color = DESIGN_COLORS[i % len(DESIGN_COLORS)]
        
        fig.add_trace(go.Scatter(
            x=distances,
            y=deltas,
            mode='lines',
            name=f"vs {baseline['design_name']}",
            line=dict(color=color, width=1.5),
            fill='tozeroy',
            fillcolor=f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.1)"
        ))
    
    # Zero reference line
    fig.add_hline(y=0, line_dash='dash', line_color='#999', line_width=1)
    
    fig.update_layout(
        title=dict(text=f"Time Delta vs {baseline['design_name']}", font=dict(size=14, color='#333')),
        xaxis=dict(
            title="Distance (m)",
            gridcolor='#e0e0e0',
            linecolor='#ccc',
            tickfont=dict(size=10)
        ),
        yaxis=dict(
            title="Delta (s)",
            gridcolor='#e0e0e0',
            linecolor='#ccc',
            tickfont=dict(size=10),
            zeroline=True,
            zerolinecolor='#666'
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='left',
            x=0,
            font=dict(size=10)
        ),
        margin=dict(l=50, r=20, t=50, b=50),
        height=250
    )
    
    return fig


def render_design_params(design: Dict) -> None:
    """Render design parameters in sidebar."""
    params = design['design_params']
    
    st.markdown(f"""
    <div style="font-weight: 600; margin-bottom: 8px;">{design['design_name']}</div>
    <div class="param-row">
        <span class="param-label">Mass</span>
        <span class="param-value">{params['m']:.0f} kg</span>
    </div>
    <div class="param-row">
        <span class="param-label">Aero Load (C_L)</span>
        <span class="param-value">{params['C_L']:.2f}</span>
    </div>
    <div class="param-row">
        <span class="param-label">Aero Drag (C_D)</span>
        <span class="param-value">{params['C_D']:.2f}</span>
    </div>
    <div class="param-row">
        <span class="param-label">Electric Frac.</span>
        <span class="param-value">{params['alpha_elec']:.0%}</span>
    </div>
    <div class="param-row">
        <span class="param-label">Energy Deploy</span>
        <span class="param-value">{params['E_deploy']:.1f} MJ</span>
    </div>
    <div class="param-row" style="border-bottom: none;">
        <span class="param-label">Cooling Factor</span>
        <span class="param-value">{params['gamma_cool']:.2f}</span>
    </div>
    """, unsafe_allow_html=True)


def main():
    """Main application."""
    
    # Load data
    data = load_data()
    
    if data is None:
        st.error("⚠️ No prediction data found. Run `python -m src.analysis.segment_predictor` first.")
        st.stop()
    
    track_data = data['track']
    predictions = data['predictions']
    
    # Header
    render_header()
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-title">Setup Selection</div>', unsafe_allow_html=True)
        
        design_names = [p['design_name'] for p in predictions]
        selected_names = st.multiselect(
            "Compare Designs",
            options=design_names,
            default=design_names[:3],
            max_selections=5,
            label_visibility="collapsed"
        )
        
        if len(selected_names) < 2:
            st.warning("Select at least 2 designs")
            st.stop()
        
        selected_designs = [p for p in predictions if p['design_name'] in selected_names]
        
        st.markdown("---")
        st.markdown('<div class="sidebar-title">Design Parameters</div>', unsafe_allow_html=True)
        
        for design in selected_designs:
            with st.expander(design['design_name'], expanded=False):
                render_design_params(design)
        
        st.markdown("---")
        st.markdown('<div class="sidebar-title">Analysis Options</div>', unsafe_allow_html=True)
        
        show_track = st.checkbox("Track Map", value=True)
        show_speed = st.checkbox("Speed Trace", value=True)
        show_energy = st.checkbox("Energy Deployment", value=True)
        show_delta = st.checkbox("Time Delta", value=True)
        
        # Animation control
        if show_track:
            st.markdown("---")
            st.markdown('<div class="sidebar-title">Lap Animation</div>', unsafe_allow_html=True)
            
            animation_progress = st.slider(
                "Lap Progress",
                min_value=0.0,
                max_value=1.0,
                value=1.0,
                step=0.01,
                format="%.0f%%",
                label_visibility="collapsed"
            )
            
            # Show current lap time for reference
            if selected_designs:
                fastest = min(selected_designs, key=lambda x: x['total_lap_time'])
                current_time = animation_progress * fastest['total_lap_time']
                st.caption(f"Reference: {current_time:.1f}s / {fastest['total_lap_time']:.1f}s")
        else:
            animation_progress = 1.0
    
    # Main content
    # Summary metrics
    render_summary_metrics(selected_designs)
    
    # Track map with animation
    if show_track:
        st.markdown('<div class="section-header">Circuit Position</div>', unsafe_allow_html=True)
        track_fig = create_track_map(selected_designs, animation_progress)
        st.plotly_chart(track_fig, use_container_width=True)
    
    # Timing grid
    st.markdown('<div class="section-header">Lap Time Analysis</div>', unsafe_allow_html=True)
    render_timing_grid(selected_designs, track_data)
    
    # Plots
    if show_speed or show_energy:
        st.markdown('<div class="section-header">Telemetry Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        if show_speed:
            with col1:
                fig = create_speed_trace(selected_designs)
                st.plotly_chart(fig, use_container_width=True)
        
        if show_energy:
            with col2:
                fig = create_energy_trace(selected_designs)
                st.plotly_chart(fig, use_container_width=True)
    
    if show_delta and len(selected_designs) >= 2:
        fig = create_delta_trace(selected_designs)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("""
    <div style="text-align: center; color: #999; font-size: 11px; margin-top: 40px; padding-top: 20px; border-top: 1px solid #e0e0e0;">
        F1 Design Trade-Space Analysis | ML-Predicted Performance | Physics-Informed Synthetic Data
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
