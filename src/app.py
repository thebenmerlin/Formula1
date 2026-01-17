"""
F1 Design Setup Comparison Tool

Professional engineering dashboard for comparing vehicle design configurations
at Spa-Francorchamps. Users define setups, run simulation, watch animated comparison.

Core Workflow:
1. User configures 2-4 design setups with different parameters
2. Click "Run Comparison" to compute predictions
3. Watch animated dots progress around the track at predicted speeds
4. Analyze results in comprehensive data tables below
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json
import os
import sys
import subprocess
import time
from typing import Dict, List, Tuple, Optional
from PIL import Image
import base64
from io import BytesIO

# Page config - collapsed sidebar, wide layout
st.set_page_config(
    page_title="F1 Design Comparison",
    page_icon="🏁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Professional CSS
st.markdown("""
<style>
    .stApp {
        background-color: #fafafa;
    }
    
    /* Top bar styling */
    .top-bar {
        background: #1a1a1a;
        color: white;
        padding: 15px 25px;
        border-radius: 8px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .top-bar h1 {
        margin: 0;
        font-size: 20px;
        font-weight: 600;
    }
    
    .top-bar .subtitle {
        color: #888;
        font-size: 12px;
    }
    
    /* Setup card */
    .setup-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
    }
    
    .setup-header {
        display: flex;
        align-items: center;
        margin-bottom: 12px;
        padding-bottom: 10px;
        border-bottom: 1px solid #f0f0f0;
    }
    
    .setup-color {
        width: 16px;
        height: 16px;
        border-radius: 3px;
        margin-right: 10px;
    }
    
    .setup-name {
        font-weight: 600;
        font-size: 14px;
    }
    
    /* Data table */
    .results-table {
        background: white;
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #e0e0e0;
        margin-top: 20px;
    }
    
    .results-table table {
        width: 100%;
        border-collapse: collapse;
    }
    
    .results-table th {
        background: #f5f5f5;
        padding: 12px 16px;
        text-align: left;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #666;
    }
    
    .results-table td {
        padding: 14px 16px;
        font-size: 13px;
        font-family: 'Monaco', 'Consolas', monospace;
        border-top: 1px solid #f0f0f0;
    }
    
    .fastest { color: #9b59b6; font-weight: 700; }
    .slower { color: #e74c3c; }
    .better { color: #27ae60; }
    
    /* Section title */
    .section-title {
        font-size: 13px;
        font-weight: 600;
        color: #333;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin: 25px 0 15px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid #e0e0e0;
    }
    
    /* Track container */
    .track-container {
        background: white;
        border-radius: 8px;
        padding: 20px;
        border: 1px solid #e0e0e0;
        position: relative;
    }
    
    /* Animation control */
    .anim-control {
        display: flex;
        align-items: center;
        gap: 15px;
        padding: 15px;
        background: #f5f5f5;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    
    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Constants
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPA_IMAGE_PATH = os.path.join(PROJECT_ROOT, "Spa.png")

# Setup colors (F1 team inspired)
SETUP_COLORS = [
    {"name": "Red", "hex": "#E10600"},
    {"name": "Teal", "hex": "#00A19C"},
    {"name": "Blue", "hex": "#0600EF"},
    {"name": "Orange", "hex": "#FF8000"},
]

# Default parameter ranges
PARAM_DEFAULTS = {
    "m": {"min": 740, "max": 800, "default": 770, "step": 5, "unit": "kg", "label": "Mass"},
    "C_L": {"min": 0.80, "max": 1.30, "default": 1.0, "step": 0.05, "unit": "", "label": "Aero Load (C_L)"},
    "C_D": {"min": 0.85, "max": 1.20, "default": 1.0, "step": 0.05, "unit": "", "label": "Aero Drag (C_D)"},
    "alpha_elec": {"min": 0.35, "max": 0.60, "default": 0.47, "step": 0.01, "unit": "", "label": "Electric Fraction"},
    "E_deploy": {"min": 5.0, "max": 9.0, "default": 7.0, "step": 0.5, "unit": "MJ", "label": "Energy Deploy"},
    "gamma_cool": {"min": 0.8, "max": 1.4, "default": 1.0, "step": 0.05, "unit": "", "label": "Cooling Factor"},
}


def load_spa_image() -> Optional[str]:
    """Load Spa track image as base64 for display."""
    if not os.path.exists(SPA_IMAGE_PATH):
        return None
    try:
        img = Image.open(SPA_IMAGE_PATH)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    except Exception:
        return None


def run_physics_simulation(setups: List[Dict]) -> List[Dict]:
    """
    Run physics simulation for given setups.
    Returns list of result dicts with lap_time, energy, thermal, segments.
    """
    results = []
    
    # Import physics model
    sys.path.insert(0, PROJECT_ROOT)
    try:
        from src.data.physics_model import PhysicsModel
        from src.data.track import SPA_TRACK
    except ImportError:
        st.error("Could not import physics model")
        return results
    
    for setup in setups:
        params = setup["params"]
        model = PhysicsModel(params)
        lap_time, energy_used, thermal_risk, segment_results = model.simulate_lap()
        
        # Calculate sector times
        sectors = {1: 0, 2: 0, 3: 0}
        for seg in segment_results:
            segment = SPA_TRACK.segments[seg.segment_id]
            sectors[segment.sector] += seg.time
        
        results.append({
            "name": setup["name"],
            "color": setup["color"],
            "params": params,
            "lap_time": lap_time,
            "energy_used": energy_used,
            "thermal_risk": thermal_risk,
            "sector_1": sectors[1],
            "sector_2": sectors[2],
            "sector_3": sectors[3],
            "segments": segment_results,
        })
    
    return results


def create_track_animation(
    results: List[Dict],
    progress: float,
    spa_image_b64: Optional[str] = None
) -> go.Figure:
    """
    Create track visualization with animated markers.
    Uses Spa.png as background if available.
    """
    fig = go.Figure()
    
    # Add Spa track image as background
    if spa_image_b64:
        fig.add_layout_image(
            dict(
                source=f"data:image/png;base64,{spa_image_b64}",
                xref="x", yref="y",
                x=0, y=1,
                sizex=1, sizey=1,
                sizing="stretch",
                opacity=1.0,
                layer="below"
            )
        )
    
    # Track coordinates for marker positions (normalized 0-1)
    # Matched to Spa.png image orientation:
    # - Start/finish at right side with La Source hairpin
    # - Track flows counter-clockwise
    # - Eau Rouge/Raidillon climbing left
    # - Long Kemmel straight going left-down
    # - Forest section at bottom-left
    # - Returns via right side
    TRACK_PATH = [
        # Start/Finish → La Source (top right)
        (0.88, 0.15), (0.82, 0.12), (0.76, 0.12), (0.72, 0.15), (0.70, 0.18),
        # La Source exit → Eau Rouge (going down and left)
        (0.68, 0.22), (0.65, 0.28), (0.60, 0.35),
        # Eau Rouge → Raidillon (sharp left climb)
        (0.55, 0.40), (0.48, 0.42), (0.42, 0.38),
        # Raidillon top → Kemmel Straight (long straight going down-left)
        (0.38, 0.32), (0.32, 0.28), (0.25, 0.25), (0.18, 0.24), (0.10, 0.26),
        # Les Combes (bottom left corner area)
        (0.06, 0.30), (0.05, 0.38), (0.08, 0.45),
        # Rivage → Pouhon (climbing back up through forest)
        (0.12, 0.52), (0.18, 0.60), (0.25, 0.68),
        # Fagnes → Campus → Stavelot (middle section going right)
        (0.32, 0.75), (0.42, 0.80), (0.52, 0.82),
        # Blanchimont → Bus Stop (back toward start)
        (0.62, 0.78), (0.72, 0.70), (0.80, 0.58),
        # Final approach to Start/Finish
        (0.85, 0.45), (0.88, 0.32), (0.88, 0.20),
        # Complete lap
        (0.88, 0.15),
    ]
    
    # Add marker for each setup
    for result in results:
        # Calculate position based on time progress
        total_time = result["lap_time"]
        current_time = progress * total_time
        
        # Find position along track
        cumulative_time = 0
        segment_idx = 0
        seg_progress = 0
        
        for seg in result["segments"]:
            if cumulative_time + seg.time >= current_time:
                seg_progress = (current_time - cumulative_time) / seg.time
                break
            cumulative_time += seg.time
            segment_idx += 1
        
        # Map to track coordinates
        n_track_pts = len(TRACK_PATH)
        overall_progress = (segment_idx + seg_progress) / len(result["segments"])
        pt_idx = int(overall_progress * (n_track_pts - 1))
        pt_idx = min(pt_idx, n_track_pts - 2)
        
        t = (overall_progress * (n_track_pts - 1)) - pt_idx
        x1, y1 = TRACK_PATH[pt_idx]
        x2, y2 = TRACK_PATH[pt_idx + 1]
        
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        
        # Add marker
        fig.add_trace(go.Scatter(
            x=[x], y=[1 - y],  # Flip Y for image coordinates
            mode='markers+text',
            marker=dict(
                size=20,
                color=result["color"],
                line=dict(color='white', width=3),
                symbol='circle'
            ),
            text=[result["name"][:3]],
            textposition='bottom center',
            textfont=dict(size=10, color='#333'),
            name=result["name"],
            showlegend=True,
            hovertemplate=(
                f"<b>{result['name']}</b><br>"
                f"Time: {current_time:.2f}s<br>"
                f"Lap: {result['lap_time']:.2f}s<br>"
                "<extra></extra>"
            )
        ))
    
    fig.update_layout(
        xaxis=dict(range=[0, 1], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[0, 1], showgrid=False, zeroline=False, showticklabels=False, scaleanchor='x'),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=0, r=0, t=0, b=0),
        height=500,
        legend=dict(
            orientation='h',
            yanchor='top',
            y=-0.02,
            xanchor='center',
            x=0.5
        )
    )
    
    return fig


def render_results_table(results: List[Dict]) -> None:
    """Render comprehensive results table."""
    if not results:
        return
    
    # Find best values
    min_lap = min(r["lap_time"] for r in results)
    min_energy = min(r["energy_used"] for r in results)
    
    html = '<div class="results-table"><table>'
    html += '''
    <tr>
        <th>Setup</th>
        <th>Lap Time</th>
        <th>Gap</th>
        <th>Sector 1</th>
        <th>Sector 2</th>
        <th>Sector 3</th>
        <th>Energy (MJ)</th>
        <th>Thermal</th>
    </tr>
    '''
    
    baseline_time = results[0]["lap_time"]
    
    for r in sorted(results, key=lambda x: x["lap_time"]):
        gap = r["lap_time"] - baseline_time
        gap_str = f"+{gap:.3f}" if gap > 0 else f"{gap:.3f}" if gap < 0 else "—"
        gap_class = "slower" if gap > 0 else "better" if gap < 0 else ""
        lap_class = "fastest" if abs(r["lap_time"] - min_lap) < 0.001 else ""
        
        html += f'''
        <tr>
            <td>
                <span style="display:inline-block;width:12px;height:12px;background:{r["color"]};border-radius:2px;margin-right:8px;vertical-align:middle;"></span>
                {r["name"]}
            </td>
            <td class="{lap_class}">{r["lap_time"]:.3f}s</td>
            <td class="{gap_class}">{gap_str}</td>
            <td>{r["sector_1"]:.3f}s</td>
            <td>{r["sector_2"]:.3f}s</td>
            <td>{r["sector_3"]:.3f}s</td>
            <td>{r["energy_used"]:.2f}</td>
            <td>{r["thermal_risk"]:.3f}</td>
        </tr>
        '''
    
    html += '</table></div>'
    st.markdown(html, unsafe_allow_html=True)


def render_setup_params_table(results: List[Dict]) -> None:
    """Render setup parameters comparison table."""
    if not results:
        return
    
    html = '<div class="results-table"><table>'
    html += '''
    <tr>
        <th>Setup</th>
        <th>Mass (kg)</th>
        <th>C_L</th>
        <th>C_D</th>
        <th>L/D Ratio</th>
        <th>Electric %</th>
        <th>Energy (MJ)</th>
        <th>Cooling</th>
    </tr>
    '''
    
    for r in results:
        p = r["params"]
        ld_ratio = p["C_L"] / p["C_D"] if p["C_D"] > 0 else 0
        
        html += f'''
        <tr>
            <td>
                <span style="display:inline-block;width:12px;height:12px;background:{r["color"]};border-radius:2px;margin-right:8px;vertical-align:middle;"></span>
                {r["name"]}
            </td>
            <td>{p["m"]:.0f}</td>
            <td>{p["C_L"]:.2f}</td>
            <td>{p["C_D"]:.2f}</td>
            <td>{ld_ratio:.2f}</td>
            <td>{p["alpha_elec"]*100:.0f}%</td>
            <td>{p["E_deploy"]:.1f}</td>
            <td>{p["gamma_cool"]:.2f}</td>
        </tr>
        '''
    
    html += '</table></div>'
    st.markdown(html, unsafe_allow_html=True)


def main():
    """Main application."""
    
    # Header
    st.markdown("""
    <div class="top-bar">
        <div>
            <h1>🏁 F1 Design Setup Comparison</h1>
            <span class="subtitle">Spa-Francorchamps | 7.004 km | 2026 Hybrid Regulations</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if "setups" not in st.session_state:
        st.session_state.setups = [
            {"name": "Baseline", "params": {k: v["default"] for k, v in PARAM_DEFAULTS.items()}},
            {"name": "Low Drag", "params": {"m": 755, "C_L": 0.85, "C_D": 0.88, "alpha_elec": 0.52, "E_deploy": 8.0, "gamma_cool": 0.9}},
        ]
    if "results" not in st.session_state:
        st.session_state.results = None
    if "animation_progress" not in st.session_state:
        st.session_state.animation_progress = 0.0
    
    # Setup configuration section
    st.markdown('<div class="section-title">Design Setups</div>', unsafe_allow_html=True)
    
    # Setup columns
    n_setups = len(st.session_state.setups)
    cols = st.columns(min(n_setups + 1, 5))
    
    for i, setup in enumerate(st.session_state.setups):
        with cols[i]:
            color = SETUP_COLORS[i % len(SETUP_COLORS)]
            
            st.markdown(f"""
            <div class="setup-card">
                <div class="setup-header">
                    <div class="setup-color" style="background:{color['hex']}"></div>
                    <span class="setup-name">Setup {i+1}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Name input
            name = st.text_input("Name", value=setup["name"], key=f"name_{i}", label_visibility="collapsed")
            st.session_state.setups[i]["name"] = name
            
            # Parameter sliders
            for param, config in PARAM_DEFAULTS.items():
                val = st.slider(
                    config["label"],
                    min_value=float(config["min"]),
                    max_value=float(config["max"]),
                    value=float(setup["params"].get(param, config["default"])),
                    step=float(config["step"]),
                    key=f"{param}_{i}"
                )
                st.session_state.setups[i]["params"][param] = val
    
    # Add setup button
    if n_setups < 4:
        with cols[n_setups]:
            if st.button("➕ Add Setup", use_container_width=True):
                st.session_state.setups.append({
                    "name": f"Setup {n_setups + 1}",
                    "params": {k: v["default"] for k, v in PARAM_DEFAULTS.items()}
                })
                st.rerun()
    
    # Run button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🏁 Run Comparison", type="primary", use_container_width=True):
            with st.spinner("Running physics simulation..."):
                # Add colors to setups
                setups_with_colors = []
                for i, setup in enumerate(st.session_state.setups):
                    setups_with_colors.append({
                        **setup,
                        "color": SETUP_COLORS[i % len(SETUP_COLORS)]["hex"]
                    })
                
                st.session_state.results = run_physics_simulation(setups_with_colors)
                st.session_state.animation_progress = 0.0
            st.rerun()
    
    # Results section
    if st.session_state.results:
        results = st.session_state.results
        
        # Track animation
        st.markdown('<div class="section-title">Lap Animation</div>', unsafe_allow_html=True)
        
        # Animation controls
        animation_progress = st.slider(
            "Progress",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.animation_progress,
            step=0.01,
            format="%.0f%%",
            key="progress_slider"
        )
        st.session_state.animation_progress = animation_progress
        
        # Show current time
        if results:
            fastest = min(results, key=lambda x: x["lap_time"])
            current_time = animation_progress * fastest["lap_time"]
            st.caption(f"Reference time: {current_time:.1f}s / {fastest['lap_time']:.1f}s")
        
        # Track visualization
        spa_image = load_spa_image()
        track_fig = create_track_animation(results, animation_progress, spa_image)
        st.plotly_chart(track_fig, use_container_width=True)
        
        # Results tables
        st.markdown('<div class="section-title">Lap Time Results</div>', unsafe_allow_html=True)
        render_results_table(results)
        
        st.markdown('<div class="section-title">Setup Parameters</div>', unsafe_allow_html=True)
        render_setup_params_table(results)
        
        # Key insights
        st.markdown('<div class="section-title">Analysis</div>', unsafe_allow_html=True)
        
        fastest = min(results, key=lambda x: x["lap_time"])
        slowest = max(results, key=lambda x: x["lap_time"])
        most_efficient = min(results, key=lambda x: x["energy_used"])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Fastest Lap", f"{fastest['lap_time']:.3f}s", delta=fastest['name'])
        
        with col2:
            spread = slowest["lap_time"] - fastest["lap_time"]
            st.metric("Performance Spread", f"{spread:.3f}s", delta=f"{fastest['name']} → {slowest['name']}")
        
        with col3:
            st.metric("Most Efficient", f"{most_efficient['energy_used']:.2f} MJ", delta=most_efficient['name'])
    
    # Footer
    st.markdown("""
    <div style="text-align: center; color: #999; font-size: 11px; margin-top: 40px; padding-top: 20px; border-top: 1px solid #e0e0e0;">
        F1 Design Trade-Space Analysis | Physics-Informed Simulation | ML Surrogate Models
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
