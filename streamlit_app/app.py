"""
F1 Spa-Francorchamps Analytics Dashboard
Enterprise-grade, professional analytics UI
"""

import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from inference import F1Predictor, Setup, PARAM_BOUNDS
from analytics import (
    check_monotonicity, compute_tradeoff_curve, compute_aero_efficiency_curve,
    get_model_stats
)
from plots import (
    plot_segment_deltas, plot_cumulative_time,
    plot_feature_importance, plot_pdp_grid, plot_tradeoff
)


# ============================================================================
# UTILITIES
# ============================================================================

def format_laptime(seconds: float) -> str:
    minutes = int(seconds // 60)
    remaining = seconds % 60
    return f"{minutes}:{remaining:06.3f}"

def format_delta(delta: float) -> str:
    if abs(delta) < 0.001:
        return "-"
    return f"{'+' if delta > 0 else ''}{delta:.3f}s"

def get_delta_color(delta: float) -> str:
    if delta < -0.1:
        return "#10B981"  # Green - faster
    elif delta > 0.1:
        return "#EF4444"  # Red - slower
    return "#6B7280"  # Gray - neutral

def calculate_performance_score(pred: dict, baseline_pred: dict = None) -> float:
    """Calculate a performance score 0-100 based on prediction."""
    # Base score from lap time (lower is better)
    base_score = max(0, min(100, 100 - (pred['lap_time'] - 100) * 2))
    
    # Penalty for invalid setups
    if not pred['is_valid']:
        base_score *= 0.9
    if pred['is_ood']:
        base_score *= 0.95
    
    return round(base_score, 1)


# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="F1 Performance Analytics | Spa-Francorchamps",
    page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>F1</text></svg>",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Light-Mode CSS styling
st.markdown("""
<style>
    /* ===== GLOBAL STYLES ===== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background: #f8fafc !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    /* ===== SIDEBAR ===== */
    section[data-testid="stSidebar"] {
        background: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
    }
    
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label {
        color: #334155 !important;
    }
    
    /* ===== TYPOGRAPHY ===== */
    .stApp h1 {
        color: #0f172a !important;
        font-weight: 700 !important;
        font-size: 1.75rem !important;
        letter-spacing: -0.025em !important;
    }
    
    .stApp h2 {
        color: #1e293b !important;
        font-weight: 600 !important;
        font-size: 1.125rem !important;
        border-bottom: 1px solid #e2e8f0 !important;
        padding-bottom: 0.75rem !important;
        margin-top: 1.5rem !important;
    }
    
    .stApp h3 {
        color: #334155 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }
    
    .stApp p, .stApp span, .stApp label, .stApp div {
        color: #475569 !important;
    }
    
    /* ===== METRIC CARDS ===== */
    [data-testid="stMetric"] {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05) !important;
    }
    
    [data-testid="stMetricValue"] {
        color: #0f172a !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.025em !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #64748b !important;
        font-weight: 500 !important;
        text-transform: uppercase !important;
        font-size: 0.7rem !important;
        letter-spacing: 0.05em !important;
    }
    
    [data-testid="stMetricDelta"] {
        font-weight: 600 !important;
    }
    
    /* ===== DATA TABLES ===== */
    .stDataFrame {
        background: #ffffff !important;
        border-radius: 8px !important;
        border: 1px solid #e2e8f0 !important;
        overflow: hidden !important;
    }
    
    .stDataFrame table {
        color: #334155 !important;
    }
    
    .stDataFrame th {
        background: #f1f5f9 !important;
        color: #475569 !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        font-size: 0.7rem !important;
        letter-spacing: 0.05em !important;
        padding: 0.75rem 1rem !important;
    }
    
    .stDataFrame td {
        color: #334155 !important;
        padding: 0.75rem 1rem !important;
        border-bottom: 1px solid #f1f5f9 !important;
    }
    
    .stDataFrame tr:hover td {
        background: #f8fafc !important;
    }
    
    /* ===== BUTTONS ===== */
    .stButton > button {
        background-color: #1e293b !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1rem !important;
    }
    
    .stButton > button:hover {
        background-color: #0f172a !important;
        color: #ffffff !important;
    }
    
    .stButton > button * {
        color: #ffffff !important;
    }
    
    /* Sidebar specific button styling */
    section[data-testid="stSidebar"] .stButton > button {
        background-color: #475569 !important;
        color: #ffffff !important;
    }
    
    section[data-testid="stSidebar"] .stButton > button:hover {
        background-color: #334155 !important;
    }
    
    section[data-testid="stSidebar"] .stButton > button * {
        color: #ffffff !important;
    }
    
    /* Primary button styling */
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="stBaseButton-primary"],
    button[data-testid="baseButton-primary"] {
        background-color: #1d4ed8 !important;
        color: #ffffff !important;
    }
    
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="stBaseButton-primary"]:hover,
    button[data-testid="baseButton-primary"]:hover {
        background-color: #1e40af !important;
    }
    
    .stButton > button[kind="primary"] *,
    button[data-testid="baseButton-primary"] * {
        color: #ffffff !important;
    }
    
    /* ===== EXPANDERS ===== */
    details {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 6px !important;
        margin-bottom: 0.5rem !important;
    }
    
    details summary {
        color: #334155 !important;
        font-weight: 500 !important;
        padding: 0.75rem !important;
    }
    
    details summary:hover {
        background: #f8fafc !important;
    }
    
    /* ===== SLIDERS ===== */
    .stSlider > div > div > div {
        background: #1e293b !important;
    }
    
    .stSlider label {
        color: #475569 !important;
        font-weight: 500 !important;
    }
    
    /* ===== TABS ===== */
    .stTabs [data-baseweb="tab-list"] {
        background: #f1f5f9 !important;
        border-radius: 6px !important;
        padding: 4px !important;
        gap: 4px !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: #64748b !important;
        border-radius: 4px !important;
        font-weight: 500 !important;
        padding: 0.5rem 1rem !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: #ffffff !important;
        color: #0f172a !important;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
    }
    
    /* ===== DIVIDERS ===== */
    hr {
        border-color: #e2e8f0 !important;
        margin: 1.5rem 0 !important;
    }
    
    /* ===== ALERTS ===== */
    .stAlert {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 6px !important;
    }
    
    /* ===== SELECTBOX ===== */
    .stSelectbox > div > div {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 6px !important;
        color: #334155 !important;
    }
    
    /* ===== CUSTOM CLASSES ===== */
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.25rem;
        text-align: center;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    }
    
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.2;
    }
    
    .kpi-label {
        font-size: 0.7rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 0.5rem;
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-size: 0.65rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .status-valid {
        background: #dcfce7;
        color: #166534;
        border: 1px solid #bbf7d0;
    }
    
    .status-warning {
        background: #fef3c7;
        color: #92400e;
        border: 1px solid #fde68a;
    }
    
    .status-error {
        background: #fee2e2;
        color: #991b1b;
        border: 1px solid #fecaca;
    }
    
    .header-subtitle {
        color: #64748b;
        font-size: 0.8rem;
        margin-top: -0.5rem;
    }
    
    .section-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    .delta-positive {
        color: #dc2626 !important;
        font-weight: 600;
    }
    
    .delta-negative {
        color: #16a34a !important;
        font-weight: 600;
    }
    
    .insight-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 0.875rem;
        margin: 0.5rem 0;
    }
    
    .insight-title {
        color: #1e293b;
        font-weight: 600;
        font-size: 0.8rem;
        margin-bottom: 0.375rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# LOAD MODEL
# ============================================================================

if 'predictor' not in st.session_state:
    try:
        st.session_state.predictor = F1Predictor()
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        st.stop()

if 'predictions' not in st.session_state:
    st.session_state.predictions = []
if 'setups' not in st.session_state:
    st.session_state.setups = []
if 'analysis_timestamp' not in st.session_state:
    st.session_state.analysis_timestamp = None


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    # Logo/Branding area
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0 1.5rem 0;">
        <div style="font-size: 1.5rem; font-weight: 800; color: #0f172a; letter-spacing: -0.05em; margin-bottom: 0.25rem;">F1</div>
        <div style="font-size: 1rem; font-weight: 600; color: #1e293b; letter-spacing: -0.025em;">Analytics</div>
        <div style="font-size: 0.65rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 0.25rem;">Performance Lab</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Quick Stats
    stats = get_model_stats(st.session_state.predictor)
    st.markdown(f"""
    <div style="background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 6px; padding: 0.75rem; margin-bottom: 1rem;">
        <div style="font-size: 0.65rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.25rem;">Model Status</div>
        <div style="font-size: 0.8rem; color: #1e293b; font-weight: 500;">{stats['model_name'].upper()} | RMSE: {stats['cv_rmse']:.3f}s</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Configuration")
    
    n_setups = st.selectbox("Number of Setups", [1, 2, 3], index=1, help="Compare up to 3 different vehicle configurations")
    
    st.divider()
    
    setups = []
    setup_colors = ['#3b82f6', '#10b981', '#f59e0b']
    
    for i in range(n_setups):
        label = "Baseline" if i == 0 else f"Setup {chr(64+i)}"
        
        with st.expander(label, expanded=(i == 0)):
            st.markdown(f"""
            <div style="height: 3px; background: {setup_colors[i]}; border-radius: 2px; margin-bottom: 1rem;"></div>
            """, unsafe_allow_html=True)
            
            mass = st.slider("Mass (kg)", 700.0, 850.0, 780.0 + i*20, 5.0, key=f"m{i}",
                           help="Total vehicle mass including driver")
            
            st.markdown("**Aerodynamics**")
            c1, c2 = st.columns(2)
            c_l = c1.slider("Cₗ (Lift)", 0.8, 1.5, 1.2, 0.05, key=f"cl{i}", help="Lift coefficient")
            c_d = c2.slider("Cᴅ (Drag)", 0.7, 1.3, 1.0, 0.05, key=f"cd{i}", help="Drag coefficient")
            
            st.markdown("**Powertrain**")
            alpha = st.slider("α (Electric Fraction)", 0.0, 0.4, 0.15, 0.02, key=f"a{i}",
                            help="Fraction of power from electric motor")
            
            c1, c2 = st.columns(2)
            e_dep = c1.slider("E_deploy (MJ)", 2.0, 4.0, 3.0, 0.1, key=f"e{i}",
                             help="Deployable energy per lap")
            gamma = c2.slider("γ (Cooling)", 0.8, 1.2, 1.0, 0.05, key=f"g{i}",
                             help="Cooling system aggressiveness")
            
            # Show aero efficiency
            aero_eff = c_l / c_d if c_d > 0 else 0
            st.markdown(f"""
            <div style="background: #f1f5f9; border-radius: 4px; padding: 0.5rem; margin-top: 0.5rem;">
                <span style="font-size: 0.7rem; color: #64748b;">Aero Efficiency:</span>
                <span style="font-size: 0.8rem; color: #1e293b; font-weight: 600; margin-left: 0.25rem;">{aero_eff:.2f}</span>
            </div>
            """, unsafe_allow_html=True)
            
            setups.append((label, Setup(mass, c_l, c_d, alpha, e_dep, gamma)))
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Reset", width='stretch'):
            for key in list(st.session_state.keys()):
                if key not in ['predictor']:
                    del st.session_state[key]
            st.rerun()
    
    with col2:
        if st.button("Analyze", width='stretch', type="primary"):
            st.session_state.setups = setups
            st.session_state.predictions = [
                st.session_state.predictor.predict(s) for _, s in setups
            ]
            st.session_state.analysis_timestamp = datetime.now().strftime("%H:%M:%S")


# ============================================================================
# MAIN PAGE
# ============================================================================

# Header
col1, col2, col3 = st.columns([2, 3, 2])
with col2:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h1 style="margin: 0; font-size: 1.75rem; color: #0f172a;">Track Performance Analytics</h1>
        <p style="color: #64748b; margin-top: 0.5rem; font-size: 0.8rem;">
            Spa-Francorchamps Circuit | 7.004 km | Machine Learning Prediction Engine
        </p>
    </div>
    """, unsafe_allow_html=True)

if not st.session_state.predictions:
    # Welcome screen
    st.markdown("""
    <div style="background: #ffffff; 
                border: 1px solid #e2e8f0; border-radius: 12px; padding: 2rem; 
                text-align: center; margin: 2rem auto; max-width: 600px;">
        <h2 style="color: #0f172a; margin-bottom: 0.5rem; border: none;">Welcome to F1 Analytics</h2>
        <p style="color: #64748b; margin-bottom: 1.5rem;">
            Configure your vehicle setups in the sidebar and click <strong>Analyze</strong> to run performance predictions.
        </p>
        <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
            <div style="text-align: center;">
                <div style="font-size: 0.75rem; font-weight: 600; color: #1e293b;">Real-time Analysis</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 0.75rem; font-weight: 600; color: #1e293b;">ML Predictions</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 0.75rem; font-weight: 600; color: #1e293b;">Sensitivity Analysis</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

setups = st.session_state.setups
predictions = st.session_state.predictions
predictor = st.session_state.predictor


# ----------------------------------------------------------------------------
# EXECUTIVE SUMMARY - KPI DASHBOARD
# ----------------------------------------------------------------------------

st.markdown("## Executive Summary")

# Top KPI row
kpi_cols = st.columns(4)

# Best Lap Time
best_pred = min(predictions, key=lambda p: p['lap_time'])
best_setup_idx = predictions.index(best_pred)
best_setup_name = setups[best_setup_idx][0]

with kpi_cols[0]:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value" style="color: #166534;">{format_laptime(best_pred['lap_time'])}</div>
        <div class="kpi-label">Best Lap Time</div>
        <div style="font-size: 0.7rem; color: #64748b; margin-top: 0.25rem;">{best_setup_name}</div>
    </div>
    """, unsafe_allow_html=True)

# Gap to Best (if multiple setups)
with kpi_cols[1]:
    if len(predictions) > 1:
        worst_pred = max(predictions, key=lambda p: p['lap_time'])
        gap = worst_pred['lap_time'] - best_pred['lap_time']
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value" style="color: #b45309;">{gap:.3f}s</div>
            <div class="kpi-label">Performance Spread</div>
            <div style="font-size: 0.7rem; color: #64748b; margin-top: 0.25rem;">Max gap between setups</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        perf_score = calculate_performance_score(best_pred)
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value" style="color: #1d4ed8;">{perf_score}</div>
            <div class="kpi-label">Performance Score</div>
            <div style="font-size: 0.7rem; color: #64748b; margin-top: 0.25rem;">Out of 100</div>
        </div>
        """, unsafe_allow_html=True)

with kpi_cols[2]:
    avg_uncertainty = np.mean([p['uncertainty'] for p in predictions])
    confidence_pct = max(0, min(100, 100 - (avg_uncertainty * 20)))
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value" style="color: #6d28d9;">{confidence_pct:.0f}%</div>
        <div class="kpi-label">Model Confidence</div>
        <div style="font-size: 0.7rem; color: #64748b; margin-top: 0.25rem;">+/-{avg_uncertainty:.3f}s uncertainty</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_cols[3]:
    all_valid = all(p['is_valid'] for p in predictions)
    any_ood = any(p['is_ood'] for p in predictions)
    
    if all_valid and not any_ood:
        status_text = "VALID"
        status_color = "#166534"
    elif all_valid:
        status_text = "OOD"
        status_color = "#b45309"
    else:
        status_text = "INVALID"
        status_color = "#dc2626"
    
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value" style="color: {status_color}; font-size: 1.5rem;">{status_text}</div>
        <div class="kpi-label">Validation Status</div>
        <div style="font-size: 0.7rem; color: #64748b; margin-top: 0.25rem;">{'All setups valid' if all_valid else 'Check configuration'}</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()


# ----------------------------------------------------------------------------
# LAP TIMES - DETAILED COMPARISON
# ----------------------------------------------------------------------------

st.markdown("## Lap Time Analysis")

# Setup comparison cards
cols = st.columns(len(setups))
setup_colors = ['#3b82f6', '#10b981', '#f59e0b']

for idx, (col, (name, setup), pred) in enumerate(zip(cols, setups, predictions)):
    with col:
        is_fastest = pred['lap_time'] == best_pred['lap_time']
        delta_str = ""
        delta_color = "#64748b"
        
        if idx > 0:
            d = pred['lap_time'] - predictions[0]['lap_time']
            delta_str = format_delta(d)
            delta_color = "#166534" if d < 0 else "#dc2626" if d > 0 else "#64748b"
        
        fastest_badge = '<span class="status-badge status-valid" style="margin-left: 0.5rem;">FASTEST</span>' if is_fastest and len(setups) > 1 else ''
        
        # Card header with name and badge
        card_html = f'''<div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1.25rem; border-top: 3px solid {setup_colors[idx]};">
<div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 1rem;">
<div style="font-size: 0.8rem; font-weight: 600; color: {setup_colors[idx]};">{name}</div>
{fastest_badge}
</div>
<div style="font-size: 1.75rem; font-weight: 700; color: #0f172a; letter-spacing: -0.025em;">{format_laptime(pred['lap_time'])}</div>
<div style="font-size: 0.8rem; color: {delta_color}; margin-top: 0.25rem;">{delta_str if delta_str else 'Reference'}</div>
<div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #e2e8f0;">
<div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;"><span style="font-size: 0.7rem; color: #64748b;">S1</span><span style="font-size: 0.8rem; color: #334155;">{pred['sector_1']:.3f}s</span></div>
<div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;"><span style="font-size: 0.7rem; color: #64748b;">S2</span><span style="font-size: 0.8rem; color: #334155;">{pred['sector_2']:.3f}s</span></div>
<div style="display: flex; justify-content: space-between;"><span style="font-size: 0.7rem; color: #64748b;">S3</span><span style="font-size: 0.8rem; color: #334155;">{pred['sector_3']:.3f}s</span></div>
</div>
</div>'''
        st.markdown(card_html, unsafe_allow_html=True)

# Detailed sector comparison table
if len(setups) > 1:
    st.markdown("### Sector Breakdown")
    
    rows = []
    for (name, _), pred in zip(setups, predictions):
        base = predictions[0]
        s1_delta = pred['sector_1'] - base['sector_1']
        s2_delta = pred['sector_2'] - base['sector_2']
        s3_delta = pred['sector_3'] - base['sector_3']
        
        row = {
            'Setup': name,
            'Sector 1': f"{pred['sector_1']:.3f}s",
            'S1 Δ': format_delta(s1_delta) if name != "Baseline" else '—',
            'Sector 2': f"{pred['sector_2']:.3f}s",
            'S2 Δ': format_delta(s2_delta) if name != "Baseline" else '—',
            'Sector 3': f"{pred['sector_3']:.3f}s",
            'S3 Δ': format_delta(s3_delta) if name != "Baseline" else '—',
            'Total': format_laptime(pred['lap_time']),
            'Gap': format_delta(pred['lap_time'] - base['lap_time']) if name != "Baseline" else '—'
        }
        rows.append(row)
    
    st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')

st.divider()


# ----------------------------------------------------------------------------
# SEGMENT ANALYSIS - TABBED INTERFACE
# ----------------------------------------------------------------------------

st.markdown("##Circuit Segment Analysis")

SEGMENTS = [
    "La Source", "Eau Rouge Approach", "Eau Rouge", "Raidillon", "Kemmel Straight",
    "Les Combes Entry", "Les Combes Exit", "Malmedy", "Rivage", "Pre-Pouhon",
    "Pouhon", "Fagnes", "Campus Straight", "Campus", "Stavelot",
    "Paul Frere", "Blanchimont", "Chicane Approach", "Bus Stop", "Start/Finish"
]

SEGMENT_TYPES = {
    "High-Speed": [4, 10, 12, 16],  # Kemmel, Pouhon, Campus Straight, Blanchimont
    "Braking Zone": [0, 5, 8, 13, 17, 18],  # La Source, Les Combes, Rivage, Campus, Chicane, Bus Stop
    "Technical": [1, 2, 3, 6, 7, 9, 11, 14, 15, 19]  # Others
}

tab1, tab2, tab3 = st.tabs(["Segment Table", "Progression Chart", "Delta Analysis"])

with tab1:
    col1, col2 = st.columns([3, 2])
    
    with col1:
        seg_data = {'#': list(range(1, 21)), 'Segment': SEGMENTS}
        
        for (name, _), pred in zip(setups, predictions):
            seg_data[f"{name} (s)"] = [f"{t:.3f}" for t in pred['segment_times']]
        
        if len(predictions) > 1:
            baseline_times = predictions[0]['segment_times']
            deltas = [p - b for p, b in zip(predictions[1]['segment_times'], baseline_times)]
            seg_data['Δ vs Baseline'] = [format_delta(d) for d in deltas]
        
        st.dataframe(pd.DataFrame(seg_data), hide_index=True, width='stretch', height=450)
    
    with col2:
        # Segment type breakdown
        st.markdown("### Segment Categories")
        
        for seg_type, indices in SEGMENT_TYPES.items():
            type_time = sum(predictions[0]['segment_times'][i] for i in indices)
            pct = (type_time / predictions[0]['lap_time']) * 100
            
            if seg_type == "High-Speed":
                color = "#166534"
            elif seg_type == "Braking Zone":
                color = "#dc2626"
            else:
                color = "#b45309"
            
            st.markdown(f"""
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 6px; padding: 0.75rem; margin-bottom: 0.5rem;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="color: #334155; font-weight: 500;">{seg_type}</span>
                    <span style="color: {color}; font-weight: 600;">{type_time:.2f}s ({pct:.1f}%)</span>
                </div>
                <div style="background: #e2e8f0; border-radius: 4px; height: 4px; margin-top: 0.5rem;">
                    <div style="background: {color}; width: {pct}%; height: 100%; border-radius: 4px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

with tab2:
    st.markdown("### Cumulative Time Progression")
    fig = plot_cumulative_time([n for n, _ in setups], [p['segment_times'] for p in predictions])
    st.pyplot(fig)
    
    # Add insights
    if len(setups) > 1:
        cumulative_baseline = np.cumsum(predictions[0]['segment_times'])
        cumulative_alt = np.cumsum(predictions[1]['segment_times'])
        
        # Find where gap changes most
        gaps = cumulative_alt - cumulative_baseline
        max_gain_seg = np.argmin(np.diff(gaps))
        max_loss_seg = np.argmax(np.diff(gaps))
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="insight-box">
                <div class="insight-title">Biggest Gain</div>
                <div style="color: #166534; font-weight: 600;">{SEGMENTS[max_gain_seg]}</div>
                <div style="color: #64748b; font-size: 0.7rem;">Segment {max_gain_seg + 1}</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="insight-box">
                <div class="insight-title">Biggest Loss</div>
                <div style="color: #dc2626; font-weight: 600;">{SEGMENTS[max_loss_seg]}</div>
                <div style="color: #64748b; font-size: 0.7rem;">Segment {max_loss_seg + 1}</div>
            </div>
            """, unsafe_allow_html=True)

with tab3:
    if len(setups) > 1:
        st.markdown("### Delta vs Baseline by Segment")
        deltas = [predictions[1]['segment_times'][i] - predictions[0]['segment_times'][i] for i in range(20)]
        fig = plot_segment_deltas(deltas, SEGMENTS)
        st.pyplot(fig)
        
        # Summary stats
        faster_count = sum(1 for d in deltas if d < -0.001)
        slower_count = sum(1 for d in deltas if d > 0.001)
        neutral_count = 20 - faster_count - slower_count
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Faster Segments", faster_count, delta=f"{faster_count/20*100:.0f}%")
        with col2:
            st.metric("Slower Segments", slower_count, delta=f"-{slower_count/20*100:.0f}%", delta_color="inverse")
        with col3:
            st.metric("Neutral", neutral_count)
    else:
        st.info("Add a second setup to compare segment deltas.")

st.divider()


# ----------------------------------------------------------------------------
# SENSITIVITY ANALYSIS - ENHANCED
# ----------------------------------------------------------------------------

st.markdown("## Model Sensitivity Analysis")

sens_tab1, sens_tab2, sens_tab3 = st.tabs(["Feature Importance", "Partial Dependence", "Physics Validation"])

with sens_tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        _ = predictor.predict(setups[0][1])
        importance_df = predictor.get_feature_importance()
        fig = plot_feature_importance(importance_df, top_n=12)
        st.pyplot(fig)
    
    with col2:
        st.markdown("### Key Drivers")
        st.markdown("""
        <div class="insight-box">
            <div class="insight-title">Interpretation Guide</div>
            <p style="font-size: 0.75rem; color: #64748b; margin: 0;">
                Feature importance shows which vehicle parameters have the strongest influence on predicted lap time. 
                Higher values indicate greater impact on performance.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Top 5 features with descriptions
        top_features = importance_df.head(5)
        for _, row in top_features.iterrows():
            feature = row['feature']
            importance = row['importance']
            
            # Feature descriptions
            descriptions = {
                'mass': 'Vehicle mass affects acceleration and cornering',
                'c_d': 'Drag coefficient impacts top speed',
                'c_l': 'Lift coefficient provides downforce',
                'alpha_elec': 'Electric power deployment ratio',
                'e_deploy': 'Total deployable energy per lap',
                'gamma_cool': 'Cooling system aggressiveness',
                'aero_efficiency': 'Ratio of downforce to drag',
                'power_to_weight': 'Effective power density'
            }
            
            desc = descriptions.get(feature, 'Engineered feature')
            
            st.markdown(f"""
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 6px; padding: 0.75rem; margin-bottom: 0.5rem;">
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: #334155; font-weight: 500;">{feature}</span>
                    <span style="color: #1d4ed8; font-weight: 600;">{importance:.3f}</span>
                </div>
                <div style="color: #64748b; font-size: 0.7rem; margin-top: 0.25rem;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

with sens_tab2:
    st.markdown("### Partial Dependence Plots")
    st.markdown("""
    <p style="color: #64748b; font-size: 0.8rem; margin-bottom: 1rem;">
        How each parameter affects predicted lap time while holding other variables constant.
    </p>
    """, unsafe_allow_html=True)
    
    pdp_data = {}
    for p in ['mass', 'c_l', 'c_d', 'alpha_elec', 'e_deploy', 'gamma_cool']:
        vals, times = predictor.compute_partial_dependence(p, setups[0][1])
        pdp_data[p] = (vals, times)
    
    fig = plot_pdp_grid(pdp_data, {'mass': 'kg', 'e_deploy': 'MJ'})
    st.pyplot(fig)
    
    # Quick insights
    st.markdown("### Quick Insights")
    
    insights_col1, insights_col2, insights_col3 = st.columns(3)
    
    with insights_col1:
        mass_vals, mass_times = pdp_data['mass']
        mass_sensitivity = (mass_times[-1] - mass_times[0]) / (mass_vals[-1] - mass_vals[0])
        st.markdown(f"""
        <div class="insight-box">
            <div class="insight-title">Mass Sensitivity</div>
            <div style="color: #0f172a; font-size: 1.125rem; font-weight: 600;">{mass_sensitivity:.4f} s/kg</div>
            <div style="color: #64748b; font-size: 0.7rem;">Per kilogram change</div>
        </div>
        """, unsafe_allow_html=True)
    
    with insights_col2:
        cd_vals, cd_times = pdp_data['c_d']
        cd_sensitivity = (cd_times[-1] - cd_times[0]) / (cd_vals[-1] - cd_vals[0])
        st.markdown(f"""
        <div class="insight-box">
            <div class="insight-title">Drag Sensitivity</div>
            <div style="color: #0f172a; font-size: 1.125rem; font-weight: 600;">{cd_sensitivity:.2f} s/C_D</div>
            <div style="color: #64748b; font-size: 0.7rem;">Per drag coefficient unit</div>
        </div>
        """, unsafe_allow_html=True)
    
    with insights_col3:
        cl_vals, cl_times = pdp_data['c_l']
        cl_sensitivity = (cl_times[-1] - cl_times[0]) / (cl_vals[-1] - cl_vals[0])
        st.markdown(f"""
        <div class="insight-box">
            <div class="insight-title">Downforce Sensitivity</div>
            <div style="color: #0f172a; font-size: 1.125rem; font-weight: 600;">{cl_sensitivity:.2f} s/C_L</div>
            <div style="color: #64748b; font-size: 0.7rem;">Per lift coefficient unit</div>
        </div>
        """, unsafe_allow_html=True)

with sens_tab3:
    st.markdown("### Physics Consistency Check")
    st.markdown("""
    <p style="color: #64748b; font-size: 0.8rem; margin-bottom: 1rem;">
        Validates that the ML model respects fundamental physical relationships.
    </p>
    """, unsafe_allow_html=True)
    
    mono_df = check_monotonicity(predictor, setups[0][1])
    
    for _, row in mono_df.iterrows():
        status = row['Status']
        param = row['Parameter']
        expected = row['Expected']
        observed = row['Observed']
        
        if status == 'OK':
            status_class = "status-valid"
        else:
            status_class = "status-warning"
        
        st.markdown(f"""
        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 6px; padding: 1rem; margin-bottom: 0.5rem;
                    display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div style="color: #334155; font-weight: 600;">{param}</div>
                <div style="color: #64748b; font-size: 0.7rem;">{expected}</div>
            </div>
            <div style="text-align: right;">
                <div style="color: #64748b; font-size: 0.8rem;">{observed}</div>
                <span class="status-badge {status_class}">{status}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Summary
    ok_count = len(mono_df[mono_df['Status'] == 'OK'])
    total_count = len(mono_df)
    
    st.markdown(f"""
    <div style="background: #dcfce7;
                border: 1px solid #bbf7d0; border-radius: 6px; padding: 1rem; margin-top: 1rem;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div style="color: #166534; font-weight: 600;">Physics Consistency Score</div>
                <div style="color: #64748b; font-size: 0.7rem;">Model respects {ok_count}/{total_count} physical relationships</div>
            </div>
            <div style="font-size: 1.75rem; font-weight: 700; color: #166534;">{ok_count/total_count*100:.0f}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.divider()


# ----------------------------------------------------------------------------
# TRADE-OFF ANALYSIS - ENHANCED
# ----------------------------------------------------------------------------

st.markdown("## Trade-Off Analysis")

st.markdown("""
<p style="color: #64748b; font-size: 0.8rem; margin-bottom: 1rem;">
    Explore the relationship between key parameters and lap time to find optimal configurations.
</p>
""", unsafe_allow_html=True)

trade_col1, trade_col2, trade_col3 = st.columns(3)

with trade_col1:
    st.markdown("### Mass vs Performance")
    df_mass = compute_tradeoff_curve(predictor, setups[0][1], 'mass')
    fig = plot_tradeoff(df_mass['mass'].values, df_mass['Lap Time (s)'].values, "Mass (kg)", title="")
    st.pyplot(fig)
    
    # Find optimal
    opt_idx = df_mass['Lap Time (s)'].idxmin()
    opt_mass = df_mass.loc[opt_idx, 'mass']
    opt_time = df_mass.loc[opt_idx, 'Lap Time (s)']
    st.markdown(f"""
    <div style="background: #dcfce7; border: 1px solid #bbf7d0; border-radius: 6px; padding: 0.75rem; text-align: center;">
        <div style="color: #166534; font-size: 0.7rem; text-transform: uppercase;">Optimal Mass</div>
        <div style="color: #0f172a; font-weight: 600;">{opt_mass:.0f} kg | {opt_time:.3f}s</div>
    </div>
    """, unsafe_allow_html=True)

with trade_col2:
    st.markdown("### Energy Deployment")
    df_energy = compute_tradeoff_curve(predictor, setups[0][1], 'e_deploy')
    fig = plot_tradeoff(df_energy['e_deploy'].values, df_energy['Lap Time (s)'].values, "E_deploy (MJ)", title="")
    st.pyplot(fig)
    
    opt_idx = df_energy['Lap Time (s)'].idxmin()
    opt_energy = df_energy.loc[opt_idx, 'e_deploy']
    opt_time = df_energy.loc[opt_idx, 'Lap Time (s)']
    st.markdown(f"""
    <div style="background: #dbeafe; border: 1px solid #bfdbfe; border-radius: 6px; padding: 0.75rem; text-align: center;">
        <div style="color: #1d4ed8; font-size: 0.7rem; text-transform: uppercase;">Optimal Energy</div>
        <div style="color: #0f172a; font-weight: 600;">{opt_energy:.1f} MJ | {opt_time:.3f}s</div>
    </div>
    """, unsafe_allow_html=True)

with trade_col3:
    st.markdown("### Aero Efficiency")
    df_aero = compute_aero_efficiency_curve(predictor, setups[0][1])
    fig = plot_tradeoff(df_aero['Aero Efficiency (C_L/C_D)'].values, df_aero['Lap Time (s)'].values, "C_L / C_D", title="")
    st.pyplot(fig)
    
    opt_idx = df_aero['Lap Time (s)'].idxmin()
    opt_aero = df_aero.loc[opt_idx, 'Aero Efficiency (C_L/C_D)']
    opt_time = df_aero.loc[opt_idx, 'Lap Time (s)']
    st.markdown(f"""
    <div style="background: #ede9fe; border: 1px solid #ddd6fe; border-radius: 6px; padding: 0.75rem; text-align: center;">
        <div style="color: #6d28d9; font-size: 0.7rem; text-transform: uppercase;">Optimal Efficiency</div>
        <div style="color: #0f172a; font-weight: 600;">{opt_aero:.2f} | {opt_time:.3f}s</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()


# ----------------------------------------------------------------------------
# MODEL INFORMATION - ENHANCED
# ----------------------------------------------------------------------------

st.markdown("## Model Information")

model_col1, model_col2 = st.columns([1, 1])

with model_col1:
    st.markdown("### Model Specifications")
    
    stats = get_model_stats(predictor)
    
    specs = [
        ("Model Type", stats['model_name'].upper()),
        ("Training Samples", "100,000"),
        ("Cross-Validation RMSE", f"{stats['cv_rmse']:.4f} seconds"),
        ("95% Confidence Interval", f"+/- {stats['uncertainty_95']:.3f} seconds"),
        ("Mean Error Estimate", f"{stats['mean_error']:.4f} seconds"),
    ]
    
    for label, value in specs:
        st.markdown(f"""
        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 6px; padding: 0.75rem; margin-bottom: 0.5rem;
                    display: flex; justify-content: space-between; align-items: center;">
            <span style="color: #64748b;">{label}</span>
            <span style="color: #0f172a; font-weight: 600;">{value}</span>
        </div>
        """, unsafe_allow_html=True)

with model_col2:
    st.markdown("### Prediction Confidence")
    
    for (name, _), pred in zip(setups, predictions):
        u = pred['uncertainty']
        lt = pred['lap_time']
        is_valid = pred['is_valid']
        is_ood = pred['is_ood']
        
        if is_valid and not is_ood:
            confidence = "HIGH"
            conf_color = "#166534"
        elif is_valid:
            confidence = "MEDIUM"
            conf_color = "#b45309"
        else:
            confidence = "LOW"
            conf_color = "#dc2626"
        
        st.markdown(f"""
        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 6px; padding: 1rem; margin-bottom: 0.5rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <span style="color: #334155; font-weight: 600;">{name}</span>
                <span style="color: {conf_color}; font-weight: 600; font-size: 0.7rem;">{confidence}</span>
            </div>
            <div style="color: #64748b; font-size: 0.8rem;">
                {format_laptime(lt)} +/- {u:.3f}s (95% CI: {lt - u*1.96:.3f}s - {lt + u*1.96:.3f}s)
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show warnings if any
        if pred['warnings']:
            for warning in pred['warnings']:
                st.markdown(f"""
                <div style="background: #fee2e2; border: 1px solid #fecaca;
                            border-radius: 4px; padding: 0.5rem; margin-top: 0.25rem; font-size: 0.7rem; color: #dc2626;">
                    {warning}
                </div>
                """, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# FOOTER
# ----------------------------------------------------------------------------

st.divider()

footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown("""
    <div style="text-align: center; color: #64748b; font-size: 0.7rem;">
        <div style="margin-bottom: 0.25rem; font-weight: 600;">F1 Track Analytics</div>
        <div>Spa-Francorchamps Circuit</div>
    </div>
    """, unsafe_allow_html=True)

with footer_col2:
    timestamp = st.session_state.get('analysis_timestamp', 'N/A')
    st.markdown(f"""
    <div style="text-align: center; color: #64748b; font-size: 0.7rem;">
        <div style="margin-bottom: 0.25rem; font-weight: 600;">Last Analysis</div>
        <div>{timestamp}</div>
    </div>
    """, unsafe_allow_html=True)

with footer_col3:
    st.markdown("""
    <div style="text-align: center; color: #64748b; font-size: 0.7rem;">
        <div style="margin-bottom: 0.25rem; font-weight: 600;">Powered by</div>
        <div>XGBoost ML Engine</div>
    </div>
    """, unsafe_allow_html=True)
