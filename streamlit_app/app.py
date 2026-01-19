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
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enterprise-grade CSS styling
st.markdown("""
<style>
    /* ===== GLOBAL STYLES ===== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    /* ===== SIDEBAR ===== */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%) !important;
        border-right: 1px solid rgba(148, 163, 184, 0.1) !important;
    }
    
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label {
        color: #e2e8f0 !important;
    }
    
    /* ===== TYPOGRAPHY ===== */
    .stApp h1 {
        color: #f8fafc !important;
        font-weight: 700 !important;
        font-size: 2rem !important;
        letter-spacing: -0.025em !important;
    }
    
    .stApp h2 {
        color: #f1f5f9 !important;
        font-weight: 600 !important;
        font-size: 1.25rem !important;
        border-bottom: 1px solid rgba(148, 163, 184, 0.2) !important;
        padding-bottom: 0.75rem !important;
        margin-top: 1.5rem !important;
    }
    
    .stApp h3 {
        color: #e2e8f0 !important;
        font-weight: 500 !important;
        font-size: 1rem !important;
    }
    
    .stApp p, .stApp span, .stApp label, .stApp div {
        color: #cbd5e1 !important;
    }
    
    /* ===== METRIC CARDS ===== */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%) !important;
        border: 1px solid rgba(148, 163, 184, 0.15) !important;
        border-radius: 12px !important;
        padding: 1.25rem !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -2px rgba(0, 0, 0, 0.2) !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    }
    
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.4), 0 4px 6px -4px rgba(0, 0, 0, 0.3) !important;
    }
    
    [data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-size: 1.875rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.025em !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-weight: 500 !important;
        text-transform: uppercase !important;
        font-size: 0.75rem !important;
        letter-spacing: 0.05em !important;
    }
    
    [data-testid="stMetricDelta"] {
        font-weight: 600 !important;
    }
    
    /* ===== DATA TABLES ===== */
    .stDataFrame {
        background: rgba(30, 41, 59, 0.6) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(148, 163, 184, 0.1) !important;
        overflow: hidden !important;
    }
    
    .stDataFrame table {
        color: #e2e8f0 !important;
    }
    
    .stDataFrame th {
        background: rgba(15, 23, 42, 0.8) !important;
        color: #94a3b8 !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        font-size: 0.7rem !important;
        letter-spacing: 0.05em !important;
        padding: 0.75rem 1rem !important;
    }
    
    .stDataFrame td {
        color: #e2e8f0 !important;
        padding: 0.75rem 1rem !important;
        border-bottom: 1px solid rgba(148, 163, 184, 0.1) !important;
    }
    
    .stDataFrame tr:hover td {
        background: rgba(51, 65, 85, 0.5) !important;
    }
    
    /* ===== BUTTONS ===== */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.625rem 1.25rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.3) !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 10px -2px rgba(37, 99, 235, 0.4) !important;
    }
    
    /* ===== EXPANDERS ===== */
    details {
        background: rgba(30, 41, 59, 0.6) !important;
        border: 1px solid rgba(148, 163, 184, 0.15) !important;
        border-radius: 8px !important;
        margin-bottom: 0.5rem !important;
    }
    
    details summary {
        color: #e2e8f0 !important;
        font-weight: 500 !important;
        padding: 0.75rem !important;
    }
    
    details summary:hover {
        background: rgba(51, 65, 85, 0.5) !important;
    }
    
    /* ===== SLIDERS ===== */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6) !important;
    }
    
    .stSlider label {
        color: #94a3b8 !important;
        font-weight: 500 !important;
    }
    
    /* ===== TABS ===== */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(15, 23, 42, 0.6) !important;
        border-radius: 10px !important;
        padding: 4px !important;
        gap: 4px !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: #94a3b8 !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        padding: 0.5rem 1rem !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
        color: white !important;
    }
    
    /* ===== DIVIDERS ===== */
    hr {
        border-color: rgba(148, 163, 184, 0.15) !important;
        margin: 1.5rem 0 !important;
    }
    
    /* ===== ALERTS ===== */
    .stAlert {
        background: rgba(30, 41, 59, 0.8) !important;
        border: 1px solid rgba(148, 163, 184, 0.2) !important;
        border-radius: 10px !important;
    }
    
    /* ===== SELECTBOX ===== */
    .stSelectbox > div > div {
        background: rgba(30, 41, 59, 0.8) !important;
        border: 1px solid rgba(148, 163, 184, 0.2) !important;
        border-radius: 8px !important;
        color: #e2e8f0 !important;
    }
    
    /* ===== CUSTOM CLASSES ===== */
    .kpi-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    
    .kpi-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #f8fafc;
        line-height: 1.2;
    }
    
    .kpi-label {
        font-size: 0.75rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 0.5rem;
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .status-valid {
        background: rgba(16, 185, 129, 0.2);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .status-warning {
        background: rgba(245, 158, 11, 0.2);
        color: #F59E0B;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    
    .status-error {
        background: rgba(239, 68, 68, 0.2);
        color: #EF4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    .header-subtitle {
        color: #64748b;
        font-size: 0.875rem;
        margin-top: -0.5rem;
    }
    
    .section-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    
    .delta-positive {
        color: #EF4444 !important;
        font-weight: 600;
    }
    
    .delta-negative {
        color: #10B981 !important;
        font-weight: 600;
    }
    
    .insight-box {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .insight-title {
        color: #60a5fa;
        font-weight: 600;
        font-size: 0.875rem;
        margin-bottom: 0.5rem;
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
        <div style="font-size: 2.5rem; margin-bottom: 0.25rem;">🏎️</div>
        <div style="font-size: 1.25rem; font-weight: 700; color: #f8fafc; letter-spacing: -0.025em;">F1 Analytics</div>
        <div style="font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.1em;">Performance Lab</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Quick Stats
    stats = get_model_stats(st.session_state.predictor)
    st.markdown(f"""
    <div style="background: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59, 130, 246, 0.2); border-radius: 8px; padding: 0.75rem; margin-bottom: 1rem;">
        <div style="font-size: 0.7rem; color: #60a5fa; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.25rem;">Model Status</div>
        <div style="font-size: 0.875rem; color: #e2e8f0; font-weight: 500;">{stats['model_name'].upper()} • RMSE: {stats['cv_rmse']:.3f}s</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### ⚙️ Configuration")
    
    n_setups = st.selectbox("Number of Setups", [1, 2, 3], index=1, help="Compare up to 3 different vehicle configurations")
    
    st.divider()
    
    setups = []
    setup_colors = ['#3b82f6', '#10b981', '#f59e0b']
    
    for i in range(n_setups):
        label = "🔵 Baseline" if i == 0 else f"{'🟢' if i == 1 else '🟠'} Setup {chr(65+i)}"
        
        with st.expander(label, expanded=(i == 0)):
            st.markdown(f"""
            <div style="height: 3px; background: {setup_colors[i]}; border-radius: 2px; margin-bottom: 1rem;"></div>
            """, unsafe_allow_html=True)
            
            mass = st.slider("Mass (kg)", 700.0, 850.0, 780.0 + i*20, 5.0, key=f"m{i}",
                           help="Total vehicle mass including driver")
            
            st.markdown("**Aerodynamics**")
            c1, c2 = st.columns(2)
            c_l = c1.slider("C_L", 0.8, 1.5, 1.2, 0.05, key=f"cl{i}", help="Lift coefficient")
            c_d = c2.slider("C_D", 0.7, 1.3, 1.0, 0.05, key=f"cd{i}", help="Drag coefficient")
            
            st.markdown("**Powertrain**")
            alpha = st.slider("Electric Fraction", 0.0, 0.4, 0.15, 0.02, key=f"a{i}",
                            help="Fraction of power from electric motor")
            
            c1, c2 = st.columns(2)
            e_dep = c1.slider("E_deploy (MJ)", 2.0, 4.0, 3.0, 0.1, key=f"e{i}",
                             help="Deployable energy per lap")
            gamma = c2.slider("Cooling", 0.8, 1.2, 1.0, 0.05, key=f"g{i}",
                             help="Cooling system aggressiveness")
            
            # Show aero efficiency
            aero_eff = c_l / c_d if c_d > 0 else 0
            st.markdown(f"""
            <div style="background: rgba(148, 163, 184, 0.1); border-radius: 6px; padding: 0.5rem; margin-top: 0.5rem;">
                <span style="font-size: 0.75rem; color: #94a3b8;">Aero Efficiency:</span>
                <span style="font-size: 0.875rem; color: #e2e8f0; font-weight: 600; margin-left: 0.25rem;">{aero_eff:.2f}</span>
            </div>
            """, unsafe_allow_html=True)
            
            setups.append((label.split(' ', 1)[1] if i > 0 else "Baseline", Setup(mass, c_l, c_d, alpha, e_dep, gamma)))
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Reset", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key not in ['predictor']:
                    del st.session_state[key]
            st.rerun()
    
    with col2:
        if st.button("▶️ Analyze", use_container_width=True, type="primary"):
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
        <h1 style="margin: 0; font-size: 2rem;">🏁 Track Performance Analytics</h1>
        <p style="color: #64748b; margin-top: 0.5rem; font-size: 0.875rem;">
            Spa-Francorchamps Circuit • 7.004 km • Machine Learning Prediction Engine
        </p>
    </div>
    """, unsafe_allow_html=True)

if not st.session_state.predictions:
    # Welcome screen
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%); 
                border: 1px solid rgba(59, 130, 246, 0.2); border-radius: 16px; padding: 2rem; 
                text-align: center; margin: 2rem auto; max-width: 600px;">
        <div style="font-size: 3rem; margin-bottom: 1rem;">📊</div>
        <h2 style="color: #f8fafc; margin-bottom: 0.5rem; border: none;">Welcome to F1 Analytics</h2>
        <p style="color: #94a3b8; margin-bottom: 1.5rem;">
            Configure your vehicle setups in the sidebar and click <strong>Analyze</strong> to run performance predictions.
        </p>
        <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
            <div style="text-align: center;">
                <div style="font-size: 1.5rem; color: #3b82f6;">⚡</div>
                <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 0.25rem;">Real-time Analysis</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 1.5rem; color: #10b981;">📈</div>
                <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 0.25rem;">ML Predictions</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 1.5rem; color: #f59e0b;">🔬</div>
                <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 0.25rem;">Sensitivity Analysis</div>
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

st.markdown("## 📊 Executive Summary")

# Top KPI row
kpi_cols = st.columns(4)

# Best Lap Time
best_pred = min(predictions, key=lambda p: p['lap_time'])
best_setup_idx = predictions.index(best_pred)
best_setup_name = setups[best_setup_idx][0]

with kpi_cols[0]:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value" style="color: #10b981;">{format_laptime(best_pred['lap_time'])}</div>
        <div class="kpi-label">Best Lap Time</div>
        <div style="font-size: 0.75rem; color: #64748b; margin-top: 0.25rem;">{best_setup_name}</div>
    </div>
    """, unsafe_allow_html=True)

# Gap to Best (if multiple setups)
with kpi_cols[1]:
    if len(predictions) > 1:
        worst_pred = max(predictions, key=lambda p: p['lap_time'])
        gap = worst_pred['lap_time'] - best_pred['lap_time']
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value" style="color: #f59e0b;">{gap:.3f}s</div>
            <div class="kpi-label">Performance Spread</div>
            <div style="font-size: 0.75rem; color: #64748b; margin-top: 0.25rem;">Max gap between setups</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        perf_score = calculate_performance_score(best_pred)
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value" style="color: #3b82f6;">{perf_score}</div>
            <div class="kpi-label">Performance Score</div>
            <div style="font-size: 0.75rem; color: #64748b; margin-top: 0.25rem;">Out of 100</div>
        </div>
        """, unsafe_allow_html=True)

# Model Confidence
with kpi_cols[2]:
    avg_uncertainty = np.mean([p['uncertainty'] for p in predictions])
    confidence_pct = max(0, min(100, 100 - (avg_uncertainty * 20)))
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value" style="color: #8b5cf6;">{confidence_pct:.0f}%</div>
        <div class="kpi-label">Model Confidence</div>
        <div style="font-size: 0.75rem; color: #64748b; margin-top: 0.25rem;">±{avg_uncertainty:.3f}s uncertainty</div>
    </div>
    """, unsafe_allow_html=True)

# Validation Status
with kpi_cols[3]:
    all_valid = all(p['is_valid'] for p in predictions)
    any_ood = any(p['is_ood'] for p in predictions)
    
    if all_valid and not any_ood:
        status_emoji = "✅"
        status_text = "All Valid"
        status_color = "#10b981"
    elif all_valid:
        status_emoji = "⚠️"
        status_text = "OOD Warning"
        status_color = "#f59e0b"
    else:
        status_emoji = "❌"
        status_text = "Invalid Setup"
        status_color = "#ef4444"
    
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value" style="color: {status_color};">{status_emoji}</div>
        <div class="kpi-label">Validation Status</div>
        <div style="font-size: 0.75rem; color: #64748b; margin-top: 0.25rem;">{status_text}</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()


# ----------------------------------------------------------------------------
# LAP TIMES - DETAILED COMPARISON
# ----------------------------------------------------------------------------

st.markdown("## 🏁 Lap Time Analysis")

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
            delta_color = "#10b981" if d < 0 else "#ef4444" if d > 0 else "#64748b"
        
        fastest_badge = '<span class="status-badge status-valid" style="margin-left: 0.5rem;">FASTEST</span>' if is_fastest and len(setups) > 1 else ''
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.95) 100%);
                    border: 1px solid {setup_colors[idx]}40; border-radius: 12px; padding: 1.25rem;
                    border-top: 3px solid {setup_colors[idx]};">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 1rem;">
                <div style="font-size: 0.875rem; font-weight: 600; color: {setup_colors[idx]};">{name}</div>
                {fastest_badge}
            </div>
            <div style="font-size: 2rem; font-weight: 700; color: #f8fafc; letter-spacing: -0.025em;">
                {format_laptime(pred['lap_time'])}
            </div>
            <div style="font-size: 0.875rem; color: {delta_color}; margin-top: 0.25rem;">
                {delta_str if delta_str else 'Reference'}
            </div>
            <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(148, 163, 184, 0.1);">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span style="font-size: 0.75rem; color: #94a3b8;">S1</span>
                    <span style="font-size: 0.875rem; color: #e2e8f0;">{pred['sector_1']:.3f}s</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span style="font-size: 0.75rem; color: #94a3b8;">S2</span>
                    <span style="font-size: 0.875rem; color: #e2e8f0;">{pred['sector_2']:.3f}s</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span style="font-size: 0.75rem; color: #94a3b8;">S3</span>
                    <span style="font-size: 0.875rem; color: #e2e8f0;">{pred['sector_3']:.3f}s</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

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
    
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

st.divider()


# ----------------------------------------------------------------------------
# SEGMENT ANALYSIS - TABBED INTERFACE
# ----------------------------------------------------------------------------

st.markdown("## 🗺️ Circuit Segment Analysis")

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

tab1, tab2, tab3 = st.tabs(["📋 Segment Table", "📈 Progression Chart", "🔍 Delta Analysis"])

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
        
        st.dataframe(pd.DataFrame(seg_data), hide_index=True, use_container_width=True, height=450)
    
    with col2:
        # Segment type breakdown
        st.markdown("### Segment Categories")
        
        for seg_type, indices in SEGMENT_TYPES.items():
            type_time = sum(predictions[0]['segment_times'][i] for i in indices)
            pct = (type_time / predictions[0]['lap_time']) * 100
            
            if seg_type == "High-Speed":
                color = "#10b981"
                icon = "⚡"
            elif seg_type == "Braking Zone":
                color = "#ef4444"
                icon = "🛑"
            else:
                color = "#f59e0b"
                icon = "🔧"
            
            st.markdown(f"""
            <div style="background: rgba(30, 41, 59, 0.6); border-radius: 8px; padding: 0.75rem; margin-bottom: 0.5rem;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="color: #e2e8f0; font-weight: 500;">{icon} {seg_type}</span>
                    <span style="color: {color}; font-weight: 600;">{type_time:.2f}s ({pct:.1f}%)</span>
                </div>
                <div style="background: rgba(148, 163, 184, 0.2); border-radius: 4px; height: 4px; margin-top: 0.5rem;">
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
                <div class="insight-title">📉 Biggest Gain</div>
                <div style="color: #10b981; font-weight: 600;">{SEGMENTS[max_gain_seg]}</div>
                <div style="color: #94a3b8; font-size: 0.75rem;">Segment {max_gain_seg + 1}</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="insight-box">
                <div class="insight-title">📈 Biggest Loss</div>
                <div style="color: #ef4444; font-weight: 600;">{SEGMENTS[max_loss_seg]}</div>
                <div style="color: #94a3b8; font-size: 0.75rem;">Segment {max_loss_seg + 1}</div>
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

st.markdown("## 🔬 Model Sensitivity Analysis")

sens_tab1, sens_tab2, sens_tab3 = st.tabs(["📊 Feature Importance", "📈 Partial Dependence", "✅ Physics Validation"])

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
            <div class="insight-title">💡 Interpretation Guide</div>
            <p style="font-size: 0.8rem; color: #94a3b8; margin: 0;">
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
            <div style="background: rgba(30, 41, 59, 0.5); border-radius: 8px; padding: 0.75rem; margin-bottom: 0.5rem;">
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: #e2e8f0; font-weight: 500;">{feature}</span>
                    <span style="color: #3b82f6; font-weight: 600;">{importance:.3f}</span>
                </div>
                <div style="color: #64748b; font-size: 0.75rem; margin-top: 0.25rem;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

with sens_tab2:
    st.markdown("### Partial Dependence Plots")
    st.markdown("""
    <p style="color: #94a3b8; font-size: 0.875rem; margin-bottom: 1rem;">
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
            <div class="insight-title">⚖️ Mass Sensitivity</div>
            <div style="color: #f8fafc; font-size: 1.25rem; font-weight: 600;">{mass_sensitivity:.4f} s/kg</div>
            <div style="color: #94a3b8; font-size: 0.75rem;">Per kilogram change</div>
        </div>
        """, unsafe_allow_html=True)
    
    with insights_col2:
        cd_vals, cd_times = pdp_data['c_d']
        cd_sensitivity = (cd_times[-1] - cd_times[0]) / (cd_vals[-1] - cd_vals[0])
        st.markdown(f"""
        <div class="insight-box">
            <div class="insight-title">💨 Drag Sensitivity</div>
            <div style="color: #f8fafc; font-size: 1.25rem; font-weight: 600;">{cd_sensitivity:.2f} s/C_D</div>
            <div style="color: #94a3b8; font-size: 0.75rem;">Per drag coefficient unit</div>
        </div>
        """, unsafe_allow_html=True)
    
    with insights_col3:
        cl_vals, cl_times = pdp_data['c_l']
        cl_sensitivity = (cl_times[-1] - cl_times[0]) / (cl_vals[-1] - cl_vals[0])
        st.markdown(f"""
        <div class="insight-box">
            <div class="insight-title">🔽 Downforce Sensitivity</div>
            <div style="color: #f8fafc; font-size: 1.25rem; font-weight: 600;">{cl_sensitivity:.2f} s/C_L</div>
            <div style="color: #94a3b8; font-size: 0.75rem;">Per lift coefficient unit</div>
        </div>
        """, unsafe_allow_html=True)

with sens_tab3:
    st.markdown("### Physics Consistency Check")
    st.markdown("""
    <p style="color: #94a3b8; font-size: 0.875rem; margin-bottom: 1rem;">
        Validates that the ML model respects fundamental physical relationships.
    </p>
    """, unsafe_allow_html=True)
    
    mono_df = check_monotonicity(predictor, setups[0][1])
    
    # Enhanced display
    for _, row in mono_df.iterrows():
        status = row['Status']
        param = row['Parameter']
        expected = row['Expected']
        observed = row['Observed']
        
        if status == 'OK':
            status_class = "status-valid"
            icon = "✅"
        else:
            status_class = "status-warning"
            icon = "⚠️"
        
        st.markdown(f"""
        <div style="background: rgba(30, 41, 59, 0.5); border-radius: 8px; padding: 1rem; margin-bottom: 0.5rem;
                    display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div style="color: #e2e8f0; font-weight: 600;">{param}</div>
                <div style="color: #64748b; font-size: 0.75rem;">{expected}</div>
            </div>
            <div style="text-align: right;">
                <div style="color: #94a3b8; font-size: 0.875rem;">{observed}</div>
                <span class="status-badge {status_class}">{icon} {status}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Summary
    ok_count = len(mono_df[mono_df['Status'] == 'OK'])
    total_count = len(mono_df)
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(16, 185, 129, 0.05) 100%);
                border: 1px solid rgba(16, 185, 129, 0.2); border-radius: 10px; padding: 1rem; margin-top: 1rem;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div style="color: #10b981; font-weight: 600;">Physics Consistency Score</div>
                <div style="color: #94a3b8; font-size: 0.75rem;">Model respects {ok_count}/{total_count} physical relationships</div>
            </div>
            <div style="font-size: 2rem; font-weight: 700; color: #10b981;">{ok_count/total_count*100:.0f}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.divider()


# ----------------------------------------------------------------------------
# TRADE-OFF ANALYSIS - ENHANCED
# ----------------------------------------------------------------------------

st.markdown("## ⚖️ Trade-Off Analysis")

st.markdown("""
<p style="color: #94a3b8; font-size: 0.875rem; margin-bottom: 1rem;">
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
    <div style="background: rgba(16, 185, 129, 0.1); border-radius: 8px; padding: 0.75rem; text-align: center;">
        <div style="color: #10b981; font-size: 0.75rem; text-transform: uppercase;">Optimal Mass</div>
        <div style="color: #f8fafc; font-weight: 600;">{opt_mass:.0f} kg → {opt_time:.3f}s</div>
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
    <div style="background: rgba(59, 130, 246, 0.1); border-radius: 8px; padding: 0.75rem; text-align: center;">
        <div style="color: #3b82f6; font-size: 0.75rem; text-transform: uppercase;">Optimal Energy</div>
        <div style="color: #f8fafc; font-weight: 600;">{opt_energy:.1f} MJ → {opt_time:.3f}s</div>
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
    <div style="background: rgba(139, 92, 246, 0.1); border-radius: 8px; padding: 0.75rem; text-align: center;">
        <div style="color: #8b5cf6; font-size: 0.75rem; text-transform: uppercase;">Optimal Efficiency</div>
        <div style="color: #f8fafc; font-weight: 600;">{opt_aero:.2f} → {opt_time:.3f}s</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()


# ----------------------------------------------------------------------------
# MODEL INFORMATION - ENHANCED
# ----------------------------------------------------------------------------

st.markdown("## 🤖 Model Information")

model_col1, model_col2 = st.columns([1, 1])

with model_col1:
    st.markdown("### Model Specifications")
    
    stats = get_model_stats(predictor)
    
    specs = [
        ("Model Type", stats['model_name'].upper(), "🔧"),
        ("Training Samples", "100,000", "📊"),
        ("Cross-Validation RMSE", f"{stats['cv_rmse']:.4f} seconds", "📏"),
        ("95% Confidence Interval", f"± {stats['uncertainty_95']:.3f} seconds", "📐"),
        ("Mean Error Estimate", f"{stats['mean_error']:.4f} seconds", "📈"),
    ]
    
    for label, value, icon in specs:
        st.markdown(f"""
        <div style="background: rgba(30, 41, 59, 0.5); border-radius: 8px; padding: 0.75rem; margin-bottom: 0.5rem;
                    display: flex; justify-content: space-between; align-items: center;">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="font-size: 1rem;">{icon}</span>
                <span style="color: #94a3b8;">{label}</span>
            </div>
            <span style="color: #e2e8f0; font-weight: 600;">{value}</span>
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
            confidence = "High"
            conf_color = "#10b981"
            conf_icon = "🟢"
        elif is_valid:
            confidence = "Medium"
            conf_color = "#f59e0b"
            conf_icon = "🟡"
        else:
            confidence = "Low"
            conf_color = "#ef4444"
            conf_icon = "🔴"
        
        st.markdown(f"""
        <div style="background: rgba(30, 41, 59, 0.5); border-radius: 8px; padding: 1rem; margin-bottom: 0.5rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <span style="color: #e2e8f0; font-weight: 600;">{name}</span>
                <span style="color: {conf_color};">{conf_icon} {confidence} Confidence</span>
            </div>
            <div style="color: #94a3b8; font-size: 0.875rem;">
                {format_laptime(lt)} ± {u:.3f}s (95% CI: {lt - u*1.96:.3f}s - {lt + u*1.96:.3f}s)
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show warnings if any
        if pred['warnings']:
            for warning in pred['warnings']:
                st.markdown(f"""
                <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.2);
                            border-radius: 6px; padding: 0.5rem; margin-top: 0.25rem; font-size: 0.75rem; color: #ef4444;">
                    ⚠️ {warning}
                </div>
                """, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# FOOTER
# ----------------------------------------------------------------------------

st.divider()

footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown("""
    <div style="text-align: center; color: #64748b; font-size: 0.75rem;">
        <div style="margin-bottom: 0.25rem;">🏎️ F1 Track Analytics</div>
        <div>Spa-Francorchamps Circuit</div>
    </div>
    """, unsafe_allow_html=True)

with footer_col2:
    timestamp = st.session_state.get('analysis_timestamp', 'N/A')
    st.markdown(f"""
    <div style="text-align: center; color: #64748b; font-size: 0.75rem;">
        <div style="margin-bottom: 0.25rem;">🕐 Last Analysis</div>
        <div>{timestamp}</div>
    </div>
    """, unsafe_allow_html=True)

with footer_col3:
    st.markdown("""
    <div style="text-align: center; color: #64748b; font-size: 0.75rem;">
        <div style="margin-bottom: 0.25rem;">🤖 Powered by</div>
        <div>XGBoost ML Engine</div>
    </div>
    """, unsafe_allow_html=True)
