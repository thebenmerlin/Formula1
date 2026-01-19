"""
F1 Spa-Francorchamps Analytics Dashboard
Clean, readable, professional design
"""

import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
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


# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="F1 Spa Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Minimal clean CSS - forces readable colors
st.markdown("""
<style>
    /* Force white background everywhere */
    .stApp {background: white !important;}
    section[data-testid="stSidebar"] {background: #f5f5f5 !important;}
    
    /* Force black text everywhere */
    .stApp, .stApp p, .stApp span, .stApp label, .stApp div {
        color: #111111 !important;
    }
    
    /* Headers - dark blue */
    .stApp h1 {color: #0a2540 !important; font-size: 2.2rem !important;}
    .stApp h2 {color: #0a2540 !important; font-size: 1.5rem !important; border-bottom: 2px solid #0a2540; padding-bottom: 8px;}
    .stApp h3 {color: #0a2540 !important; font-size: 1.2rem !important;}
    
    /* Sidebar text */
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] div {
        color: #111111 !important;
    }
    
    /* Metric cards */
    [data-testid="stMetric"] {
        background: #f8f9fa !important;
        border: 1px solid #dee2e6 !important;
        border-radius: 8px !important;
        padding: 16px !important;
    }
    [data-testid="stMetricValue"] {
        color: #0a2540 !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #333333 !important;
    }
    [data-testid="stMetricDelta"] {
        color: #333333 !important;
    }
    
    /* Tables */
    .stDataFrame, table, th, td {
        color: #111111 !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: #0a2540 !important;
        color: white !important;
        border: none !important;
    }
    
    /* Expanders */
    details summary {
        color: #111111 !important;
        background: #f0f0f0 !important;
    }
    
    /* Slider labels */
    .stSlider label, .stSelectbox label {
        color: #111111 !important;
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


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.header("Vehicle Configuration")
    
    n_setups = st.selectbox("Compare Setups", [1, 2, 3], index=1)
    
    st.divider()
    
    setups = []
    for i in range(n_setups):
        label = "Baseline" if i == 0 else f"Setup {chr(65+i)}"
        
        with st.expander(label, expanded=True):
            mass = st.slider("Mass (kg)", 700.0, 850.0, 780.0 + i*20, 5.0, key=f"m{i}")
            
            c1, c2 = st.columns(2)
            c_l = c1.slider("C_L", 0.8, 1.5, 1.2, 0.05, key=f"cl{i}")
            c_d = c2.slider("C_D", 0.7, 1.3, 1.0, 0.05, key=f"cd{i}")
            
            alpha = st.slider("Electric Fraction", 0.0, 0.4, 0.15, 0.02, key=f"a{i}")
            
            c1, c2 = st.columns(2)
            e_dep = c1.slider("E_deploy (MJ)", 2.0, 4.0, 3.0, 0.1, key=f"e{i}")
            gamma = c2.slider("Cooling", 0.8, 1.2, 1.0, 0.05, key=f"g{i}")
            
            setups.append((label, Setup(mass, c_l, c_d, alpha, e_dep, gamma)))
    
    st.divider()
    
    if st.button("Run Analysis", use_container_width=True, type="primary"):
        st.session_state.setups = setups
        st.session_state.predictions = [
            st.session_state.predictor.predict(s) for _, s in setups
        ]


# ============================================================================
# MAIN PAGE
# ============================================================================

st.title("F1 Track Analytics")
st.markdown("**Spa-Francorchamps Circuit** — Machine Learning Lap Time Prediction")

st.divider()

if not st.session_state.predictions:
    st.info("Configure setups in the sidebar and click Run Analysis to begin.")
    st.stop()

setups = st.session_state.setups
predictions = st.session_state.predictions
predictor = st.session_state.predictor


# ----------------------------------------------------------------------------
# LAP TIMES
# ----------------------------------------------------------------------------

st.header("Lap Time Results")

cols = st.columns(len(setups))
for col, (name, _), pred in zip(cols, setups, predictions):
    with col:
        delta_str = None
        if name != "Baseline":
            d = pred['lap_time'] - predictions[0]['lap_time']
            delta_str = format_delta(d)
        st.metric(name, format_laptime(pred['lap_time']), delta=delta_str, delta_color="inverse")

st.markdown("#### Sector Breakdown")

rows = []
for (name, _), pred in zip(setups, predictions):
    base = predictions[0]
    row = {
        'Setup': name,
        'Sector 1': format_laptime(pred['sector_1']),
        'Sector 2': format_laptime(pred['sector_2']),
        'Sector 3': format_laptime(pred['sector_3']),
        'Total': format_laptime(pred['lap_time']),
        'Delta': format_delta(pred['lap_time'] - base['lap_time']) if name != "Baseline" else '-'
    }
    rows.append(row)

st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

st.divider()


# ----------------------------------------------------------------------------
# SEGMENT ANALYSIS
# ----------------------------------------------------------------------------

st.header("Segment Analysis")

SEGMENTS = [
    "La Source", "Eau Rouge Approach", "Eau Rouge", "Raidillon", "Kemmel Straight",
    "Les Combes Entry", "Les Combes Exit", "Malmedy", "Rivage", "Pre-Pouhon",
    "Pouhon", "Fagnes", "Campus Straight", "Campus", "Stavelot",
    "Paul Frere", "Blanchimont", "Chicane Approach", "Bus Stop", "Start/Finish"
]

left, right = st.columns([1.3, 1])

with left:
    seg_data = {'Segment': [f"{i+1}. {n}" for i, n in enumerate(SEGMENTS)]}
    for (name, _), pred in zip(setups, predictions):
        seg_data[name] = [f"{t:.3f}s" for t in pred['segment_times']]
    st.dataframe(pd.DataFrame(seg_data), hide_index=True, use_container_width=True, height=400)

with right:
    st.markdown("**Cumulative Time**")
    fig = plot_cumulative_time([n for n, _ in setups], [p['segment_times'] for p in predictions])
    st.pyplot(fig)

if len(setups) > 1:
    st.markdown("#### Segment Delta vs Baseline")
    deltas = [predictions[1]['segment_times'][i] - predictions[0]['segment_times'][i] for i in range(20)]
    fig = plot_segment_deltas(deltas, SEGMENTS)
    st.pyplot(fig)

st.divider()


# ----------------------------------------------------------------------------
# SENSITIVITY
# ----------------------------------------------------------------------------

st.header("Sensitivity Analysis")

left, right = st.columns(2)

with left:
    st.markdown("#### Feature Importance")
    _ = predictor.predict(setups[0][1])
    fig = plot_feature_importance(predictor.get_feature_importance(), top_n=10)
    st.pyplot(fig)

with right:
    st.markdown("#### Physical Consistency")
    mono_df = check_monotonicity(predictor, setups[0][1])
    st.dataframe(mono_df, hide_index=True, use_container_width=True)

st.markdown("#### Partial Dependence Plots")

pdp_data = {}
for p in ['mass', 'c_l', 'c_d', 'alpha_elec', 'e_deploy', 'gamma_cool']:
    vals, times = predictor.compute_partial_dependence(p, setups[0][1])
    pdp_data[p] = (vals, times)

fig = plot_pdp_grid(pdp_data, {'mass': 'kg', 'e_deploy': 'MJ'})
st.pyplot(fig)

st.divider()


# ----------------------------------------------------------------------------
# TRADE-OFFS
# ----------------------------------------------------------------------------

st.header("Trade-Off Analysis")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("**Mass vs Lap Time**")
    df = compute_tradeoff_curve(predictor, setups[0][1], 'mass')
    fig = plot_tradeoff(df['mass'].values, df['Lap Time (s)'].values, "Mass (kg)", title="")
    st.pyplot(fig)

with c2:
    st.markdown("**Energy Deployment**")
    df = compute_tradeoff_curve(predictor, setups[0][1], 'e_deploy')
    fig = plot_tradeoff(df['e_deploy'].values, df['Lap Time (s)'].values, "E_deploy (MJ)", title="")
    st.pyplot(fig)

with c3:
    st.markdown("**Aero Efficiency**")
    df = compute_aero_efficiency_curve(predictor, setups[0][1])
    fig = plot_tradeoff(df['Aero Efficiency (C_L/C_D)'].values, df['Lap Time (s)'].values, "C_L/C_D", title="")
    st.pyplot(fig)

st.divider()


# ----------------------------------------------------------------------------
# MODEL INFO
# ----------------------------------------------------------------------------

st.header("Model Information")

stats = get_model_stats(predictor)

left, right = st.columns(2)

with left:
    st.markdown(f"""
    **Model Type:** {stats['model_name'].upper()}  
    **Cross-Validation RMSE:** {stats['cv_rmse']:.4f} seconds  
    **95% Confidence Interval:** ± {stats['uncertainty_95']:.3f} seconds  
    **Training Samples:** 100,000
    """)

with right:
    for (name, _), pred in zip(setups, predictions):
        u = pred['uncertainty']
        lt = pred['lap_time']
        status = "High confidence" if pred['is_valid'] and not pred['is_ood'] else "Reduced confidence"
        st.markdown(f"**{name}:** {format_laptime(lt)} ± {u:.3f}s — {status}")

st.divider()
st.caption("F1 Track Analytics | Spa-Francorchamps | XGBoost Model")
