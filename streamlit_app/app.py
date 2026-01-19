"""
Formula1 Analytics Dashboard
Professional, research-grade setup-based analysis for track time prediction.

Usage:
    streamlit run streamlit_app/app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from inference import F1Predictor, Setup, DEFAULT_SETUP, PARAM_BOUNDS
from analytics import (
    compare_setups, create_summary_table, create_segment_table,
    check_monotonicity, compute_tradeoff_curve, compute_aero_efficiency_curve,
    get_model_stats
)
from plots import (
    plot_sector_comparison, plot_segment_deltas, plot_cumulative_time,
    plot_feature_importance, plot_partial_dependence, plot_pdp_grid,
    plot_tradeoff, COLORS
)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_laptime(seconds: float) -> str:
    """Convert seconds to F1 lap time format (M:SS.sss)."""
    minutes = int(seconds // 60)
    remaining = seconds % 60
    return f"{minutes}:{remaining:06.3f}"


def format_delta(delta_seconds: float) -> str:
    """Format delta time with +/- prefix."""
    if abs(delta_seconds) < 0.001:
        return "-"
    sign = "+" if delta_seconds > 0 else ""
    return f"{sign}{delta_seconds:.3f}s"


# ============================================================================
# PRELOADED SETUPS
# ============================================================================

PRELOADED_SETUPS = {
    "Balanced (Reference)": Setup(
        mass=780.0, c_l=1.2, c_d=1.0, alpha_elec=0.15, e_deploy=3.0, gamma_cool=1.0
    ),
    "Low Drag (Monza Spec)": Setup(
        mass=770.0, c_l=0.95, c_d=0.80, alpha_elec=0.18, e_deploy=3.2, gamma_cool=0.95
    ),
    "High Downforce (Monaco Spec)": Setup(
        mass=785.0, c_l=1.45, c_d=1.25, alpha_elec=0.12, e_deploy=2.8, gamma_cool=1.05
    ),
    "Efficiency Focus": Setup(
        mass=775.0, c_l=1.15, c_d=0.90, alpha_elec=0.20, e_deploy=3.5, gamma_cool=1.0
    ),
    "Power Priority": Setup(
        mass=780.0, c_l=1.10, c_d=0.95, alpha_elec=0.25, e_deploy=3.8, gamma_cool=0.90
    ),
    "Conservative (Endurance)": Setup(
        mass=790.0, c_l=1.25, c_d=1.05, alpha_elec=0.10, e_deploy=2.5, gamma_cool=1.15
    ),
    "Custom": None  # Placeholder for custom input
}


# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="F1 Track Analytics",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Light Mode CSS
st.markdown("""
<style>
    /* Light mode background */
    .main {background-color: #FFFFFF;}
    .stApp {background-color: #FFFFFF;}
    
    /* Headers */
    h1 {color: #1a1a2e !important; font-weight: 600 !important;}
    h2 {color: #2d2d44 !important; border-bottom: 1px solid #e0e0e0; padding-bottom: 8px; font-weight: 500 !important;}
    h3 {color: #1a1a2e !important; font-weight: 500 !important;}
    
    /* Metrics styling */
    .stMetric {
        background-color: #f8f9fa; 
        padding: 16px; 
        border-radius: 4px; 
        border: 1px solid #e9ecef;
    }
    .stMetric label {color: #6c757d !important; font-size: 12px !important;}
    .stMetric [data-testid="stMetricValue"] {color: #1a1a2e !important; font-weight: 600 !important;}
    
    /* Tables */
    .stDataFrame {font-size: 13px;}
    
    /* Sidebar */
    [data-testid="stSidebar"] {background-color: #f8f9fa;}
    [data-testid="stSidebar"] h2 {border-bottom: 1px solid #dee2e6;}
    
    /* Buttons */
    .stButton > button {
        background-color: #1a1a2e;
        color: white;
        border: none;
        font-weight: 500;
    }
    .stButton > button:hover {
        background-color: #2d2d44;
        color: white;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border-radius: 4px;
    }
    
    /* Remove default padding */
    .block-container {padding-top: 2rem;}
</style>
""", unsafe_allow_html=True)


# ============================================================================
# SESSION STATE
# ============================================================================

if 'predictor' not in st.session_state:
    try:
        st.session_state.predictor = F1Predictor()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

if 'setups' not in st.session_state:
    st.session_state.setups = []

if 'predictions' not in st.session_state:
    st.session_state.predictions = []


# ============================================================================
# SIDEBAR - SETUP DEFINITION
# ============================================================================

with st.sidebar:
    st.markdown("## Setup Configuration")
    st.markdown("---")
    
    # Number of setups
    n_setups = st.selectbox(
        "Number of Configurations",
        options=[1, 2, 3, 4],
        index=1,
        help="Compare up to 4 different vehicle configurations"
    )
    
    setups = []
    
    for i in range(n_setups):
        setup_label = "Baseline" if i == 0 else f"Comparison {i}"
        
        with st.expander(f"{setup_label}", expanded=(i < 2)):
            
            # Preset selector
            preset_options = list(PRELOADED_SETUPS.keys())
            default_preset = 0 if i == 0 else min(i, len(preset_options)-2)
            
            selected_preset = st.selectbox(
                "Preset Configuration",
                options=preset_options,
                index=default_preset,
                key=f"preset_{i}"
            )
            
            if selected_preset != "Custom" and PRELOADED_SETUPS[selected_preset]:
                preset = PRELOADED_SETUPS[selected_preset]
                default_mass = preset.mass
                default_cl = preset.c_l
                default_cd = preset.c_d
                default_alpha = preset.alpha_elec
                default_edeploy = preset.e_deploy
                default_gamma = preset.gamma_cool
            else:
                default_mass = 780.0
                default_cl = 1.2
                default_cd = 1.0
                default_alpha = 0.15
                default_edeploy = 3.0
                default_gamma = 1.0
            
            st.markdown("**Parameters**")
            
            mass = st.slider(
                "Mass (kg)",
                min_value=700.0, max_value=850.0,
                value=default_mass,
                step=5.0,
                key=f"mass_{i}"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                c_l = st.slider(
                    "C_L",
                    min_value=0.8, max_value=1.5,
                    value=default_cl,
                    step=0.05,
                    key=f"c_l_{i}"
                )
            with col2:
                c_d = st.slider(
                    "C_D",
                    min_value=0.7, max_value=1.3,
                    value=default_cd,
                    step=0.05,
                    key=f"c_d_{i}"
                )
            
            alpha_elec = st.slider(
                "Electric Fraction",
                min_value=0.0, max_value=0.4,
                value=default_alpha,
                step=0.01,
                key=f"alpha_{i}"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                e_deploy = st.slider(
                    "E_deploy (MJ)",
                    min_value=2.0, max_value=4.0,
                    value=default_edeploy,
                    step=0.1,
                    key=f"e_deploy_{i}"
                )
            with col2:
                gamma_cool = st.slider(
                    "Cooling",
                    min_value=0.8, max_value=1.2,
                    value=default_gamma,
                    step=0.02,
                    key=f"gamma_{i}"
                )
            
            setup = Setup(
                mass=mass,
                c_l=c_l,
                c_d=c_d,
                alpha_elec=alpha_elec,
                e_deploy=e_deploy,
                gamma_cool=gamma_cool
            )
            setups.append((setup_label, setup))
    
    st.markdown("---")
    
    run_analysis = st.button("Run Analysis", use_container_width=True, type="primary")
    
    if run_analysis:
        st.session_state.setups = setups
        st.session_state.predictions = [
            st.session_state.predictor.predict(s) for _, s in setups
        ]


# ============================================================================
# MAIN CONTENT
# ============================================================================

st.title("F1 Track Analytics")
st.markdown("**Spa-Francorchamps Circuit** | ML-Based Lap Time Prediction")
st.markdown("---")

# Check if analysis has been run
if not st.session_state.predictions:
    st.info("Configure setups in the sidebar and click **Run Analysis** to begin.")
    st.stop()

setups = st.session_state.setups
predictions = st.session_state.predictions
predictor = st.session_state.predictor


# ============================================================================
# SECTION 1: LAP TIME OVERVIEW
# ============================================================================

st.header("Lap Time Overview")

# Primary metrics row
cols = st.columns(len(setups))
for col, (name, setup), pred in zip(cols, setups, predictions):
    with col:
        lap_formatted = format_laptime(pred['lap_time'])
        
        if name != "Baseline":
            delta = pred['lap_time'] - predictions[0]['lap_time']
            delta_str = format_delta(delta)
        else:
            delta_str = None
        
        st.metric(
            label=name,
            value=lap_formatted,
            delta=delta_str,
            delta_color="inverse"
        )

st.markdown("")

# Sector times table
st.markdown("### Sector Times")

sector_data = []
for (name, _), pred in zip(setups, predictions):
    baseline_pred = predictions[0]
    
    if name == "Baseline":
        s1_delta = s2_delta = s3_delta = "-"
    else:
        s1_delta = format_delta(pred['sector_1'] - baseline_pred['sector_1'])
        s2_delta = format_delta(pred['sector_2'] - baseline_pred['sector_2'])
        s3_delta = format_delta(pred['sector_3'] - baseline_pred['sector_3'])
    
    sector_data.append({
        'Configuration': name,
        'Sector 1': format_laptime(pred['sector_1']),
        'S1 Delta': s1_delta,
        'Sector 2': format_laptime(pred['sector_2']),
        'S2 Delta': s2_delta,
        'Sector 3': format_laptime(pred['sector_3']),
        'S3 Delta': s3_delta,
        'Lap Time': format_laptime(pred['lap_time'])
    })

sector_df = pd.DataFrame(sector_data)
st.dataframe(sector_df, use_container_width=True, hide_index=True)

# Warnings
for (name, _), pred in zip(setups, predictions):
    if not pred['is_valid']:
        st.warning(f"**{name}**: Parameters out of valid range - {', '.join(pred['warnings'])}")
    if pred['is_ood']:
        st.warning(f"**{name}**: Out-of-distribution input (severity: {pred['ood_severity']:.2f})")

st.markdown("---")


# ============================================================================
# SECTION 2: SEGMENT ANALYSIS
# ============================================================================

st.header("Segment Analysis")

# Segment names for Spa
SEGMENT_NAMES = [
    "La Source", "Eau Rouge Approach", "Eau Rouge", "Raidillon", "Kemmel Straight",
    "Les Combes Entry", "Les Combes Exit", "Malmedy", "Rivage", "Pre-Pouhon",
    "Pouhon", "Fagnes", "Campus Straight", "Campus", "Stavelot",
    "Paul Frere", "Blanchimont", "Chicane Approach", "Bus Stop", "Start/Finish"
]

col1, col2 = st.columns([1.5, 1])

with col1:
    st.markdown("### Segment Times")
    
    segment_data = {'Segment': [f"{i+1}. {name}" for i, name in enumerate(SEGMENT_NAMES)]}
    
    for (setup_name, _), pred in zip(setups, predictions):
        segment_data[setup_name] = [f"{t:.3f}s" for t in pred['segment_times']]
    
    # Add deltas
    if len(predictions) > 1:
        baseline_times = predictions[0]['segment_times']
        for (setup_name, _), pred in zip(setups[1:], predictions[1:]):
            deltas = [p - b for p, b in zip(pred['segment_times'], baseline_times)]
            segment_data[f"{setup_name} Delta"] = [format_delta(d) for d in deltas]
    
    segment_df = pd.DataFrame(segment_data)
    st.dataframe(segment_df, use_container_width=True, hide_index=True, height=450)

with col2:
    st.markdown("### Cumulative Time")
    fig = plot_cumulative_time(
        [name for name, _ in setups],
        [pred['segment_times'] for pred in predictions]
    )
    st.pyplot(fig)

# Segment delta visualization
if len(setups) > 1:
    st.markdown("### Segment Delta Analysis")
    
    comparison_idx = st.selectbox(
        "Compare configuration:",
        options=range(1, len(setups)),
        format_func=lambda x: setups[x][0],
        index=0
    )
    
    segment_deltas = [
        predictions[comparison_idx]['segment_times'][i] - predictions[0]['segment_times'][i]
        for i in range(20)
    ]
    
    fig = plot_segment_deltas(segment_deltas, SEGMENT_NAMES)
    st.pyplot(fig)

st.markdown("---")


# ============================================================================
# SECTION 3: SENSITIVITY ANALYSIS
# ============================================================================

st.header("Sensitivity Analysis")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Feature Importance")
    
    _ = predictor.predict(setups[0][1])
    importance_df = predictor.get_feature_importance()
    
    fig = plot_feature_importance(importance_df, top_n=12)
    st.pyplot(fig)

with col2:
    st.markdown("### Physical Consistency")
    st.markdown("Verification that model respects expected physical relationships:")
    
    monotonicity_df = check_monotonicity(predictor, setups[0][1])
    
    def highlight_status(val):
        if val == 'OK':
            return 'background-color: #d4edda; color: #155724;'
        elif val == 'WARN':
            return 'background-color: #f8d7da; color: #721c24;'
        return ''
    
    st.dataframe(
        monotonicity_df.style.applymap(highlight_status, subset=['Status']),
        use_container_width=True,
        hide_index=True
    )

# Partial Dependence
st.markdown("### Partial Dependence Plots")
st.markdown("Effect of each parameter on predicted lap time (other parameters held constant):")

pdp_params = ['mass', 'c_l', 'c_d', 'alpha_elec', 'e_deploy', 'gamma_cool']
param_units = {'mass': 'kg', 'c_l': '', 'c_d': '', 'alpha_elec': '', 'e_deploy': 'MJ', 'gamma_cool': ''}

pdp_data = {}
for param in pdp_params:
    values, lap_times = predictor.compute_partial_dependence(param, setups[0][1])
    pdp_data[param] = (values, lap_times)

fig = plot_pdp_grid(pdp_data, param_units)
st.pyplot(fig)

st.markdown("---")


# ============================================================================
# SECTION 4: TRADE-OFF ANALYSIS
# ============================================================================

st.header("Trade-Off Analysis")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Mass vs Lap Time")
    tradeoff_df = compute_tradeoff_curve(predictor, setups[0][1], 'mass')
    fig = plot_tradeoff(
        tradeoff_df['mass'].values,
        tradeoff_df['Lap Time (s)'].values,
        x_label="Mass (kg)",
        title="Mass Impact"
    )
    st.pyplot(fig)

with col2:
    st.markdown("### Energy vs Lap Time")
    tradeoff_df = compute_tradeoff_curve(predictor, setups[0][1], 'e_deploy')
    fig = plot_tradeoff(
        tradeoff_df['e_deploy'].values,
        tradeoff_df['Lap Time (s)'].values,
        x_label="E_deploy (MJ)",
        title="Energy Deployment Impact"
    )
    st.pyplot(fig)

with col3:
    st.markdown("### Aero Efficiency")
    aero_df = compute_aero_efficiency_curve(predictor, setups[0][1])
    fig = plot_tradeoff(
        aero_df['Aero Efficiency (C_L/C_D)'].values,
        aero_df['Lap Time (s)'].values,
        x_label="C_L / C_D",
        title="Aerodynamic Efficiency"
    )
    st.pyplot(fig)

st.markdown("---")


# ============================================================================
# SECTION 5: MODEL INFORMATION
# ============================================================================

st.header("Model Confidence")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Model Specifications")
    
    stats = get_model_stats(predictor)
    
    spec_data = {
        'Metric': ['Model Type', 'Cross-Validation RMSE', 'Mean Prediction Error', '95% Confidence Interval', 'Training Samples'],
        'Value': [
            stats['model_name'].upper(),
            f"{stats['cv_rmse']:.4f} seconds",
            f"+/- {stats['mean_error']:.3f} seconds",
            f"+/- {stats['uncertainty_95']:.3f} seconds",
            "100,000"
        ]
    }
    st.dataframe(pd.DataFrame(spec_data), use_container_width=True, hide_index=True)

with col2:
    st.markdown("### Prediction Confidence")
    
    for (name, _), pred in zip(setups, predictions):
        uncertainty = pred['uncertainty']
        lap = pred['lap_time']
        
        ci_low = format_laptime(lap - 1.96*uncertainty)
        ci_high = format_laptime(lap + 1.96*uncertainty)
        
        if pred['is_valid'] and not pred['is_ood']:
            status = "High confidence"
            status_color = "#155724"
        elif pred['is_ood']:
            status = "Reduced confidence (OOD)"
            status_color = "#856404"
        else:
            status = "Low confidence"
            status_color = "#721c24"
        
        st.markdown(f"""
        **{name}**: {format_laptime(lap)}  
        95% CI: [{ci_low}, {ci_high}]  
        <span style="color: {status_color};">{status}</span>
        """, unsafe_allow_html=True)

st.markdown("---")


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("""
<div style='text-align: center; color: #6c757d; font-size: 11px; padding: 20px; border-top: 1px solid #e9ecef;'>
    F1 Track Analytics | Spa-Francorchamps Circuit | XGBoost Model (CV RMSE: 0.547s)
</div>
""", unsafe_allow_html=True)
