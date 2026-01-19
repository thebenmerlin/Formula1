"""
Plots Module for F1 Dashboard

All visualization functions with:
- Neutral color palette
- Minimal styling
- Research-grade presentation
- No animations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import List, Dict, Optional, Tuple

# Neutral, professional color palette
COLORS = {
    'primary': '#2C3E50',      # Dark blue-gray
    'secondary': '#7F8C8D',    # Gray
    'accent': '#3498DB',       # Blue
    'positive': '#27AE60',     # Green (good/faster)
    'negative': '#E74C3C',     # Red (bad/slower)
    'neutral': '#95A5A6',      # Light gray
    'background': '#FAFAFA',   # Off-white
    'grid': '#ECF0F1'          # Light gray grid
}

# Comparison setup colors
SETUP_COLORS = ['#2C3E50', '#3498DB', '#9B59B6', '#E67E22']


def apply_style(ax, title: str = None, xlabel: str = None, ylabel: str = None):
    """Apply consistent styling to axis."""
    ax.set_facecolor(COLORS['background'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLORS['secondary'])
    ax.spines['bottom'].set_color(COLORS['secondary'])
    ax.tick_params(colors=COLORS['primary'])
    ax.grid(True, alpha=0.3, color=COLORS['grid'])
    
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold', color=COLORS['primary'])
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10, color=COLORS['primary'])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, color=COLORS['primary'])


def plot_sector_comparison(
    setups: List[str],
    sector_times: List[List[float]]
) -> plt.Figure:
    """
    Bar chart comparing sector times across setups.
    
    Args:
        setups: List of setup names
        sector_times: List of [S1, S2, S3] times for each setup
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    x = np.arange(3)
    width = 0.8 / len(setups)
    
    for i, (name, times) in enumerate(zip(setups, sector_times)):
        offset = (i - len(setups)/2 + 0.5) * width
        bars = ax.bar(x + offset, times, width, label=name, color=SETUP_COLORS[i % len(SETUP_COLORS)])
        
        # Add value labels
        for bar, val in zip(bars, times):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=8, color=COLORS['primary'])
    
    ax.set_xticks(x)
    ax.set_xticklabels(['Sector 1', 'Sector 2', 'Sector 3'])
    ax.legend(loc='upper right', frameon=False)
    
    apply_style(ax, title='Sector Time Comparison', ylabel='Time (s)')
    
    plt.tight_layout()
    return fig


def plot_segment_deltas(
    segment_deltas: List[float],
    segment_names: List[str] = None
) -> plt.Figure:
    """
    Horizontal bar chart showing delta per segment.
    
    Args:
        segment_deltas: Delta for each segment (positive = slower)
        segment_names: Optional segment names
        
    Returns:
        matplotlib Figure
    """
    if segment_names is None:
        segment_names = [f"Seg {i+1}" for i in range(len(segment_deltas))]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    y = np.arange(len(segment_deltas))
    colors = [COLORS['negative'] if d > 0.001 else COLORS['positive'] if d < -0.001 else COLORS['neutral']
              for d in segment_deltas]
    
    bars = ax.barh(y, segment_deltas, color=colors, height=0.7)
    
    # Add value labels
    for bar, val in zip(bars, segment_deltas):
        x_pos = bar.get_width() + 0.002 if val >= 0 else bar.get_width() - 0.002
        ha = 'left' if val >= 0 else 'right'
        ax.text(x_pos, bar.get_y() + bar.get_height()/2,
               f'{val:+.3f}s', va='center', ha=ha, fontsize=8, color=COLORS['primary'])
    
    ax.axvline(0, color=COLORS['primary'], linewidth=1, linestyle='-')
    ax.set_yticks(y)
    ax.set_yticklabels(segment_names)
    ax.invert_yaxis()
    
    apply_style(ax, title='Segment Delta vs Baseline', xlabel='Delta (s)')
    
    plt.tight_layout()
    return fig


def plot_cumulative_time(
    setups: List[str],
    segment_times: List[List[float]]
) -> plt.Figure:
    """
    Line chart showing cumulative time progression.
    
    Args:
        setups: List of setup names
        segment_times: List of segment times for each setup
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    
    x = np.arange(1, len(segment_times[0]) + 1)
    
    for i, (name, times) in enumerate(zip(setups, segment_times)):
        cumulative = np.cumsum(times)
        ax.plot(x, cumulative, marker='o', markersize=4, linewidth=2,
               color=SETUP_COLORS[i % len(SETUP_COLORS)], label=name)
    
    ax.legend(loc='lower right', frameon=False)
    ax.set_xticks(x)
    
    apply_style(ax, title='Cumulative Lap Time Progression',
               xlabel='Segment', ylabel='Cumulative Time (s)')
    
    plt.tight_layout()
    return fig


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 12
) -> plt.Figure:
    """
    Horizontal bar chart of feature importance.
    
    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        top_n: Number of top features to show
        
    Returns:
        matplotlib Figure
    """
    df = importance_df.head(top_n).iloc[::-1]  # Reverse for proper display
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y = np.arange(len(df))
    ax.barh(y, df['importance'], color=COLORS['accent'], height=0.7)
    
    ax.set_yticks(y)
    ax.set_yticklabels(df['feature'])
    
    apply_style(ax, title='Feature Importance (XGBoost)',
               xlabel='Importance Score')
    
    plt.tight_layout()
    return fig


def plot_partial_dependence(
    param_values: np.ndarray,
    lap_times: np.ndarray,
    param_name: str,
    param_unit: str = ""
) -> plt.Figure:
    """
    Partial dependence plot for a single parameter.
    
    Args:
        param_values: Array of parameter values
        lap_times: Array of corresponding lap times
        param_name: Parameter name for label
        param_unit: Optional unit string
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(param_values, lap_times, linewidth=2, color=COLORS['accent'], marker='o', markersize=4)
    ax.fill_between(param_values, lap_times.min(), lap_times, alpha=0.1, color=COLORS['accent'])
    
    xlabel = f"{param_name}" if not param_unit else f"{param_name} ({param_unit})"
    apply_style(ax, title=f'Partial Dependence: {param_name}',
               xlabel=xlabel, ylabel='Predicted Lap Time (s)')
    
    plt.tight_layout()
    return fig


def plot_pdp_grid(
    pdp_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    param_units: Dict[str, str] = None
) -> plt.Figure:
    """
    Grid of partial dependence plots.
    
    Args:
        pdp_data: Dict of param_name -> (values, lap_times)
        param_units: Optional dict of param_name -> unit string
        
    Returns:
        matplotlib Figure
    """
    if param_units is None:
        param_units = {}
    
    n_params = len(pdp_data)
    n_cols = min(3, n_params)
    n_rows = (n_params + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = np.atleast_2d(axes).flatten()
    
    for i, (param, (values, lap_times)) in enumerate(pdp_data.items()):
        ax = axes[i]
        ax.plot(values, lap_times, linewidth=2, color=COLORS['accent'], marker='o', markersize=3)
        ax.fill_between(values, lap_times.min(), lap_times, alpha=0.1, color=COLORS['accent'])
        
        unit = param_units.get(param, '')
        xlabel = f"{param}" if not unit else f"{param} ({unit})"
        apply_style(ax, title=f'{param}', xlabel=xlabel, ylabel='Lap Time (s)')
    
    # Hide unused axes
    for i in range(len(pdp_data), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig


def plot_tradeoff(
    x_values: np.ndarray,
    y_values: np.ndarray,
    x_label: str,
    y_label: str = "Lap Time (s)",
    title: str = "Trade-Off Analysis"
) -> plt.Figure:
    """
    Trade-off curve plot.
    
    Args:
        x_values: X-axis values
        y_values: Y-axis values (usually lap time)
        x_label: X-axis label
        y_label: Y-axis label
        title: Plot title
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    
    ax.plot(x_values, y_values, linewidth=2, color=COLORS['accent'], marker='o', markersize=5)
    
    # Mark min lap time
    min_idx = np.argmin(y_values)
    ax.scatter([x_values[min_idx]], [y_values[min_idx]], s=100, c=COLORS['positive'],
              zorder=5, edgecolors='white', linewidth=2)
    ax.annotate(f'Optimal: {y_values[min_idx]:.2f}s',
               xy=(x_values[min_idx], y_values[min_idx]),
               xytext=(10, 10), textcoords='offset points',
               fontsize=9, color=COLORS['positive'])
    
    apply_style(ax, title=title, xlabel=x_label, ylabel=y_label)
    
    plt.tight_layout()
    return fig


def plot_validation_histogram(
    predictions: np.ndarray,
    actuals: np.ndarray
) -> plt.Figure:
    """
    Histogram of prediction errors.
    
    Args:
        predictions: Predicted values
        actuals: Actual values
        
    Returns:
        matplotlib Figure
    """
    errors = predictions - actuals
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.hist(errors, bins=30, color=COLORS['accent'], edgecolor='white', alpha=0.8)
    ax.axvline(0, color=COLORS['negative'], linewidth=2, linestyle='--')
    ax.axvline(np.mean(errors), color=COLORS['positive'], linewidth=2, label=f'Mean: {np.mean(errors):.3f}')
    
    ax.legend(loc='upper right', frameon=False)
    
    apply_style(ax, title='Prediction Error Distribution',
               xlabel='Error (s)', ylabel='Frequency')
    
    plt.tight_layout()
    return fig
