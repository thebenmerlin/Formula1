"""
Plots Module for F1 Dashboard

All visualization functions with:
- Dark professional theme
- Enterprise-grade presentation
- Consistent color palette
- High-contrast readability
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import List, Dict, Optional, Tuple

# Professional dark theme color palette
COLORS = {
    'primary': '#f8fafc',      # Light text
    'secondary': '#94a3b8',    # Muted text
    'accent': '#3b82f6',       # Blue accent
    'accent_secondary': '#8b5cf6',  # Purple accent
    'positive': '#10b981',     # Green (good/faster)
    'negative': '#ef4444',     # Red (bad/slower)
    'neutral': '#64748b',      # Gray neutral
    'background': '#1e293b',   # Dark background
    'card': '#0f172a',         # Darker card bg
    'grid': '#334155',         # Subtle grid
    'border': '#475569'        # Border color
}

# Setup comparison colors - vibrant for contrast
SETUP_COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6']


def apply_dark_style(ax, title: str = None, xlabel: str = None, ylabel: str = None):
    """Apply consistent dark styling to axis."""
    ax.set_facecolor(COLORS['background'])
    ax.figure.set_facecolor(COLORS['card'])
    
    # Spine styling
    for spine in ax.spines.values():
        spine.set_color(COLORS['border'])
        spine.set_linewidth(0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Tick styling
    ax.tick_params(colors=COLORS['secondary'], labelsize=9)
    
    # Grid styling
    ax.grid(True, alpha=0.15, color=COLORS['grid'], linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    if title:
        ax.set_title(title, fontsize=12, fontweight='600', color=COLORS['primary'], pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10, color=COLORS['secondary'], labelpad=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, color=COLORS['secondary'], labelpad=8)


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
        bars = ax.bar(x + offset, times, width, label=name, 
                     color=SETUP_COLORS[i % len(SETUP_COLORS)],
                     edgecolor='white', linewidth=0.5, alpha=0.9)
        
        # Add value labels
        for bar, val in zip(bars, times):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=9, 
                   color=COLORS['primary'], fontweight='500')
    
    ax.set_xticks(x)
    ax.set_xticklabels(['Sector 1', 'Sector 2', 'Sector 3'], color=COLORS['primary'])
    ax.legend(loc='upper right', frameon=False, facecolor=COLORS['background'], 
             labelcolor=COLORS['primary'])
    
    apply_dark_style(ax, title='Sector Time Comparison', ylabel='Time (s)')
    
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
    
    fig, ax = plt.subplots(figsize=(10, 9))
    
    y = np.arange(len(segment_deltas))
    colors = [COLORS['negative'] if d > 0.001 else COLORS['positive'] if d < -0.001 else COLORS['neutral']
              for d in segment_deltas]
    
    bars = ax.barh(y, segment_deltas, color=colors, height=0.7, 
                   edgecolor='white', linewidth=0.3, alpha=0.85)
    
    # Add value labels
    for bar, val in zip(bars, segment_deltas):
        x_pos = bar.get_width() + 0.003 if val >= 0 else bar.get_width() - 0.003
        ha = 'left' if val >= 0 else 'right'
        color = COLORS['negative'] if val > 0.001 else COLORS['positive'] if val < -0.001 else COLORS['neutral']
        ax.text(x_pos, bar.get_y() + bar.get_height()/2,
               f'{val:+.3f}s', va='center', ha=ha, fontsize=8, 
               color=color, fontweight='600')
    
    # Center line
    ax.axvline(0, color=COLORS['primary'], linewidth=1.5, linestyle='-', alpha=0.3)
    ax.set_yticks(y)
    ax.set_yticklabels(segment_names, fontsize=9)
    ax.invert_yaxis()
    
    # Add subtle background shading for faster/slower zones
    xlim = ax.get_xlim()
    ax.axvspan(xlim[0], 0, alpha=0.03, color=COLORS['positive'])
    ax.axvspan(0, xlim[1], alpha=0.03, color=COLORS['negative'])
    
    apply_dark_style(ax, title='Segment Delta vs Baseline', xlabel='Delta (s) — Negative = Faster')
    
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
        color = SETUP_COLORS[i % len(SETUP_COLORS)]
        ax.plot(x, cumulative, marker='o', markersize=5, linewidth=2.5,
               color=color, label=name, markeredgecolor='white', 
               markeredgewidth=0.5, alpha=0.9)
        
        # Fill under curve with gradient effect
        ax.fill_between(x, cumulative.min() - 2, cumulative, alpha=0.1, color=color)
    
    ax.legend(loc='lower right', frameon=True, facecolor=COLORS['card'], 
             edgecolor=COLORS['border'], labelcolor=COLORS['primary'])
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in x], fontsize=8)
    
    apply_dark_style(ax, title='Cumulative Lap Time Progression',
               xlabel='Segment Number', ylabel='Cumulative Time (s)')
    
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
    
    # Create gradient colors based on importance
    norm_importance = df['importance'] / df['importance'].max()
    colors = [plt.cm.Blues(0.3 + 0.6 * v) for v in norm_importance]
    
    bars = ax.barh(y, df['importance'], color=colors, height=0.7,
                   edgecolor='white', linewidth=0.3)
    
    # Add value labels
    for bar, val in zip(bars, df['importance']):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
               f'{val:.3f}', va='center', ha='left', fontsize=9, 
               color=COLORS['secondary'], fontweight='500')
    
    ax.set_yticks(y)
    ax.set_yticklabels(df['feature'], fontsize=10)
    ax.set_xlim(0, df['importance'].max() * 1.15)
    
    apply_dark_style(ax, title='Feature Importance (XGBoost)',
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
    
    ax.plot(param_values, lap_times, linewidth=2.5, color=COLORS['accent'], 
            marker='o', markersize=5, markeredgecolor='white', markeredgewidth=0.5)
    ax.fill_between(param_values, lap_times.min(), lap_times, alpha=0.15, color=COLORS['accent'])
    
    xlabel = f"{param_name}" if not param_unit else f"{param_name} ({param_unit})"
    apply_dark_style(ax, title=f'Partial Dependence: {param_name}',
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
    fig.set_facecolor(COLORS['card'])
    axes = np.atleast_2d(axes).flatten()
    
    colors_cycle = [COLORS['accent'], COLORS['positive'], COLORS['accent_secondary'], 
                    '#f59e0b', '#ec4899', '#06b6d4']
    
    for i, (param, (values, lap_times)) in enumerate(pdp_data.items()):
        ax = axes[i]
        color = colors_cycle[i % len(colors_cycle)]
        
        ax.plot(values, lap_times, linewidth=2.5, color=color, 
                marker='o', markersize=4, markeredgecolor='white', markeredgewidth=0.3)
        ax.fill_between(values, lap_times.min() - 0.5, lap_times, alpha=0.12, color=color)
        
        # Add min/max markers
        min_idx = np.argmin(lap_times)
        max_idx = np.argmax(lap_times)
        ax.scatter([values[min_idx]], [lap_times[min_idx]], s=60, c=COLORS['positive'],
                  zorder=5, edgecolors='white', linewidth=1.5)
        
        unit = param_units.get(param, '')
        xlabel = f"{param}" if not unit else f"{param} ({unit})"
        apply_dark_style(ax, title=f'{param}', xlabel=xlabel, ylabel='Lap Time (s)')
    
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
    
    # Main line with gradient effect
    ax.plot(x_values, y_values, linewidth=3, color=COLORS['accent'], 
            marker='o', markersize=6, markeredgecolor='white', markeredgewidth=0.5,
            alpha=0.9)
    
    # Fill under curve
    ax.fill_between(x_values, y_values.min() - 0.5, y_values, alpha=0.1, color=COLORS['accent'])
    
    # Mark min lap time with prominent marker
    min_idx = np.argmin(y_values)
    ax.scatter([x_values[min_idx]], [y_values[min_idx]], s=150, c=COLORS['positive'],
              zorder=5, edgecolors='white', linewidth=2, marker='*')
    
    # Add annotation for optimal point
    ax.annotate(f'Optimal\n{y_values[min_idx]:.3f}s',
               xy=(x_values[min_idx], y_values[min_idx]),
               xytext=(15, 15), textcoords='offset points',
               fontsize=10, color=COLORS['positive'], fontweight='600',
               bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['card'], 
                        edgecolor=COLORS['positive'], alpha=0.9),
               arrowprops=dict(arrowstyle='->', color=COLORS['positive'], lw=1.5))
    
    apply_dark_style(ax, title=title, xlabel=x_label, ylabel=y_label)
    
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
    
    ax.hist(errors, bins=30, color=COLORS['accent'], edgecolor='white', 
            alpha=0.8, linewidth=0.5)
    ax.axvline(0, color=COLORS['negative'], linewidth=2, linestyle='--', alpha=0.7)
    ax.axvline(np.mean(errors), color=COLORS['positive'], linewidth=2, 
               label=f'Mean: {np.mean(errors):.3f}')
    
    ax.legend(loc='upper right', frameon=True, facecolor=COLORS['card'],
             edgecolor=COLORS['border'], labelcolor=COLORS['primary'])
    
    apply_dark_style(ax, title='Prediction Error Distribution',
               xlabel='Error (s)', ylabel='Frequency')
    
    plt.tight_layout()
    return fig
