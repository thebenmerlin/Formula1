# F1 ML Design Trade-Space Exploration System

A physics-informed ML system for evaluating how high-level F1 vehicle design parameters affect lap time, energy usage, and thermal risk under 2026-style hybrid constraints at Spa-Francorchamps.

## Overview

This is NOT a driving simulator or real-time physics engine. It's a **design trade-space exploration tool** that:
- Uses physics-informed synthetic data generation
- Trains ML surrogate models for fast design evaluation
- Enables comparison of different vehicle configurations

## Design Parameters

| Parameter | Symbol | Range | Description |
|-----------|--------|-------|-------------|
| Mass | m | 740-800 kg | Total vehicle mass |
| Aero Load | C_L | 0.8-1.3 | Normalized downforce coefficient |
| Aero Drag | C_D | 0.85-1.2 | Normalized drag coefficient |
| Electric Fraction | α_elec | 0.35-0.60 | P_electric / P_total |
| Energy Deploy | E_deploy | 5-9 MJ | Max electric energy per lap |
| Cooling Factor | γ_cool | 0.7-1.3 | Cooling aggressiveness |

## Outputs

For each design vector, the system predicts:
- **Lap Time**: Total time at Spa (seconds)
- **Energy Used**: Electric energy consumed (MJ)
- **Thermal Risk**: Continuous indicator (0-1)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate synthetic training data
python -m src.data.data_generator

# Train ML models
python -m src.models.train

# Evaluate and analyze
python -m src.models.evaluate

# Launch Streamlit Dashboard
streamlit run src/app.py
```

## Streamlit Dashboard

The interactive design comparison dashboard provides:
- **Track Visualization**: 2D polyline of Spa-Francorchamps with animated design markers
- **Timing Grid**: Sector-by-sector times and lap totals
- **Engineering Plots**: Energy deployment, speed profiles, and time deltas
- **Design Selection**: Compare 2-4 vehicle configurations side-by-side

```bash
# Launch the dashboard
streamlit run src/app.py
```

## Project Structure

```
Formula1/
├── src/
│   ├── config.py           # Central configuration
│   ├── data/               # Track, physics, data generation
│   ├── models/             # ML training and evaluation
│   └── analysis/           # Segment prediction, comparison
├── data/                   # Generated datasets
├── models/                 # Saved model artifacts
└── outputs/                # Plots, Streamlit-ready outputs
```

## Track: Spa-Francorchamps

- Total length: ~7.004 km
- 20 segments (corners and straights)
- Includes elevation changes and varying curvatures

## ML Models

1. **Baseline**: Ridge-regularized Linear Regression
2. **Main**: Gradient Boosting Regressor
3. **Optional**: 2-layer MLP

All models are multi-output regressors predicting lap_time, energy_used, and thermal_risk simultaneously.
