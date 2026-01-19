# Formula1 — High-Accuracy Track Time Prediction

**Analytics-First Machine Learning Project**

## Objective

Predict segment times, sector times, and total lap time at Spa-Francorchamps for an F1-style hybrid car with maximum achievable accuracy using physics-informed synthetic data and interpretable machine learning.

---

## Design Parameter Vector (Fixed)

Each vehicle setup is defined by exactly **6 parameters**:

| Parameter | Symbol | Description | Range |
|-----------|--------|-------------|-------|
| Total Mass | `m` | Vehicle mass in kg | 700–850 kg |
| Aero Load Coefficient | `C_L` | Normalized downforce coefficient | 0.8–1.5 |
| Aero Drag Coefficient | `C_D` | Normalized drag coefficient | 0.7–1.3 |
| Electric Power Fraction | `α_elec` | Fraction of power from electric (0–1) | 0.0–0.4 |
| Max Deployable Energy | `E_deploy` | Max electric energy per lap (MJ) | 2.0–4.0 MJ |
| Cooling Aggressiveness | `γ_cool` | Cooling factor affecting power availability | 0.8–1.2 |

---

## Output Targets

The model predicts:

```
y = [
    segment_times[1..N],   # N ≈ 20 Spa segments
    sector_times[1..3],    # 3 sectors
    lap_time               # Total lap time
]
```

---

## Project Structure

```
Formula1/
├── src/
│   ├── data/           # Synthetic data generation
│   ├── features/       # Feature engineering
│   ├── models/         # Model training & tuning
│   ├── evaluation/     # Evaluation & metrics
│   └── analysis/       # Sensitivity & robustness analysis
├── data/
│   ├── raw/            # Raw track/physics data
│   ├── processed/      # Preprocessed datasets
│   └── synthetic/      # Generated training data
├── models/
│   ├── checkpoints/    # Training checkpoints
│   └── final/          # Production-ready models
├── outputs/
│   ├── figures/        # Plots and visualizations
│   └── reports/        # Evaluation reports
├── notebooks/          # Exploratory analysis
├── configs/            # Configuration files
└── requirements.txt    # Dependencies
```

---

## Execution Order

1. **Finalize physics-informed data generator**
2. **Generate large, diverse datasets** (≥100,000 samples)
3. **Engineer derived features** (power-to-weight, aero efficiency, etc.)
4. **Train baseline models** (Linear, Ridge, Lasso)
5. **Train advanced models** (XGBoost, LightGBM, Random Forest)
6. **Deep evaluation & sensitivity analysis**
7. **Freeze best-performing, explainable model**

---

## Key Principles

- **Accuracy over speed**: Prioritize prediction quality
- **Physics-informed**: All assumptions grounded in engineering logic
- **Explainable**: No black-box acceptance without reasoning
- **Robust**: Stable across retraining and edge cases

---

## Quick Start

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Generate synthetic data
python -m src.data.generator

# Train models
python -m src.models.train

# Evaluate
python -m src.evaluation.evaluate
```

---

## License

MIT License
