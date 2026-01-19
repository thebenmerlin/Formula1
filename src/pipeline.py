#!/usr/bin/env python
"""
Formula1 Track Time Prediction - Main Pipeline

Complete execution pipeline:
1. Generate synthetic data
2. Engineer features
3. Train baseline and advanced models
4. Evaluate model performance
5. Run sensitivity analysis
6. Generate final report

Usage:
    python -m src.pipeline --all           # Run complete pipeline
    python -m src.pipeline --generate      # Generate data only
    python -m src.pipeline --train         # Train models only
    python -m src.pipeline --evaluate      # Evaluate only
    python -m src.pipeline --analyze       # Sensitivity analysis only
"""

import argparse
from pathlib import Path
from datetime import datetime


def run_data_generation():
    """Step 1: Generate synthetic dataset."""
    print("\n" + "=" * 70)
    print("STEP 1: DATA GENERATION")
    print("=" * 70)
    
    from src.data.generator import SpaTrackGenerator
    
    generator = SpaTrackGenerator()
    df = generator.generate_dataset(
        n_samples=100000,
        noise_level=0.02,
        random_seed=42
    )
    
    # Save
    output_path = Path("data/synthetic/lap_times_100k.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    
    # Also save sample CSV
    df.head(1000).to_csv(output_path.with_suffix('.csv'), index=False)
    
    print(f"\n✓ Data saved to {output_path}")
    print(f"  Samples: {len(df):,}")
    print(f"  Lap time range: [{df['lap_time'].min():.2f}, {df['lap_time'].max():.2f}] s")
    
    return df


def run_training():
    """Step 2-3: Feature engineering and model training."""
    print("\n" + "=" * 70)
    print("STEP 2-3: FEATURE ENGINEERING & MODEL TRAINING")
    print("=" * 70)
    
    import pandas as pd
    from src.features.engineering import FeatureEngineer
    from src.models.train import LapTimePredictor
    
    # Load data
    data_path = Path("data/synthetic/lap_times_100k.parquet")
    if not data_path.exists():
        print("Data not found. Running data generation first...")
        run_data_generation()
    
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df):,} samples")
    
    # Feature engineering
    print("\n[Feature Engineering]")
    engineer = FeatureEngineer(include_interactions=True)
    base_cols = ['mass', 'c_l', 'c_d', 'alpha_elec', 'e_deploy', 'gamma_cool']
    df_features = engineer.fit_transform(df[base_cols])
    feature_cols = list(df_features.columns)
    print(f"Total features: {len(feature_cols)}")
    
    # Prepare full dataset
    df_full = pd.concat([
        df_features,
        df[['lap_time', 'sector_1', 'sector_2', 'sector_3']]
    ], axis=1)
    
    # Train models
    predictor = LapTimePredictor(target_type='lap')
    X, y = predictor.prepare_data(df_full, feature_cols)
    
    # Baselines
    predictor.train_baseline(X, y)
    
    # Advanced models
    predictor.train_xgboost(X, y, n_trials=30)
    predictor.train_lightgbm(X, y, n_trials=30)
    predictor.train_random_forest(X, y, n_trials=20)
    
    # Select best
    best = predictor.select_best_model()
    
    # Save
    save_path = Path("models/final/best_lap_predictor.joblib")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    predictor.save_model(best, str(save_path))
    
    print(f"\n✓ Best model saved: {best.name} (CV RMSE: {best.val_rmse:.4f})")
    
    return predictor


def run_evaluation():
    """Step 4: Model evaluation."""
    print("\n" + "=" * 70)
    print("STEP 4: MODEL EVALUATION")
    print("=" * 70)
    
    import pandas as pd
    import joblib
    from src.features.engineering import FeatureEngineer
    from src.evaluation.evaluate import ModelEvaluator
    
    # Load
    data_path = Path("data/synthetic/lap_times_100k.parquet")
    model_path = Path("models/final/best_lap_predictor.joblib")
    
    if not model_path.exists():
        print("Model not found. Running training first...")
        run_training()
    
    df = pd.read_parquet(data_path)
    model_data = joblib.load(model_path)
    model = model_data['model']
    model_name = model_data['name']
    
    # Features
    engineer = FeatureEngineer(include_interactions=True)
    base_cols = ['mass', 'c_l', 'c_d', 'alpha_elec', 'e_deploy', 'gamma_cool']
    df_features = engineer.fit_transform(df[base_cols])
    feature_cols = list(df_features.columns)
    
    X = df_features.values
    y = df['lap_time'].values
    
    # Evaluate
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_model(
        model, X, y,
        feature_names=feature_cols,
        model_name=model_name
    )
    
    # Plots
    evaluator.plot_error_distribution(y, results['predictions'], model_name)
    evaluator.plot_residuals(X, y, results['predictions'], feature_cols, model_name)
    
    # Report
    evaluator.generate_report(model_name)
    
    print(f"\n✓ Evaluation complete for {model_name}")
    
    return results


def run_sensitivity_analysis():
    """Step 5: Sensitivity analysis."""
    print("\n" + "=" * 70)
    print("STEP 5: SENSITIVITY ANALYSIS")
    print("=" * 70)
    
    import pandas as pd
    import joblib
    from src.features.engineering import FeatureEngineer
    from src.analysis.sensitivity import SensitivityAnalyzer
    
    # Load
    data_path = Path("data/synthetic/lap_times_100k.parquet")
    model_path = Path("models/final/best_lap_predictor.joblib")
    
    if not model_path.exists():
        print("Model not found. Running training first...")
        run_training()
    
    df = pd.read_parquet(data_path)
    model_data = joblib.load(model_path)
    model = model_data['model']
    model_name = model_data['name']
    
    # Features
    engineer = FeatureEngineer(include_interactions=True)
    base_cols = ['mass', 'c_l', 'c_d', 'alpha_elec', 'e_deploy', 'gamma_cool']
    df_features = engineer.fit_transform(df[base_cols])
    feature_names = list(df_features.columns)
    X = df_features.values
    
    # Analysis
    analyzer = SensitivityAnalyzer()
    
    # Feature importance
    importance = analyzer.compute_feature_importance(model, X, feature_names)
    analyzer.plot_feature_importance(importance, model_name=model_name)
    
    # Monotonicity
    monotonicity = analyzer.check_monotonicity(model, X, feature_names)
    
    # PDP
    analyzer.plot_partial_dependence(model, X, feature_names, model_name=model_name)
    
    # OOD
    param_bounds = {
        'mass': (700, 850),
        'c_l': (0.8, 1.5),
        'c_d': (0.7, 1.3),
        'alpha_elec': (0.0, 0.4),
        'e_deploy': (2.0, 4.0),
        'gamma_cool': (0.8, 1.2)
    }
    ood = analyzer.test_out_of_distribution(model, feature_names, param_bounds)
    
    # Report
    analyzer.generate_sensitivity_report(importance, monotonicity, ood, model_name)
    
    print(f"\n✓ Sensitivity analysis complete for {model_name}")


def run_full_pipeline():
    """Run complete pipeline."""
    print("\n" + "#" * 70)
    print("#  FORMULA1 TRACK TIME PREDICTION PIPELINE")
    print("#  Spa-Francorchamps | Analytics-First ML")
    print("#" * 70)
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    run_data_generation()
    run_training()
    run_evaluation()
    run_sensitivity_analysis()
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nOutputs:")
    print("  - data/synthetic/lap_times_100k.parquet")
    print("  - models/final/best_lap_predictor.joblib")
    print("  - outputs/figures/*.png")
    print("  - outputs/reports/*.md")


def main():
    parser = argparse.ArgumentParser(
        description="Formula1 Track Time Prediction Pipeline"
    )
    parser.add_argument('--all', action='store_true', help='Run complete pipeline')
    parser.add_argument('--generate', action='store_true', help='Generate data only')
    parser.add_argument('--train', action='store_true', help='Train models only')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
    parser.add_argument('--analyze', action='store_true', help='Sensitivity analysis only')
    
    args = parser.parse_args()
    
    if args.all or not any([args.generate, args.train, args.evaluate, args.analyze]):
        run_full_pipeline()
    else:
        if args.generate:
            run_data_generation()
        if args.train:
            run_training()
        if args.evaluate:
            run_evaluation()
        if args.analyze:
            run_sensitivity_analysis()


if __name__ == "__main__":
    main()
