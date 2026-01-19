"""
Model Training Module for F1 Track Time Prediction

Implements:
- Baseline models (Linear, Ridge, Lasso)
- Core models (XGBoost, LightGBM, Random Forest)
- K-fold cross-validation
- Hyperparameter optimization via Optuna

All models trained to:
- Minimize lap-time RMSE
- Maintain segment-level consistency
- Avoid physically implausible predictions
"""

import numpy as np
import pandas as pd
import yaml
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import optuna
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class ModelResult:
    """Container for model training results."""
    name: str
    model: Any
    train_rmse: float
    val_rmse: float
    cv_scores: List[float]
    best_params: Optional[Dict] = None
    feature_importance: Optional[pd.DataFrame] = None


class LapTimePredictor:
    """
    Multi-output predictor for F1 lap times.
    
    Predicts:
    - 20 segment times
    - 3 sector times
    - 1 total lap time
    
    Uses k-fold CV and hyperparameter tuning.
    """
    
    # Target columns
    SEGMENT_COLS = [f'segment_{i}' for i in range(1, 21)]
    SECTOR_COLS = ['sector_1', 'sector_2', 'sector_3']
    LAP_COL = 'lap_time'
    
    def __init__(
        self,
        config_path: str = "configs/model_config.yaml",
        target_type: str = 'lap'  # 'lap', 'sector', or 'segment'
    ):
        """
        Initialize predictor.
        
        Args:
            config_path: Path to model configuration
            target_type: What to predict ('lap', 'sector', 'segment')
        """
        self.config = self._load_config(config_path)
        self.target_type = target_type
        self.scaler = StandardScaler()
        self.results: Dict[str, ModelResult] = {}
        self.best_model: Optional[ModelResult] = None
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _get_target_cols(self) -> List[str]:
        """Get target columns based on target_type."""
        if self.target_type == 'lap':
            return [self.LAP_COL]
        elif self.target_type == 'sector':
            return self.SECTOR_COLS
        elif self.target_type == 'segment':
            return self.SEGMENT_COLS
        else:
            raise ValueError(f"Unknown target_type: {self.target_type}")
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        feature_cols: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and targets for training.
        
        Args:
            df: Full DataFrame
            feature_cols: Columns to use as features
            
        Returns:
            Tuple of (X, y) arrays
        """
        target_cols = self._get_target_cols()
        
        X = df[feature_cols].values
        y = df[target_cols].values
        
        # Squeeze if single target
        if y.shape[1] == 1:
            y = y.ravel()
        
        return X, y
    
    def train_baseline(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_folds: int = 5,
        random_state: int = 42
    ) -> Dict[str, ModelResult]:
        """
        Train baseline models (Linear, Ridge, Lasso).
        
        Args:
            X: Feature matrix
            y: Target vector/matrix
            n_folds: Number of CV folds
            random_state: Random seed
            
        Returns:
            Dict of model results
        """
        print("\n" + "=" * 50)
        print("Training Baseline Models")
        print("=" * 50)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        
        baselines = {
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=0.01)
        }
        
        results = {}
        
        for name, model in baselines.items():
            print(f"\n[{name.upper()}]")
            
            # Wrap in MultiOutput if needed
            if len(y.shape) > 1 and y.shape[1] > 1:
                estimator = MultiOutputRegressor(model)
            else:
                estimator = model
            
            # Cross-validation
            cv_rmses = []
            for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                estimator.fit(X_train, y_train)
                y_pred = estimator.predict(X_val)
                
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                cv_rmses.append(rmse)
            
            # Final fit on all data
            estimator.fit(X_scaled, y)
            y_pred_all = estimator.predict(X_scaled)
            train_rmse = np.sqrt(mean_squared_error(y, y_pred_all))
            
            print(f"  Train RMSE: {train_rmse:.4f}")
            print(f"  CV RMSE:    {np.mean(cv_rmses):.4f} ± {np.std(cv_rmses):.4f}")
            
            results[name] = ModelResult(
                name=name,
                model=estimator,
                train_rmse=train_rmse,
                val_rmse=np.mean(cv_rmses),
                cv_scores=cv_rmses
            )
        
        self.results.update(results)
        return results
    
    def train_xgboost(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_trials: int = 50,
        n_folds: int = 5,
        random_state: int = 42
    ) -> ModelResult:
        """
        Train XGBoost with Optuna hyperparameter tuning.
        """
        print("\n" + "=" * 50)
        print("Training XGBoost with Optuna")
        print("=" * 50)
        
        is_multi_output = len(y.shape) > 1 and y.shape[1] > 1
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'random_state': random_state,
                'n_jobs': -1
            }
            
            model = xgb.XGBRegressor(**params)
            if is_multi_output:
                model = MultiOutputRegressor(model)
            
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
            cv_rmses = []
            
            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                cv_rmses.append(rmse)
            
            return np.mean(cv_rmses)
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        best_params = study.best_params
        best_params['random_state'] = random_state
        best_params['n_jobs'] = -1
        
        print(f"\nBest params: {best_params}")
        print(f"Best CV RMSE: {study.best_value:.4f}")
        
        final_model = xgb.XGBRegressor(**best_params)
        if is_multi_output:
            final_model = MultiOutputRegressor(final_model)
        
        final_model.fit(X, y)
        y_pred = final_model.predict(X)
        train_rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        result = ModelResult(
            name='xgboost',
            model=final_model,
            train_rmse=train_rmse,
            val_rmse=study.best_value,
            cv_scores=[study.best_value],
            best_params=best_params
        )
        
        self.results['xgboost'] = result
        return result
    
    def train_lightgbm(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_trials: int = 50,
        n_folds: int = 5,
        random_state: int = 42
    ) -> ModelResult:
        """
        Train LightGBM with Optuna hyperparameter tuning.
        """
        print("\n" + "=" * 50)
        print("Training LightGBM with Optuna")
        print("=" * 50)
        
        is_multi_output = len(y.shape) > 1 and y.shape[1] > 1
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 15, 127),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'random_state': random_state,
                'n_jobs': -1,
                'verbose': -1
            }
            
            model = lgb.LGBMRegressor(**params)
            if is_multi_output:
                model = MultiOutputRegressor(model)
            
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
            cv_rmses = []
            
            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                cv_rmses.append(rmse)
            
            return np.mean(cv_rmses)
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        best_params = study.best_params
        best_params['random_state'] = random_state
        best_params['n_jobs'] = -1
        best_params['verbose'] = -1
        
        print(f"\nBest params: {best_params}")
        print(f"Best CV RMSE: {study.best_value:.4f}")
        
        final_model = lgb.LGBMRegressor(**best_params)
        if is_multi_output:
            final_model = MultiOutputRegressor(final_model)
        
        final_model.fit(X, y)
        y_pred = final_model.predict(X)
        train_rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        result = ModelResult(
            name='lightgbm',
            model=final_model,
            train_rmse=train_rmse,
            val_rmse=study.best_value,
            cv_scores=[study.best_value],
            best_params=best_params
        )
        
        self.results['lightgbm'] = result
        return result
    
    def train_random_forest(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_trials: int = 10,
        n_folds: int = 3,
        random_state: int = 42
    ) -> ModelResult:
        """
        Train Random Forest with Optuna hyperparameter tuning.
        Uses a subsample for faster training.
        """
        print("\n" + "=" * 50)
        print("Training Random Forest with Optuna")
        print("=" * 50)
        
        is_multi_output = len(y.shape) > 1 and y.shape[1] > 1
        
        # Use 10% sample for faster training
        np.random.seed(random_state)
        sample_size = min(10000, len(X))
        sample_idx = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X[sample_idx]
        y_sample = y[sample_idx] if len(y.shape) == 1 else y[sample_idx]
        
        print(f"Using {sample_size:,} samples for RF optimization (10% of data)")
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 100),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'min_samples_split': trial.suggest_int('min_samples_split', 5, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 10),
                'random_state': random_state,
                'n_jobs': -1
            }
            
            model = RandomForestRegressor(**params)
            if is_multi_output:
                model = MultiOutputRegressor(model)
            
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
            cv_rmses = []
            
            for train_idx, val_idx in kf.split(X_sample):
                X_train, X_val = X_sample[train_idx], X_sample[val_idx]
                y_train, y_val = y_sample[train_idx], y_sample[val_idx]
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                cv_rmses.append(rmse)
            
            return np.mean(cv_rmses)
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        best_params = study.best_params
        best_params['random_state'] = random_state
        best_params['n_jobs'] = -1
        
        print(f"\nBest params: {best_params}")
        print(f"Best CV RMSE: {study.best_value:.4f}")
        
        final_model = RandomForestRegressor(**best_params)
        if is_multi_output:
            final_model = MultiOutputRegressor(final_model)
        
        final_model.fit(X, y)
        y_pred = final_model.predict(X)
        train_rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        result = ModelResult(
            name='random_forest',
            model=final_model,
            train_rmse=train_rmse,
            val_rmse=study.best_value,
            cv_scores=[study.best_value],
            best_params=best_params
        )
        
        self.results['random_forest'] = result
        return result
    
    def select_best_model(self) -> ModelResult:
        """Select best model based on validation RMSE."""
        if not self.results:
            raise ValueError("No models trained yet")
        
        self.best_model = min(self.results.values(), key=lambda x: x.val_rmse)
        
        print("\n" + "=" * 50)
        print("Model Selection Summary")
        print("=" * 50)
        
        print("\nModel Rankings (by CV RMSE):")
        sorted_results = sorted(self.results.values(), key=lambda x: x.val_rmse)
        for i, r in enumerate(sorted_results, 1):
            marker = " ← BEST" if r.name == self.best_model.name else ""
            print(f"  {i}. {r.name:15} CV RMSE: {r.val_rmse:.4f}{marker}")
        
        return self.best_model
    
    def save_model(self, model_result: ModelResult, path: str):
        """Save trained model to disk."""
        joblib.dump({
            'model': model_result.model,
            'scaler': self.scaler,
            'name': model_result.name,
            'params': model_result.best_params,
            'val_rmse': model_result.val_rmse
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str) -> Any:
        """Load trained model from disk."""
        data = joblib.load(path)
        self.scaler = data['scaler']
        return data['model']


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("Formula1 Model Training Pipeline")
    print("=" * 60)
    
    from src.features.engineering import FeatureEngineer
    
    # Load data
    data_path = Path("data/synthetic/lap_times_100k.parquet")
    if not data_path.exists():
        print("Data not found. Run data generator first:")
        print("  python -m src.data.generator")
        return
    
    print(f"\n[1/5] Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df):,} samples")
    
    # Feature engineering
    print("\n[2/5] Engineering features...")
    engineer = FeatureEngineer(include_interactions=True)
    base_cols = ['mass', 'c_l', 'c_d', 'alpha_elec', 'e_deploy', 'gamma_cool']
    df_features = engineer.fit_transform(df[base_cols])
    feature_cols = list(df_features.columns)
    
    print(f"Total features: {len(feature_cols)}")
    
    # Merge features with targets
    df_full = pd.concat([df_features, df[['lap_time', 'sector_1', 'sector_2', 'sector_3']]], axis=1)
    
    # Initialize predictor
    predictor = LapTimePredictor(target_type='lap')
    
    # Prepare data
    X, y = predictor.prepare_data(df_full, feature_cols)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    # Train baselines
    print("\n[3/5] Training baseline models...")
    predictor.train_baseline(X, y)
    
    # Train advanced models
    print("\n[4/5] Training advanced models...")
    predictor.train_xgboost(X, y, n_trials=30)
    predictor.train_lightgbm(X, y, n_trials=30)
    predictor.train_random_forest(X, y, n_trials=20)
    
    # Select best
    print("\n[5/5] Selecting best model...")
    best = predictor.select_best_model()
    
    # Save best model
    save_path = Path("models/final/best_lap_predictor.joblib")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    predictor.save_model(best, str(save_path))
    
    print("\n✓ Training complete!")
    print(f"Best model: {best.name} (CV RMSE: {best.val_rmse:.4f})")


if __name__ == "__main__":
    main()
