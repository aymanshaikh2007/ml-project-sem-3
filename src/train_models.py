import numpy as np
import pandas as pd
import time
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV
from src.evaluate import compute_regression_metrics, format_euro

def train_and_compare_models(data_dict):
    """
    Trains 4 baseline models, evaluates validation/test performance, performs hyperparameter tuning,
    and returns model performance comparisons and the selected best model.
    """
    X_train = data_dict['X_train_scaled']
    y_train_log = data_dict['y_train_log']

    X_val = data_dict['X_val_scaled']
    y_val_raw = data_dict['y_val_raw']
    y_val_log = data_dict['y_val_log']

    X_test = data_dict['X_test_scaled']
    y_test_raw = data_dict['y_test_raw']
    y_test_log = data_dict['y_test_log']

    # 1. Define Baseline Models
    models = {
        'Ridge Regression': Ridge(alpha=10.0, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42),
        'LightGBM': LGBMRegressor(n_estimators=150, learning_rate=0.08, num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
    }

    results = []
    trained_models = {}

    print("--- Training Baseline Models ---")
    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train, y_train_log)
        train_time = time.time() - t0

        # Predict log and transform back to raw currency via expm1
        val_pred_log = model.predict(X_val)
        val_pred_raw = np.expm1(val_pred_log)

        val_metrics = compute_regression_metrics(y_val_raw, val_pred_raw, y_val_log, val_pred_log)

        results.append({
            'Model': name,
            'Val Log R2': val_metrics['R2_Log'],
            'Val Raw R2': val_metrics['R2_Raw'],
            'Val MAE (EUR)': val_metrics['MAE_EUR'],
            'Val RMSE (EUR)': val_metrics['RMSE_EUR'],
            'Val MedAPE (%)': val_metrics['MedAPE_pct'],
            'Training Time (s)': round(train_time, 3)
        })
        trained_models[name] = model
        print(f"-> {name:20s} | Log R2: {val_metrics['R2_Log']:.4f} | MAE: {format_euro(val_metrics['MAE_EUR'])} | Time: {train_time:.2f}s")

    # 2. Hyperparameter Tuning for Best Tree Model (LightGBM)
    print("\n--- Performing Hyperparameter Tuning (LightGBM) ---")
    lgbm_base = LGBMRegressor(random_state=42, verbose=-1, n_jobs=-1)
    param_dist = {
        'n_estimators': [100, 150, 200, 250],
        'learning_rate': [0.03, 0.05, 0.08, 0.1],
        'num_leaves': [20, 31, 45, 60],
        'max_depth': [6, 8, 10, -1],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
    }

    search = RandomizedSearchCV(
        lgbm_base,
        param_distributions=param_dist,
        n_iter=15,
        cv=3,
        scoring='r2',
        random_state=42,
        n_jobs=-1
    )
    search.fit(X_train, y_train_log)
    best_tuned_lgbm = search.best_estimator_

    # Evaluate Tuned LightGBM on Validation Set
    tuned_val_pred_log = best_tuned_lgbm.predict(X_val)
    tuned_val_pred_raw = np.expm1(tuned_val_pred_log)
    tuned_val_metrics = compute_regression_metrics(y_val_raw, tuned_val_pred_raw, y_val_log, tuned_val_pred_log)

    results.append({
        'Model': 'Tuned LightGBM (Final)',
        'Val Log R2': tuned_val_metrics['R2_Log'],
        'Val Raw R2': tuned_val_metrics['R2_Raw'],
        'Val MAE (EUR)': tuned_val_metrics['MAE_EUR'],
        'Val RMSE (EUR)': tuned_val_metrics['RMSE_EUR'],
        'Val MedAPE (%)': tuned_val_metrics['MedAPE_pct'],
        'Training Time (s)': round(search.cv_results_['mean_fit_time'].sum(), 3)
    })
    trained_models['Tuned LightGBM (Final)'] = best_tuned_lgbm

    print(f"-> Tuned LightGBM (Final)   | Log R2: {tuned_val_metrics['R2_Log']:.4f} | MAE: {format_euro(tuned_val_metrics['MAE_EUR'])}")
    print(f"   Best Parameters: {search.best_params_}")

    # 3. Final Evaluation on Unseen Test Set
    test_pred_log = best_tuned_lgbm.predict(X_test)
    test_pred_raw = np.expm1(test_pred_log)
    test_metrics = compute_regression_metrics(y_test_raw, test_pred_raw, y_test_log, test_pred_log)

    print("\n--- Final Model Performance on Test Set ---")
    print(f"-> Test Log R2:   {test_metrics['R2_Log']:.4f}")
    print(f"-> Test Raw R2:   {test_metrics['R2_Raw']:.4f}")
    print(f"-> Test MAE:      {format_euro(test_metrics['MAE_EUR'])}")
    print(f"-> Test RMSE:     {format_euro(test_metrics['RMSE_EUR'])}")
    print(f"-> Test MedAPE:   {test_metrics['MedAPE_pct']:.2f}%")

    comparison_df = pd.DataFrame(results).sort_values(by='Val Log R2', ascending=False).reset_index(drop=True)

    return {
        'comparison_df': comparison_df,
        'trained_models': trained_models,
        'best_model': best_tuned_lgbm,
        'best_params': search.best_params_,
        'test_metrics': test_metrics,
        'test_pred_raw': test_pred_raw
    }
