from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Tuple

from pipeline import create_preprocessing_pipeline, create_full_pipeline
from config.core import fetch_config_from_yaml
from preprocessing.data_manager import load_dataset, fill_with_predictions

from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score
import xgboost as xgb
import optuna

N_TRIALS = 100

def temporal_train_test_split(df: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
   """
   Split data based on window_start timestamps
   """
   df = df.sort_values('window_start')
   split_date = df['window_start'].iloc[int(len(df) * train_ratio)]
   
   train = df[df['window_start'] < split_date]
   test = df[df['window_start'] >= split_date]
   
   return train, test

def objective_xgb(trial, X, y, feature_columns: Dict) -> float:
    params = {
        "booster": "gbtree",
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1.0, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 1, 16),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
    }

    preprocessor = create_preprocessing_pipeline(feature_columns)
    clf = xgb.XGBClassifier(**params, use_label_encoder=False, eval_metric="logloss", n_jobs=-1)
    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])

    unique_dates = sorted(X['window_start'].unique())
    n_splits = 3
    scores = []
    
    for i in range(1, n_splits):
        split_point = int(len(unique_dates) * (i/n_splits))
        if split_point >= len(unique_dates):
            continue
            
        split_date = unique_dates[split_point]
        train_mask = X['window_start'] < split_date
        val_mask = (X['window_start'] >= split_date) & (X['window_start'] < unique_dates[-1])
        
        if train_mask.sum() == 0 or val_mask.sum() == 0:
            continue
            
        pipeline.fit(X[train_mask], y[train_mask])
        y_pred = pipeline.predict(X[val_mask])
        scores.append(f1_score(y[val_mask], y_pred))
    
    return np.mean(scores) if scores else 0.0

def main():
    data = load_dataset()
    config = fetch_config_from_yaml()

    data = fill_with_predictions(
            data,
            use_predicted_cwe=config['use_predictions']['cwe'],
            use_predicted_cvss=config['use_predictions']['cvss']
        )

    unique_dates = sorted(data['window_start'].unique())
    split_idx = int(len(unique_dates) * 0.8)
    if split_idx >= len(unique_dates):
        split_idx = len(unique_dates) - 1
    split_date = unique_dates[split_idx]

    train_mask = data['window_start'] < split_date
    test_mask = data['window_start'] >= split_date
    
    X_train = data[train_mask]
    X_test = data[test_mask]
    y_train = data[train_mask]["target"]
    y_test = data[test_mask]["target"]

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective_xgb(trial, X_train, y_train, config["feature_columns"]),
        n_trials=N_TRIALS
    )

    print("\nBest hyperparameters:", study.best_params)

    clf = xgb.XGBClassifier(
        **study.best_params,
        use_label_encoder=False,
        eval_metric="logloss",
        n_jobs=-1
    )
    
    clf = create_full_pipeline(clf, config["feature_columns"])
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()