import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

from pathlib import Path

from features import engineer_features, get_feature_columns, get_target_column
from models import (
    build_model, build_rf_model,
    get_class_weight,
    tune_model, tune_rf_model,
    find_best_threshold,
)
from evaluate import plot_confusion_matrix

ROOT           = Path(__file__).resolve().parent.parent
DATA_PATH      = ROOT / 'data' / 'ai4i2020.csv'
XGB_MODEL_PATH = ROOT / 'outputs' / 'models' / 'xgb_model.pkl'
RF_MODEL_PATH  = ROOT / 'outputs' / 'models' / 'rf_model.pkl'
CM_PATH        = ROOT / 'outputs' / 'figures' / 'confusion_matrix.png'


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop(columns=['UDI', 'Product ID'], errors='ignore')
    return df


def run(model_type: str = 'xgb') -> None:
    """
    model_type:
      'xgb' — XGBoost (default)
      'rf'  — Random Forest
    """
    model_type = model_type.lower()
    if model_type not in ('xgb', 'rf'):
        raise ValueError(f"model_type must be 'xgb' or 'rf', got '{model_type}'")

    model_path = XGB_MODEL_PATH if model_type == 'xgb' else RF_MODEL_PATH
    model_name = 'XGBoost' if model_type == 'xgb' else 'Random Forest'
    print(f"{'='*50}")
    print(f"  Training: {model_name}")
    print(f"{'='*50}\n")

    # ── Data ────────────────────────────────────────────────────────────────
    df = load_data(DATA_PATH)
    df = engineer_features(df)

    X = df[get_feature_columns()]
    y = df[get_target_column()]

    # 80 / 10 / 10 split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    n_nofail = int(np.sum(y_train == 0))
    n_fail   = int(np.sum(y_train == 1))

    print(f"Data:    {len(df)} rows, {len(get_feature_columns())} features")
    print(f"Split:   train {len(X_train)} / val {len(X_val)} / test {len(X_test)}")

    # ── Hyperparameter tuning ───────────────────────────────────────────────
    print(f"\nTuning hyperparameters (100 Optuna trials, 5-fold stratified CV)...\n")

    if model_type == 'xgb':
        weight = get_class_weight(y_train)
        print(f"Weight:  {weight:.2f}  ({n_nofail} no-fail vs {n_fail} fail)")
        tune_results = tune_model(X_train, y_train, scale_pos_weight=weight, n_trials=100)
        best_params = tune_results["best_params"]
        print(f"  Optuna search MCC: {tune_results['best_mcc']:.4f}")

        model = build_model(scale_pos_weight=weight, **best_params)
        model.fit(X_train, y_train)
        print(f"\nFinal model: {getattr(model, 'n_estimators', '?')} trees")
    else:
        print(f"Class:   {n_nofail} no-fail / {n_fail} fail  (using class_weight='balanced')")
        tune_results = tune_rf_model(X_train, y_train, n_trials=100)
        best_params = tune_results["best_params"]
        print(f"  Optuna search MCC: {tune_results['best_mcc']:.4f}")

        model = build_rf_model(class_weight="balanced", **best_params)
        model.fit(X_train, y_train)
        print(f"\nFinal model: {getattr(model, 'n_estimators', '?')} trees")

    # ── Find optimal threshold on val set ───────────────────────────────────
    threshold = find_best_threshold(model, X_val, y_val)
    print(f"Threshold: {threshold}  (best MCC on val set with >=80% recall floor)")

    # ── Evaluate ────────────────────────────────────────────────────────────
    val_m  = plot_confusion_matrix(model, X_val,  y_val,  threshold=threshold, show=False)
    test_m = plot_confusion_matrix(model, X_test, y_test, threshold=threshold,
                                   save_path=CM_PATH, show=False)

    mean_mcc  = (val_m['mcc']       + test_m['mcc'])       / 2
    mean_f1   = (val_m['f1']        + test_m['f1'])        / 2
    mean_rec  = (val_m['recall']    + test_m['recall'])    / 2
    mean_prec = (val_m['precision'] + test_m['precision']) / 2

    print(f"\n{'Set':<8} {'MCC':>6} {'F1':>6} {'Recall':>7} {'Precision':>10}")
    print(f"{'─'*40}")
    print(f"{'Val':<8} {val_m['mcc']:>6.3f} {val_m['f1']:>6.3f} {val_m['recall']:>7.3f} {val_m['precision']:>10.3f}")
    print(f"{'Test':<8} {test_m['mcc']:>6.3f} {test_m['f1']:>6.3f} {test_m['recall']:>7.3f} {test_m['precision']:>10.3f}")
    print(f"{'─'*40}")
    print(f"{'Mean':<8} {mean_mcc:>6.3f} {mean_f1:>6.3f} {mean_rec:>7.3f} {mean_prec:>10.3f}")

    # ── Save ────────────────────────────────────────────────────────────────
    joblib.dump({
        'model': model,
        'threshold': threshold,
        'model_type': model_type,
    }, model_path)
    print(f"\nSaved {model_name} model + threshold -> {model_path}")


if __name__ == '__main__':
    import sys
    choice = '1'
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        print("Select model to train:")
        print("  1 - XGBoost  (default)")
        print("  2 - Random Forest")
        choice = input("Choice [1]: ").strip() or '1'

    model_map = {'1': 'xgb', '2': 'rf', 'xgb': 'xgb', 'rf': 'rf'}
    selected = model_map.get(choice, 'xgb')
    run(model_type=selected)
