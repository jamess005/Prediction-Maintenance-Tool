import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

from features import engineer_features, get_feature_columns, get_target_column
from models import build_model, get_class_weight, tune_model, find_best_threshold
from evaluate import plot_confusion_matrix

DATA_PATH  = '/home/james/ml-proj/predmain/data/ai4i2020.csv'
MODEL_PATH = '/home/james/ml-proj/predmain/outputs/models/xgb_model.pkl'
CM_PATH    = '/home/james/ml-proj/predmain/outputs/figures/confusion_matrix.png'


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop(columns=['UDI', 'Product ID'], errors='ignore')
    return df


def run():
    # ── Data ────────────────────────────────────────────────────────────────
    df = load_data(DATA_PATH)
    df = engineer_features(df)

    X = df[get_feature_columns()]
    y = df[get_target_column()]

    # 70 / 15 / 15 split
    # val: threshold selection (~1497 rows, ~52 failure cases — more reliable
    #      than the previous 4-way split which gave only ~34 failure cases)
    # test: final honest evaluation (untouched until reporting)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    weight = get_class_weight(y_train)

    print(f"Data:    {len(df)} rows, {len(get_feature_columns())} features")
    print(f"Split:   train {len(X_train)} / val {len(X_val)} / test {len(X_test)}")
    print(f"Weight:  {weight:.2f}  ({int((y_train==0).sum())} no-fail vs {int((y_train==1).sum())} fail)")

    # ── Hyperparameter tuning ───────────────────────────────────────────────
    print("\nTuning hyperparameters (100 Optuna trials, 5-fold stratified CV)...")
    print("  Optimises MCC at 0.5 threshold — fair, deterministic comparison across trials.")
    print("  Deployment threshold is selected separately on val set.\n")
    tune_results = tune_model(X_train, y_train, scale_pos_weight=weight, n_trials=100)
    best_params = tune_results["best_params"]
    print(f"  Optuna search MCC: {tune_results['best_mcc']:.4f}")

    # ── Final fit ───────────────────────────────────────────────────────────
    # Use Optuna's best params directly — training takes ~1 min so early
    # stopping adds architectural complexity for no practical benefit.
    model = build_model(scale_pos_weight=weight, **best_params)
    model.fit(X_train, y_train)
    print(f"\nFinal model: {model.n_estimators} trees")

    # ── Find optimal threshold on val set ───────────────────────────────────
    # Recall floor is 0.90 — missing a failure (false negative) is far more
    # costly than a false alarm in predictive maintenance.
    threshold = find_best_threshold(model, X_val, y_val)
    print(f"Threshold: {threshold}  (best MCC on val set with ≥80% recall floor)")

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
    joblib.dump({'model': model, 'threshold': threshold}, MODEL_PATH)
    print(f"\nSaved model + threshold -> {MODEL_PATH}")


if __name__ == '__main__':
    run()