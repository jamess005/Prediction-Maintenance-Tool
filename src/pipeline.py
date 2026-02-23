import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

from features import engineer_features, get_feature_columns, get_target_column
from models import build_model, get_class_weight, tune_model
from evaluate import evaluate_model, plot_confusion_matrix

THRESHOLD = 0.35  # probability cut-off: lower → more failures flagged, higher recall

DATA_PATH  = '/home/james/ml-proj/predmain/data/ai4i2020.csv'
MODEL_PATH = '/home/james/ml-proj/predmain/outputs/models/xgb_model.pkl'
CM_PATH    = '/home/james/ml-proj/predmain/outputs/figures/confusion_matrix.png'


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop(columns=['UDI', 'Product ID', 'Type'], errors='ignore')
    return df


def run():
    print("=== LOADING DATA ===")
    df = load_data(DATA_PATH)
    print(f"Loaded {len(df)} rows")

    print("\n=== ENGINEERING FEATURES ===")
    df = engineer_features(df)
    print("Features created:", get_feature_columns())

    X = df[get_feature_columns()]
    y = df[get_target_column()]

    print("\n=== 70 / 15 / 15 SPLIT ===")
    # Step 1: hold out 30% as val + test combined
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    # Step 2: split the 30% equally → 15% val, 15% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    print("\n=== BUILDING MODEL ===")
    weight = get_class_weight(y_train)

    print("\n=== HYPERPARAMETER TUNING (Optuna, 75 trials) ===")
    tune_results = tune_model(X_train, y_train, scale_pos_weight=weight, n_trials=75)
    best_params = tune_results["best_params"]

    print("\n=== CROSS VALIDATION (train set, tuned params) ===")
    model_cv = build_model(scale_pos_weight=weight, **best_params)
    evaluate_model(model_cv, X_train, y_train)

    print("\n=== FINAL FIT (early stopping on val set) ===")
    # Merge: early-stopping overrides take priority over Optuna's n_estimators
    final_params = {**best_params, 'n_estimators': 3000, 'early_stopping_rounds': 75}
    model_final = build_model(scale_pos_weight=weight, **final_params)
    model_final.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    print(f"Early stopping: best iteration = {model_final.best_iteration} (of up to 3000 trees)")

    print(f"\n=== VALIDATION SET EVALUATION (threshold={THRESHOLD}) ===")
    val_metrics = plot_confusion_matrix(
        model_final, X_val, y_val, threshold=THRESHOLD, show=False
    )
    print(f"Val   MCC: {val_metrics['mcc']:.3f}  F1: {val_metrics['f1']:.3f}")

    print(f"\n=== TEST SET EVALUATION (threshold={THRESHOLD}) ===")
    plot_confusion_matrix(
        model_final, X_test, y_test,
        threshold=THRESHOLD, save_path=CM_PATH, show=False,
    )

    print("\n=== SAVING MODEL ===")
    joblib.dump(model_final, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == '__main__':
    run()