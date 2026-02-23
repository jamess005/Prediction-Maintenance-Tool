import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

from features import engineer_features, get_feature_columns, get_target_column
from models import build_model, get_class_weight, tune_model
from evaluate import evaluate_model, plot_confusion_matrix

DATA_PATH  = '/home/james/ml-proj/predmain/data/ai4i2020.csv'
MODEL_PATH = '/home/james/ml-proj/predmain/outputs/models/xgb_model.pkl'
CM_PATH    = '/home/james/ml-proj/predmain/outputs/figures/confusion_matrix.png'


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Drop identifier columns that have no predictive value
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

    print("\n=== TRAIN/TEST SPLIT ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    print("\n=== BUILDING MODEL ===")
    weight = get_class_weight(y_train)

    print("\n=== HYPERPARAMETER TUNING (Optuna) ===")
    tune_results = tune_model(X_train, y_train, scale_pos_weight=weight, n_trials=50)
    model = build_model(scale_pos_weight=weight, **tune_results['best_params'])

    print("\n=== CROSS VALIDATION ===")
    evaluate_model(model, X_train, y_train)

    print("\n=== FINAL FIT AND TEST EVALUATION ===")
    model.fit(X_train, y_train)
    plot_confusion_matrix(model, X_test, y_test, save_path=CM_PATH)

    print("\n=== SAVING MODEL ===")
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == '__main__':
    run()