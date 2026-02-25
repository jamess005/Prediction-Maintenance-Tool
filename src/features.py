import pandas as pd

# Failure-mode sub-type columns — sub-labels of the target, never model features
# (direct data leakage).
_FAILURE_MODE_COLS = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']

# Raw sensor columns that need bracket-free names for XGBoost compatibility.
_RAW_RENAME = {
    'Air temperature [K]':     'air_temp_K',
    'Process temperature [K]': 'proc_temp_K',
    'Rotational speed [rpm]':  'rot_speed_rpm',
    'Torque [Nm]':             'torque_Nm',
    'Tool wear [min]':         'tool_wear_min',
}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare features for the XGBoost model.

    Steps
    -----
    1. Drop rows where RNF=1 — random failures have no learnable pattern.
       NOTE: In production the model encounters RNF events without knowing
       they are random, so live recall may be slightly lower than metrics here.
    2. Rename raw sensor columns (XGBoost rejects square brackets).
    3. Add four engineered features.
    4. Encode product quality variant as ordinal integer.
    5. Drop failure-mode sub-type columns (data leakage).
    6. Drop 'Type' and any remaining ID columns.

    Feature notes
    -------------
    product_type (L=0, M=1, H=2) has low individual SHAP importance (~0.2)
    but is retained — removing it reduced Optuna search MCC from 0.808 to
    0.798, suggesting it contributes marginal signal that aids hyperparameter
    search even if it rarely drives individual predictions. The ordinal
    encoding is acceptable since XGBoost treats it as a split threshold
    rather than a true numeric value.

    wear_per_torque and speed_torque_ratio were tested to capture the high-RPM
    low-torque TWF failure sub-type (~9 persistent misses). Both were removed —
    correlation with existing features (0.82–0.90) added noise and forced the
    threshold down to 0.24, hurting precision without improving recall on the
    genuinely uncatchable cases (proba 0.001–0.031, scoring below any threshold).

    high_wear_flag and wear_torque_zone (binary TWF domain-knowledge flags)
    were also tested — they hard-encode the known failure condition
    (tool wear > 190 min, torque < 35 Nm).  They had no measurable impact on
    XGBoost MCC (0.878 vs 0.880 baseline) because the model already discovers
    the same thresholds via tree splits.  Removed to keep the feature set lean.
    """
    out = df.copy()

    # 1. Remove random-failure rows
    if 'RNF' in out.columns:
        n_before = len(out)
        out = out[out['RNF'] != 1].reset_index(drop=True)
        print(f"Dropped {n_before - len(out)} RNF rows ({n_before} → {len(out)})")

    # 2. Rename raw sensor columns
    out = out.rename(columns=_RAW_RENAME)

    # 3. Engineered features
    #    power_kW:     torque × RPM / 9550  (mechanical power in kilowatts)
    #    temp_delta_K: proc_temp − air_temp  (cooling effectiveness)
    #    torque_wear:  torque × tool_wear    (cumulative mechanical stress)
    out['power_kW']     = (out['torque_Nm'] * out['rot_speed_rpm']) / 9550
    out['temp_delta_K'] = out['proc_temp_K'] - out['air_temp_K']
    out['torque_wear']  = out['torque_Nm'] * out['tool_wear_min']

    # 4. Product type — ordinal encoding (L=0, M=1, H=2)
    out['product_type'] = out['Type'].map({'L': 0, 'M': 1, 'H': 2})

    # 5 & 6. Drop leakage columns and Type
    out = out.drop(columns=_FAILURE_MODE_COLS + ['Type'], errors='ignore')

    return out


def get_feature_columns() -> list:
    """
    Definitive ordered list of feature columns produced by engineer_features.
    9 features: 5 raw sensors + 4 engineered.
    """
    return [
        # Raw sensors
        'air_temp_K',
        'proc_temp_K',
        'rot_speed_rpm',
        'torque_Nm',
        'tool_wear_min',
        # Engineered
        'power_kW',
        'temp_delta_K',
        'torque_wear',
        'product_type',
    ]


def get_target_column() -> str:
    return 'Machine failure'