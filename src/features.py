import pandas as pd

# Failure-mode sub-type columns — sub-labels of the target, never model features
# (direct data leakage).
_FAILURE_MODE_COLS = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']

# Raw sensor columns renamed to bracket-free names (required by XGBoost,
# and cleaner for all downstream code).
_RAW_RENAME = {
    'Air temperature [K]':     'air_temp_K',
    'Process temperature [K]': 'proc_temp_K',
    'Rotational speed [rpm]':  'rot_speed_rpm',
    'Torque [Nm]':             'torque_Nm',
    'Tool wear [min]':         'tool_wear_min',
}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the raw AI4I 2020 dataset for modelling.

    Steps
    -----
    1. Drop rows where RNF=1 (random failures — no learnable pattern).
    2. Rename raw sensor columns to bracket-free names.
    3. Create three engineered features (power_kW, temp_delta_K, torque_wear).
    4. Encode product quality type as ordinal integer.
    5. Drop failure-mode sub-type columns (data leakage) and 'Type'.
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
    out['power_kW']     = (out['torque_Nm'] * out['rot_speed_rpm']) / 9550   # mechanical power
    out['temp_delta_K'] = out['proc_temp_K'] - out['air_temp_K']             # cooling effectiveness
    out['torque_wear']  = out['torque_Nm'] * out['tool_wear_min']            # cumulative stress

    # 4. Product type — ordinal encoding (L=0, M=1, H=2)
    out['product_type'] = out['Type'].map({'L': 0, 'M': 1, 'H': 2})

    # 5. Drop leakage columns and Type
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