import pandas as pd

# Failure-mode sub-type columns — these are sub-labels of the target and must
# never appear as model features (direct data leakage).
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
    1. Drop rows where RNF=1 — random failures have no learnable pattern and
       add noise that degrades the model.
    2. Rename raw sensor columns (XGBoost rejects square brackets).
    3. Add three engineered features that combine/transform the raw signals.
    4. Encode the product quality variant as an ordinal integer.
    5. Drop failure-mode sub-type columns (TWF/HDF/PWF/OSF/RNF) — these are
       derived from the target label and would be pure data leakage.
    6. Drop 'Type' (already encoded as product_type) and any remaining ID cols.

    Output feature columns (see get_feature_columns for the definitive list):
        Raw sensors (renamed)
        ----------------------
        air_temp_K      : Air temperature [K]
        proc_temp_K     : Process temperature [K]
        rot_speed_rpm   : Rotational speed [rpm]
        torque_Nm       : Torque [Nm]
        tool_wear_min   : Tool wear [min]

        Engineered
        ----------
        power_W         : torque × RPM / 9550  (actual mechanical load)
        temp_delta_K    : proc_temp − air_temp  (cooling effectiveness)
        torque_wear     : torque × tool_wear   (cumulative stress on worn tool)
        product_type    : L=0, M=1, H=2  (quality variant, ordinal)
    """
    out = df.copy()

    # 1. Remove random-failure rows — no signal to learn, only noise
    if 'RNF' in out.columns:
        n_before = len(out)
        out = out[out['RNF'] != 1].reset_index(drop=True)
        print(f"Dropped {n_before - len(out)} RNF rows ({n_before} → {len(out)})")

    # 2. Rename raw sensor columns to bracket-free names
    out = out.rename(columns=_RAW_RENAME)

    # 3. Engineered features — built from the renamed raw columns
    out['power_W']      = (out['torque_Nm'] * out['rot_speed_rpm']) / 9550
    out['temp_delta_K'] = out['proc_temp_K'] - out['air_temp_K']
    out['torque_wear']  = out['torque_Nm'] * out['tool_wear_min']

    # 4. Product type — ordinal encoding of quality variant
    out['product_type'] = out['Type'].map({'L': 0, 'M': 1, 'H': 2})

    # 5 & 6. Drop leakage columns, Type, and any leftover ID-style columns
    out = out.drop(
        columns=_FAILURE_MODE_COLS + ['Type'],
        errors='ignore',
    )

    return out


def get_feature_columns() -> list:
    """
    Definitive ordered list of feature columns produced by engineer_features.
    Raw sensor features first (XGBoost tree splits benefit from all of them),
    then the three engineered interaction/ratio features.
    """
    return [
        # Raw sensors
        'air_temp_K',
        'proc_temp_K',
        'rot_speed_rpm',
        'torque_Nm',
        'tool_wear_min',
        # Engineered
        'power_W',
        'temp_delta_K',
        'torque_wear',
        'product_type',
    ]


def get_target_column() -> str:
    return 'Machine failure'