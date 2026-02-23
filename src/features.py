import pandas as pd

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes raw AI4I dataframe and returns dataframe with three
    engineered features replacing the five raw sensor features.
    
    Raw features:
        - Air temperature [K]
        - Process temperature [K]  
        - Rotational speed [rpm]
        - Torque [Nm]
        - Tool wear [min]
    
    Engineered features:
        - power_W         : mechanical power (Torque x RPM / 9550)
        - temp_delta_K    : process temp - air temp (cooling effectiveness)
        - tool_wear_min   : kept raw (independent of all other features)
    """
    out = df.copy()

    # Mechanical power — combines torque and RPM into actual load on machine
    out['power_W'] = (out['Torque [Nm]'] * out['Rotational speed [rpm]']) / 9550

    # Temperature delta — strips out ambient noise, isolates cooling signal
    out['temp_delta_K'] = out['Process temperature [K]'] - out['Air temperature [K]']

    # Tool wear stays raw — completely independent dimension
    out['tool_wear_min'] = out['Tool wear [min]']

    # Drop raw features that have been replaced
    out = out.drop(columns=[
        'Air temperature [K]',
        'Process temperature [K]',
        'Rotational speed [rpm]',
        'Torque [Nm]',
        'Tool wear [min]'
    ])

    return out


def get_feature_columns() -> list:
    return ['power_W', 'temp_delta_K', 'tool_wear_min']


def get_target_column() -> str:
    return 'Machine failure'