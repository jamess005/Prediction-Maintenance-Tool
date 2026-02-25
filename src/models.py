import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, matthews_corrcoef

optuna.logging.set_verbosity(optuna.logging.WARNING)


def get_class_weight(y) -> float:
    n_neg = int((y == 0).sum())
    n_pos = int((y == 1).sum())
    return n_neg / n_pos


def build_model(
    scale_pos_weight: float,
    random_state: int = 42,
    **override_params,
) -> xgb.XGBClassifier:
    """Build an XGBClassifier with sensible defaults. Pass kwargs to override."""
    params = dict(
        n_estimators=500,
        learning_rate=0.015,
        max_depth=11,
        subsample=0.56,
        colsample_bytree=0.64,
        min_child_weight=2,
        gamma=0.17,
        reg_alpha=0.035,
        reg_lambda=2e-7,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        eval_metric="logloss",
        random_state=random_state,
        verbosity=0,
    )
    params.update(override_params)
    return xgb.XGBClassifier(**params)


def find_best_threshold(
    model,
    X_val,
    y_val,
    *,
    low: float = 0.1,
    high: float = 0.95,
    step: float = 0.01,
    min_recall: float = 0.80,
    mcc_tolerance: float = 0.01,
) -> float:
    """
    Select deployment threshold using the recall-biased near-plateau method.

    Strategy
    --------
    1. Sweep thresholds; keep only those where recall >= min_recall.
    2. Find the peak val MCC.
    3. Collect all thresholds within *mcc_tolerance* of the peak
       (the near-plateau around the maximum).
    4. Return the **lowest** threshold in that band.

    Why this works
    --------------
    The MCC curve is relatively flat near its peak — thresholds within a
    small tolerance achieve essentially the same val MCC.  Picking the
    lowest threshold in that band maximises recall, which:
      • generalises better (val-optimal threshold often overshoots on test),
      • is appropriate for predictive maintenance where missing a failure
        is costlier than a false alarm.

    The previous plateau-end strategy always picked the highest threshold
    that held peak recall, which worked for XGBoost (sharp MCC peak) but
    overshot for Random Forest (broad MCC hump).  This near-plateau
    approach adapts to both — it picks near the "start of the hump"
    (best recall) while staying within the high-MCC zone.
    """
    from sklearn.metrics import recall_score

    proba = model.predict_proba(X_val)[:, 1]
    results: list[tuple[float, float, float]] = []
    for t in np.arange(low, high + step, step):
        y_pred = (proba >= t).astype(int)
        mcc = float(matthews_corrcoef(y_val, y_pred))
        rec = float(recall_score(y_val, y_pred, zero_division=0))
        results.append((round(float(t), 2), mcc, rec))

    # Keep thresholds that meet the recall floor
    valid = [(t, mcc, rec) for t, mcc, rec in results if rec >= min_recall]
    if not valid:
        return max(results, key=lambda x: x[2])[0]   # fallback: best recall

    # Peak MCC among valid thresholds
    peak_mcc = max(valid, key=lambda x: x[1])[1]

    # Near-plateau: all thresholds within tolerance of the peak
    near_peak = [(t, mcc, rec) for t, mcc, rec in valid
                 if mcc >= peak_mcc - mcc_tolerance]

    # Return the LOWEST threshold in the near-plateau (maximises recall)
    return min(near_peak, key=lambda x: x[0])[0]


def tune_model(
    X_train,
    y_train,
    scale_pos_weight: float,
    *,
    n_trials: int = 100,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """
    Optimise XGBoost hyperparameters using Optuna TPE, maximising mean MCC
    via stratified k-fold cross-validation at the default 0.5 threshold.

    Threshold selection is handled separately by find_best_threshold() on the
    validation set, so tuning is stable and doesn't oscillate between runs.

    Note: Optuna CV uses a fixed n_estimators budget with no early stopping,
    so search MCC will be lower than the final model's MCC — this is expected.

    Returns:
        {'best_params': dict, 'best_mcc': float, 'study': optuna.Study}
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    mcc_scorer = make_scorer(matthews_corrcoef)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":      trial.suggest_int(  "n_estimators",      200, 1000),
            "learning_rate":     trial.suggest_float("learning_rate",     0.005, 0.2,  log=True),
            "max_depth":         trial.suggest_int(  "max_depth",         3, 12),
            "subsample":         trial.suggest_float("subsample",         0.4, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree",  0.4, 1.0),
            "min_child_weight":  trial.suggest_int(  "min_child_weight",  1, 20),
            "gamma":             trial.suggest_float("gamma",             0.0, 5.0),
            "reg_alpha":         trial.suggest_float("reg_alpha",         1e-8, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda",        1e-8, 10.0, log=True),
        }
        model = build_model(scale_pos_weight=scale_pos_weight, **params)
        return cross_val_score(
            model, X_train, y_train, cv=skf, scoring=mcc_scorer
        ).mean()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    return {
        "best_params": study.best_params,
        "best_mcc":    study.best_value,
        "study":       study,
    }


# ── Random Forest ────────────────────────────────────────────────────────────


def build_rf_model(
    class_weight: str | dict = "balanced",
    random_state: int = 42,
    **override_params,
) -> RandomForestClassifier:
    """Build a RandomForestClassifier with sensible defaults."""
    params = dict(
        n_estimators=500,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight=class_weight,
        n_jobs=-1,
        random_state=random_state,
    )
    params.update(override_params)
    return RandomForestClassifier(**params)  # type: ignore[arg-type]


def tune_rf_model(
    X_train,
    y_train,
    *,
    n_trials: int = 100,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """
    Optimise RandomForest hyperparameters using Optuna TPE, maximising mean
    MCC via stratified k-fold cross-validation at the default 0.5 threshold.

    Uses class_weight='balanced' throughout — sklearn's built-in mechanism
    for handling imbalanced classes (equivalent to scale_pos_weight for XGB).

    Returns:
        {'best_params': dict, 'best_mcc': float, 'study': optuna.Study}
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    mcc_scorer = make_scorer(matthews_corrcoef)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":     trial.suggest_int("n_estimators",     200, 1000),
            "max_depth":        trial.suggest_int("max_depth",        3, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf",  1, 20),
            "max_features":     trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }
        model = build_rf_model(class_weight="balanced", **params)
        return cross_val_score(
            model, X_train, y_train, cv=skf, scoring=mcc_scorer
        ).mean()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    return {
        "best_params": study.best_params,
        "best_mcc":    study.best_value,
        "study":       study,
    }