import numpy as np
import xgboost as xgb
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
    """Build an XGBClassifier with sensible defaults.  Pass kwargs to override."""
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
    min_recall: float = 0.85,
) -> float:
    """
    Sweep probability thresholds on a validation set and return the one
    that maximises MCC **while maintaining at least `min_recall`**.

    For predictive maintenance, missing a real failure (false negative) is
    far more costly than a false alarm (false positive), so we enforce a
    recall floor before optimising for overall quality (MCC).

    If no threshold meets the recall floor, the one with the highest recall
    is selected instead.
    """
    from sklearn.metrics import recall_score

    proba = model.predict_proba(X_val)[:, 1]
    results: list[tuple[float, float, float]] = []  # (threshold, mcc, recall)
    for t in np.arange(low, high + step, step):
        y_pred = (proba >= t).astype(int)
        mcc = float(matthews_corrcoef(y_val, y_pred))
        rec = float(recall_score(y_val, y_pred, zero_division=0))
        results.append((float(t), mcc, rec))

    # 1. Filter to thresholds that meet the recall floor
    valid = [(t, mcc, rec) for t, mcc, rec in results if rec >= min_recall]
    if valid:
        # 2. Among those, pick highest MCC
        best = max(valid, key=lambda x: x[1])
    else:
        # Fallback: maximise recall (shouldn't happen with min_recall=0.85)
        best = max(results, key=lambda x: x[2])

    return round(float(best[0]), 2)


# ── Optuna hyper-parameter tuning ───────────────────────────────────────────

def tune_model(
    X_train,
    y_train,
    scale_pos_weight: float,
    *,
    n_trials: int = 75,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """
    Optimise XGBoost hyperparameters using Optuna TPE, maximising mean MCC
    via stratified k-fold cross-validation (at the default 0.5 threshold).

    Threshold selection is handled separately by find_best_threshold() on
    the validation set after training, so hyperparameter tuning is stable
    and doesn't oscillate between runs.

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

