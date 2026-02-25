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
) -> float:
    """
    Select the deployment threshold at the END of the val MCC plateau.

    Strategy
    --------
    The MCC curve forms a hump: it rises as the threshold increases above
    the noise floor, peaks, then falls when recall starts to drop.  The hump
    is flat (same MCC) across a band of thresholds where recall hasn't
    changed yet.

    1. Sweep thresholds; keep only those where recall >= min_recall.
    2. Find the peak val MCC and note the recall at that point.
    3. Collect all thresholds that still hold that same recall level
       (i.e. the full flat top of the hump before recall first drops).
    4. Return the HIGHEST threshold in that band.

    Picking the end of the plateau (rather than the start) gives a more
    conservative, higher-precision threshold that generalises better to
    unseen data — without sacrificing any val recall or val MCC.
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

    # Find peak MCC and the recall level at which it occurs
    peak_entry = max(valid, key=lambda x: x[1])
    peak_mcc   = peak_entry[1]
    peak_rec   = peak_entry[2]   # recall level that goes with the MCC peak

    # Plateau = all valid thresholds that still hold that recall level
    # (MCC is flat here; once recall drops, MCC drops too)
    plateau = [(t, mcc, rec) for t, mcc, rec in valid if rec >= peak_rec]

    # Return the HIGHEST threshold on the plateau (end of the hump)
    return max(plateau, key=lambda x: x[0])[0]


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