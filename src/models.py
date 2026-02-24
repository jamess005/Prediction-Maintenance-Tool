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
    recall_tol: float = 0.01,
) -> float:
    """
    Select the threshold that maximises val MCC within the val recall plateau.

    Two-stage logic:
      1. Find the peak val recall and collect every threshold within
         `recall_tol` of it (the recall plateau).
      2. Among those, return the one with the highest val MCC.

    This keeps recall at its best achievable level first, then maximises
    precision/MCC within that constraint — rather than trading recall for
    precision prematurely.

    With ~50 failure cases in val, recall moves in discrete jumps of ~0.02
    per missed case. recall_tol=0.01 is tight enough that only thresholds
    which genuinely haven't dropped recall yet are included in the plateau.

    Example (from observed data):
        t=0.48  val_recall=0.94  val_MCC=0.901  <- plateau, MCC peak -> selected
        t=0.50  val_recall=0.94  val_MCC=0.901  <- plateau
        t=0.70  val_recall=0.92  val_MCC=0.899  <- recall dropped, excluded
        t=0.83  val_recall=0.90  val_MCC=0.897  <- recall dropped, excluded
    """
    from sklearn.metrics import recall_score

    proba = model.predict_proba(X_val)[:, 1]
    results = []
    for t in np.arange(low, high + step, step):
        y_pred = (proba >= t).astype(int)
        mcc = float(matthews_corrcoef(y_val, y_pred))
        rec = float(recall_score(y_val, y_pred, zero_division=0))
        results.append((round(float(t), 2), mcc, rec))

    peak_recall = max(rec for _, _, rec in results)

    # Stage 1: thresholds where val recall hasn't dropped from its peak
    recall_plateau = [(t, mcc, rec) for t, mcc, rec in results
                      if rec >= peak_recall - recall_tol]

    # Stage 2: best MCC among those
    best = max(recall_plateau, key=lambda x: x[1])
    return best[0]


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