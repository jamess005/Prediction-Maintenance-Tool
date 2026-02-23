import xgboost as xgb
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, matthews_corrcoef

optuna.logging.set_verbosity(optuna.logging.WARNING)


def get_class_weight(y) -> float:
    n_negative = (y == 0).sum()
    n_positive = (y == 1).sum()
    weight = n_negative / n_positive
    print(f"Class weight (scale_pos_weight): {weight:.2f}")
    print(f"  ({n_negative} no-failure rows vs {n_positive} failure rows)")
    return weight


def build_model(
    scale_pos_weight: float,
    random_state: int = 42,
    **override_params,
) -> xgb.XGBClassifier:
    """Build an XGBClassifier with sensible defaults.

    Pass keyword arguments to override any hyperparameter (e.g. after tuning).
    """
    params = dict(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        eval_metric="logloss",
        random_state=random_state,
        verbosity=0,
    )
    params.update(override_params)
    return xgb.XGBClassifier(**params)


# ── Optuna hyper-parameter tuning ───────────────────────────────────────────

def tune_model(
    X_train,
    y_train,
    scale_pos_weight: float,
    *,
    n_trials: int = 50,
    n_splits: int = 5,
    random_state: int = 42,
    show_progress: bool = True,
) -> dict:
    """
    Use Optuna (TPE sampler) to find the best XGBoost hyperparameters,
    optimising mean MCC via stratified k-fold CV.

    Returns:
        {'best_params': dict, 'best_mcc': float, 'study': optuna.Study}
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    mcc_scorer = make_scorer(matthews_corrcoef)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        model = build_model(scale_pos_weight=scale_pos_weight, **params)
        scores = cross_val_score(model, X_train, y_train, cv=skf, scoring=mcc_scorer)
        return scores.mean()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=show_progress,
    )

    best = study.best_params
    print(f"Best MCC: {study.best_value:.4f}")
    print(f"Best params: {best}")

    return {
        "best_params": best,
        "best_mcc": study.best_value,
        "study": study,
    }