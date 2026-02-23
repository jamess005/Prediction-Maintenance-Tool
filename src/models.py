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
    """
    Build an XGBClassifier.

    Defaults are the hyperparameters found by Optuna (TPE, 50 trials, MCC objective).
    Pass keyword arguments to override any parameter.
    """
    params = dict(
        # Found by Optuna — do not change without re-running tuning
        n_estimators=500,           # Optuna best: 477; fixed to avoid early-stopping complications in CV
        learning_rate=0.0151,
        max_depth=10,
        subsample=0.528,
        colsample_bytree=0.724,
        min_child_weight=4,
        gamma=1.423,
        reg_alpha=2.8e-7,
        reg_lambda=0.0003,
        # Fixed settings
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        # logloss is the stable early-stopping signal here: the val set has only
        # ~34 positive examples so aucpr is too noisy to drive early stopping reliably.
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
    n_trials: int = 75,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """
    Optimise XGBoost hyperparameters using Optuna TPE, maximising mean MCC
    via stratified k-fold CV.  Returns:
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

    print(f"Best CV MCC : {study.best_value:.4f}")
    print(f"Best params : {study.best_params}")
    return {
        "best_params": study.best_params,
        "best_mcc":    study.best_value,
        "study":       study,
    }

