"""
Reusable evaluation helpers.

Used by:
  - pipeline.py   (automated CLI runs)
  - notebooks/    (interactive analysis)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    matthews_corrcoef,
    f1_score,
    make_scorer,
    confusion_matrix,
    classification_report,
)


# ── Cross-validated evaluation ──────────────────────────────────────────────

def evaluate_model(
    model,
    X_train,
    y_train,
    *,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """
    Run stratified k-fold CV and print MCC / F1.

    Returns a dict with arrays of per-fold scores plus means:
        {'mcc': array, 'f1': array, 'mcc_mean': float, 'f1_mean': float}
    """
    skf = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    mcc_scorer = make_scorer(matthews_corrcoef)

    mcc_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring=mcc_scorer)
    f1_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="f1")

    print(f"MCC  (mean ± std): {mcc_scores.mean():.3f} ± {mcc_scores.std():.3f}")
    print(f"F1   (mean ± std): {f1_scores.mean():.3f} ± {f1_scores.std():.3f}")

    return {
        "mcc": mcc_scores,
        "f1": f1_scores,
        "mcc_mean": mcc_scores.mean(),
        "f1_mean": f1_scores.mean(),
    }


# ── Confusion matrix ────────────────────────────────────────────────────────

def plot_confusion_matrix(
    model,
    X_test,
    y_test,
    *,
    save_path: str | None = None,
    show: bool = True,
) -> dict:
    """
    Plot an annotated confusion matrix and return scalar metrics.

    Returns:
        {'mcc': float, 'f1': float, 'cm': ndarray, 'report': str}
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=["No Failure", "Failure"]
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=["Predicted: No Failure", "Predicted: Failure"],
        yticklabels=["Actual: No Failure", "Actual: Failure"],
    )
    ax.set_title(f"Confusion Matrix — MCC: {mcc:.3f}  F1: {f1:.3f}")
    ax.set_xlabel(
        "Top-right = false alarms (unnecessary inspections)\n"
        "Bottom-left = missed failures (dangerous)"
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)

    print(report)
    return {"mcc": mcc, "f1": f1, "cm": cm, "report": report}
