"""
Reusable evaluation helpers.

Used by:
  - pipeline.py   (automated CLI runs)
  - notebooks/    (interactive analysis)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    matthews_corrcoef,
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix,
    classification_report,
)


def plot_confusion_matrix(
    model,
    X_test,
    y_test,
    *,
    threshold: float,
    save_path: str | None = None,
    show: bool = True,
) -> dict:
    """
    Plot an annotated confusion matrix and return scalar metrics.

    Returns:
        {'mcc': float, 'f1': float, 'recall': float, 'precision': float,
         'cm': ndarray, 'report': str}
    """
    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
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
    ax.set_title(
        f"Confusion Matrix — MCC: {mcc:.3f}  F1: {f1:.3f}  (threshold={threshold})"
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        import matplotlib
        if matplotlib.get_backend() != 'agg':
            plt.show()
        else:
            plt.close(fig)
    else:
        plt.close(fig)

    return {
        "mcc": float(mcc),
        "f1": float(f1),
        "recall": float(rec),
        "precision": float(prec),
        "cm": cm,
        "report": report,
    }