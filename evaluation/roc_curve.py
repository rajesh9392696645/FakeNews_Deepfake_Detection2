import os
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    auc
)


def generate_roc_curve(y_test, probabilities):
    # Create directory if it does not exist
    os.makedirs(
        "static/charts",
        exist_ok=True
    )

    # Convert labels to binary values
    y_binary = (y_test == "REAL").astype(int)

    # Probability scores for positive class
    y_score = probabilities[:, 1]

    # Compute ROC curve
    fpr, tpr, _ = roc_curve(
        y_binary,
        y_score
    )

    # Compute AUC
    roc_auc = auc(
        fpr,
        tpr
    )

    # Plot ROC Curve
    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"AUC = {roc_auc:.4f}"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "ROC Curve"
    )

    plt.legend(
        loc="lower right"
    )

    plt.tight_layout()

    plt.savefig(
        "static/charts/roc_curve.png"
    )

    plt.close()

    return round(
        roc_auc,
        4
    )