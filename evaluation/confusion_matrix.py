import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix


def generate_confusion_matrix(
    y_test,
    predictions,
    labels
):
    # Create directory if it doesn't exist
    os.makedirs(
        "static/charts",
        exist_ok=True
    )

    # Generate confusion matrix
    cm = confusion_matrix(
        y_test,
        predictions
    )

    # Plot heatmap
    plt.figure(
        figsize=(8, 6)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels
    )

    plt.title(
        "Confusion Matrix"
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )

    plt.tight_layout()

    # Save figure
    plt.savefig(
        "static/charts/confusion_matrix.png"
    )

    plt.close()