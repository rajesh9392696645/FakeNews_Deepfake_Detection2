import os

from sklearn.metrics import (
    classification_report
)


def generate_classification_report(
    y_test,
    predictions
):
    # Create directory if it does not exist
    os.makedirs(
        "results/metrics_results",
        exist_ok=True
    )

    # Generate classification report
    report = classification_report(
        y_test,
        predictions,
        zero_division=0
    )

    # Save report to file
    with open(
        "results/metrics_results/classification_report.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            report
        )

    return report