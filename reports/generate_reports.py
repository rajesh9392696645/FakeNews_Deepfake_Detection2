import os
import json
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


class ReportGenerator:

    def __init__(self, output_folder="results"):
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)

    def save_classification_report(
        self,
        y_true,
        y_pred,
        report_name="classification_report.txt"
    ):

        report = classification_report(y_true, y_pred)

        file_path = os.path.join(
            self.output_folder,
            report_name
        )

        with open(file_path, "w") as f:
            f.write(report)

        print("Report Saved:", file_path)

    def save_metrics(
        self,
        y_true,
        y_pred,
        metric_name="metrics.json"
    ):

        metrics = {
            "accuracy":
                accuracy_score(y_true, y_pred),

            "precision":
                precision_score(
                    y_true,
                    y_pred,
                    average="weighted"
                ),

            "recall":
                recall_score(
                    y_true,
                    y_pred,
                    average="weighted"
                ),

            "f1_score":
                f1_score(
                    y_true,
                    y_pred,
                    average="weighted"
                ),

            "confusion_matrix":
                confusion_matrix(
                    y_true,
                    y_pred
                ).tolist()
        }

        file_path = os.path.join(
            self.output_folder,
            metric_name
        )

        with open(file_path, "w") as f:
            json.dump(
                metrics,
                f,
                indent=4
            )

        print("Metrics Saved:", file_path)