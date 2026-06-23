from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def calculate_metrics(
    y_test,
    predictions
):
    return {

        "accuracy":
            round(
                accuracy_score(
                    y_test,
                    predictions
                ) * 100,
                2
            ),

        "precision":
            round(
                precision_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0
                ) * 100,
                2
            ),

        "recall":
            round(
                recall_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0
                ) * 100,
                2
            ),

        "f1_score":
            round(
                f1_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0
                ) * 100,
                2
            )
    }