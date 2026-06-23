import os
import json
import joblib


def save_performance_summary(metrics):
    # Create required directories
    os.makedirs(
        "results/metrics_results",
        exist_ok=True
    )

    os.makedirs(
        "saved_models",
        exist_ok=True
    )

    # Save metrics as JSON
    with open(
        "results/metrics_results/fake_news_metrics.json",
        "w"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    # Save metrics as a pickle file
    joblib.dump(
        metrics,
        "saved_models/model_metrics.pkl"
    )

    print(
        "Performance metrics saved successfully."
    )