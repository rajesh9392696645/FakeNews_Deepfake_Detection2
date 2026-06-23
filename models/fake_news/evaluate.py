import os
import joblib
import pandas as pd


import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        CURRENT_DIR,
        "..",
        ".."
    )
)

sys.path.insert(0, PROJECT_ROOT)

print("PROJECT ROOT:")
print(PROJECT_ROOT)

print("\nPYTHON PATH:")
print(sys.path[:3])
from evaluation.metrics import calculate_metrics
from evaluation.confusion_matrix import generate_confusion_matrix
from evaluation.roc_curve import generate_roc_curve
from evaluation.classification_report import generate_classification_report
from evaluation.performance_summary import save_performance_summary

from sklearn.model_selection import (
    train_test_split
)

from evaluation.metrics import (
    calculate_metrics
)

from evaluation.confusion_matrix import (
    generate_confusion_matrix
)

from evaluation.roc_curve import (
    generate_roc_curve
)

from evaluation.classification_report import (
    generate_classification_report
)

from evaluation.performance_summary import (
    save_performance_summary
)

DATASET = "datasets/fake_news/fake_news.csv"

MODEL_PATH = "saved_models/fake_news_model.pkl"


def evaluate_model():

    print("=" * 60)
    print("FAKE NEWS MODEL EVALUATION")
    print("=" * 60)

    model = joblib.load(
        MODEL_PATH
    )

    df = pd.read_csv(
        DATASET,
        low_memory=False
    )

    df = df[
        ["title", "text", "label"]
    ]

    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")

    df["label"] = (
        df["label"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df = df[
        df["label"].isin(
            ["FAKE", "REAL"]
        )
    ]

    df["news"] = (
        df["title"] + " " + df["text"]
    )

    X = df["news"]
    y = df["label"]

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )
    )

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )

    metrics = calculate_metrics(
        y_test,
        predictions
    )

    roc_auc = generate_roc_curve(
        y_test,
        probabilities
    )

    metrics["roc_auc"] = roc_auc

    generate_confusion_matrix(
        y_test,
        predictions,
        model.classes_
    )

    generate_classification_report(
        y_test,
        predictions
    )

    save_performance_summary(
        metrics
    )

    print("\nMODEL PERFORMANCE")

    print(
        f"Accuracy  : {metrics['accuracy']}%"
    )

    print(
        f"Precision : {metrics['precision']}%"
    )

    print(
        f"Recall    : {metrics['recall']}%"
    )

    print(
        f"F1 Score  : {metrics['f1_score']}%"
    )

    print(
        f"ROC AUC   : {metrics['roc_auc']}"
    )


if __name__ == "__main__":
    evaluate_model()