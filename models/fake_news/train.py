import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

DATASET = "datasets/fake_news/fake_news.csv"


def train_model():

    print("=" * 60)
    print("LOADING DATASET")
    print("=" * 60)

    df = pd.read_csv(
        DATASET,
        low_memory=False
    )

    print("Original Shape:", df.shape)

    # --------------------------------------------------
    # Validate Columns
    # --------------------------------------------------

    required_columns = [
        "title",
        "text",
        "label"
    ]

    for col in required_columns:
        if col not in df.columns:
            raise Exception(
                f"Missing Column: {col}"
            )

    # --------------------------------------------------
    # Data Cleaning
    # --------------------------------------------------

    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")

    df = df.dropna(
        subset=["label"]
    )

    df["label"] = (
        df["label"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # Combine title and text

    df["news"] = (
        df["title"] + " " + df["text"]
    )

    df = df[
        df["news"].str.strip() != ""
    ]

    print("\nDataset Shape After Cleaning:")
    print(df.shape)

    print("\nUnique Labels:")
    print(df["label"].unique())

    print("\nLabel Distribution Before Filtering:")
    print(df["label"].value_counts())

    # --------------------------------------------------
    # Remove Rare Classes
    # --------------------------------------------------

    label_counts = (
        df["label"]
        .value_counts()
    )

    valid_labels = label_counts[
        label_counts >= 2
    ].index

    df = df[
        df["label"].isin(
            valid_labels
        )
    ]

    print("\nLabel Distribution After Filtering:")
    print(df["label"].value_counts())

    # --------------------------------------------------
    # Features and Labels
    # --------------------------------------------------

    X = df["news"]
    y = df["label"]

    print("\nChecking Null Values")

    print("News Nulls :", X.isnull().sum())
    print("Label Nulls:", y.isnull().sum())

    # --------------------------------------------------
    # Train Test Split
    # --------------------------------------------------

    try:

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=0.20,
                random_state=42,
                stratify=y
            )
        )

        print("\nUsing Stratified Split")

    except ValueError as e:

        print("\nStratified Split Failed")
        print(e)

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=0.20,
                random_state=42
            )
        )

        print("Using Normal Split")

    print("\nTraining Samples:", len(X_train))
    print("Testing Samples :", len(X_test))

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    print("\nTraining Model...")

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=5000,
                stop_words="english"
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000
            )
        )
    ])

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    report = classification_report(
        y_test,
        predictions,
        zero_division=0
    )

    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE")
    print("=" * 60)

    print(f"Accuracy  : {accuracy*100:.2f}%")
    print(f"Precision : {precision*100:.2f}%")
    print(f"Recall    : {recall*100:.2f}%")
    print(f"F1 Score  : {f1*100:.2f}%")

    print("\nClassification Report")
    print(report)

    # --------------------------------------------------
    # Create Directories
    # --------------------------------------------------

    os.makedirs(
        "saved_models",
        exist_ok=True
    )

    os.makedirs(
        "static/charts",
        exist_ok=True
    )

    # --------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------

    cm = confusion_matrix(
        y_test,
        predictions
    )

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
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

    plt.savefig(
        "static/charts/confusion_matrix.png"
    )

    plt.close()

    print(
        "\nConfusion Matrix Saved"
    )

    # --------------------------------------------------
    # Metrics Dictionary
    # --------------------------------------------------

    metrics = {

        "accuracy":
            round(
                accuracy * 100,
                2
            ),

        "precision":
            round(
                precision * 100,
                2
            ),

        "recall":
            round(
                recall * 100,
                2
            ),

        "f1_score":
            round(
                f1 * 100,
                2
            ),

        "classification_report":
            report
    }

    # --------------------------------------------------
    # ROC Curve
    # --------------------------------------------------

    try:

        classes = model.classes_

        if len(classes) == 2:

            positive_class = classes[1]

            y_binary = (
                y_test ==
                positive_class
            ).astype(int)

            y_scores = (
                probabilities[:, 1]
            )

            fpr, tpr, _ = roc_curve(
                y_binary,
                y_scores
            )

            roc_auc = auc(
                fpr,
                tpr
            )

            plt.figure(figsize=(8, 6))

            plt.plot(
                fpr,
                tpr,
                label=f"AUC={roc_auc:.4f}"
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

            plt.legend()

            plt.tight_layout()

            plt.savefig(
                "static/charts/roc_curve.png"
            )

            plt.close()

            metrics["roc_auc"] = round(
                roc_auc,
                4
            )

            print(
                "ROC Curve Saved"
            )

    except Exception as e:

        print(
            "ROC Generation Error:",
            e
        )

    # --------------------------------------------------
    # Save Model
    # --------------------------------------------------

    model_path = (
        "saved_models/fake_news_model.pkl"
    )

    metrics_path = (
        "saved_models/model_metrics.pkl"
    )

    joblib.dump(
        model,
        model_path
    )

    joblib.dump(
        metrics,
        metrics_path
    )

    print("\n" + "=" * 60)
    print("MODEL SAVED SUCCESSFULLY")
    print("=" * 60)

    print(model_path)
    print(metrics_path)


if __name__ == "__main__":
    train_model()