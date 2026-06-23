import os
import json
import cv2
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import load_model

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

# =====================================================
# PATHS
# =====================================================

DATASET_PATH = "datasets/deepfake_images"

REAL_PATH = os.path.join(
    DATASET_PATH,
    "real"
)

FAKE_PATH = os.path.join(
    DATASET_PATH,
    "fake"
)

MODEL_PATH = (
    "saved_models/deepfake_image_model.h5"
)

# If using keras format:
# MODEL_PATH = "saved_models/deepfake_image_model.keras"

IMG_SIZE = 224

# =====================================================
# LOAD DATASET
# =====================================================

def load_dataset():

    images = []
    labels = []

    valid_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp"
    )

    print("\nLoading REAL Images...")

    real_count = 0

    for file in os.listdir(REAL_PATH):

        if not file.lower().endswith(
            valid_extensions
        ):
            continue

        path = os.path.join(
            REAL_PATH,
            file
        )

        image = cv2.imread(path)

        if image is None:
            continue

        image = cv2.resize(
            image,
            (IMG_SIZE, IMG_SIZE)
        )

        image = image.astype(
            np.float32
        ) / 255.0

        images.append(image)
        labels.append(0)

        real_count += 1

    print(
        f"REAL Images : {real_count}"
    )

    print("\nLoading FAKE Images...")

    fake_count = 0

    for file in os.listdir(FAKE_PATH):

        if not file.lower().endswith(
            valid_extensions
        ):
            continue

        path = os.path.join(
            FAKE_PATH,
            file
        )

        image = cv2.imread(path)

        if image is None:
            continue

        image = cv2.resize(
            image,
            (IMG_SIZE, IMG_SIZE)
        )

        image = image.astype(
            np.float32
        ) / 255.0

        images.append(image)
        labels.append(1)

        fake_count += 1

    print(
        f"FAKE Images : {fake_count}"
    )

    return (
        np.array(images),
        np.array(labels)
    )


# =====================================================
# EVALUATE MODEL
# =====================================================

def evaluate_model():

    print("=" * 60)
    print("DEEPFAKE IMAGE MODEL EVALUATION")
    print("=" * 60)

    if not os.path.exists(
        MODEL_PATH
    ):
        print(
            f"Model not found: {MODEL_PATH}"
        )
        return

    model = load_model(
        MODEL_PATH
    )

    X, y = load_dataset()

    print(
        f"\nTotal Images: {len(X)}"
    )

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )
    )

    print("\nPredicting...")

    probabilities = model.predict(
        X_test,
        verbose=0
    )

    print(
        f"Prediction Shape: {probabilities.shape}"
    )

    # =====================================================
    # HANDLE DIFFERENT MODEL OUTPUTS
    # =====================================================

    if len(probabilities.shape) == 2:

        # Softmax output (N,2)
        if probabilities.shape[1] == 2:

            predictions = np.argmax(
                probabilities,
                axis=1
            )

            roc_scores = probabilities[:, 1]

        # Sigmoid output (N,1)
        else:

            roc_scores = (
                probabilities.flatten()
            )

            predictions = (
                roc_scores > 0.5
            ).astype(int)

    else:

        roc_scores = (
            probabilities.flatten()
        )

        predictions = (
            roc_scores > 0.5
        ).astype(int)

    predictions = predictions.flatten()

    # =====================================================
    # METRICS
    # =====================================================

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE")
    print("=" * 60)

    print(
        f"Accuracy  : {accuracy * 100:.2f}%"
    )

    print(
        f"Precision : {precision * 100:.2f}%"
    )

    print(
        f"Recall    : {recall * 100:.2f}%"
    )

    print(
        f"F1 Score  : {f1 * 100:.2f}%"
    )

    # =====================================================
    # CREATE DIRECTORIES
    # =====================================================

    os.makedirs(
        "results/metrics_results",
        exist_ok=True
    )

    os.makedirs(
        "saved_models",
        exist_ok=True
    )

    os.makedirs(
        "static/charts",
        exist_ok=True
    )

    # =====================================================
    # ROC AUC
    # =====================================================

    try:

        fpr, tpr, _ = roc_curve(
            y_test,
            roc_scores
        )

        roc_auc = auc(
            fpr,
            tpr
        )

    except Exception as e:

        print(
            f"ROC Error: {e}"
        )

        roc_auc = 0.0

    # =====================================================
    # SAVE METRICS
    # =====================================================

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

        "roc_auc":
            round(
                roc_auc,
                4
            )
    }

    with open(
        "results/metrics_results/deepfake_image_metrics.json",
        "w"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    joblib.dump(
        metrics,
        "saved_models/deepfake_image_metrics.pkl"
    )

    # =====================================================
    # CLASSIFICATION REPORT
    # =====================================================

    report = classification_report(
        y_test,
        predictions,
        target_names=[
            "REAL",
            "FAKE"
        ],
        zero_division=0
    )

    with open(
        "results/metrics_results/image_classification_report.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            report
        )

    print(
        "\nClassification Report Saved"
    )

    # =====================================================
    # CONFUSION MATRIX
    # =====================================================

    cm = confusion_matrix(
        y_test,
        predictions
    )

    plt.figure(
        figsize=(8, 6)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[
            "REAL",
            "FAKE"
        ],
        yticklabels=[
            "REAL",
            "FAKE"
        ]
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
        "static/charts/image_confusion_matrix.png"
    )

    plt.close()

    print(
        "Confusion Matrix Saved"
    )

    # =====================================================
    # ROC CURVE
    # =====================================================

    try:

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
            "static/charts/image_roc_curve.png"
        )

        plt.close()

        print(
            "ROC Curve Saved"
        )

    except Exception as e:

        print(
            f"ROC Plot Error: {e}"
        )

    print("\nEvaluation Complete")


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    evaluate_model()