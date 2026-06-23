import os
import cv2
import joblib
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

# ==================================================
# PATHS
# ==================================================

MODEL_PATH = "saved_models/deepfake_image_model.h5"

# If you changed to .keras format:
# MODEL_PATH = "saved_models/deepfake_image_model.keras"

METRICS_PATH = "saved_models/deepfake_image_metrics.pkl"

CHART_PATH = "static/charts/image_prediction_confidence.png"

IMG_SIZE = 224

# ==================================================
# LOAD MODEL
# ==================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}\n"
        "Train image model first."
    )

model = load_model(MODEL_PATH)

# ==================================================
# LOAD METRICS
# ==================================================

if os.path.exists(METRICS_PATH):
    metrics = joblib.load(METRICS_PATH)
else:
    metrics = {
        "accuracy": 0,
        "precision": 0,
        "recall": 0,
        "f1_score": 0
    }

# ==================================================
# PREPROCESS IMAGE
# ==================================================

def preprocess_image(image_path):

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            f"Unable to read image: {image_path}"
        )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = cv2.resize(
        image,
        (IMG_SIZE, IMG_SIZE)
    )

    image = image / 255.0

    image = np.expand_dims(
        image,
        axis=0
    )

    return image

# ==================================================
# CONFIDENCE CHART
# ==================================================

def generate_confidence_chart(confidence):

    os.makedirs(
        "static/charts",
        exist_ok=True
    )

    plt.figure(figsize=(6, 4))

    plt.bar(
        ["Confidence"],
        [confidence]
    )

    plt.ylim(0, 100)

    plt.ylabel("Percentage")

    plt.title(
        "Deepfake Image Prediction Confidence"
    )

    plt.tight_layout()

    plt.savefig(CHART_PATH)

    plt.close()

# ==================================================
# PREDICTION
# ==================================================

def predict_image(image_path):

    image = preprocess_image(
        image_path
    )

    prediction = model.predict(
        image,
        verbose=0
    )

    predicted_class = np.argmax(
        prediction
    )

    confidence = float(
        np.max(prediction)
    ) * 100

    label = (
        "FAKE"
        if predicted_class == 1
        else "REAL"
    )

    generate_confidence_chart(
        confidence
    )

    result = {

        "prediction":
            label,

        "confidence":
            round(
                confidence,
                2
            ),

        "accuracy":
            metrics.get(
                "accuracy",
                0
            ),

        "precision":
            metrics.get(
                "precision",
                0
            ),

        "recall":
            metrics.get(
                "recall",
                0
            ),

        "f1_score":
            metrics.get(
                "f1_score",
                0
            ),

        "confidence_chart":
            CHART_PATH
    }

    return result

# ==================================================
# TEST
# ==================================================

if __name__ == "__main__":

    sample_image = (
        "datasets/deepfake_images/fake/fake_001.jpg"
    )

    result = predict_image(
        sample_image
    )

    print("\n" + "=" * 50)
    print("DEEPFAKE IMAGE DETECTION")
    print("=" * 50)

    print(
        f"Prediction : {result['prediction']}"
    )

    print(
        f"Confidence : {result['confidence']}%"
    )

    print("\nModel Performance")

    print(
        f"Accuracy  : {result['accuracy']}%"
    )

    print(
        f"Precision : {result['precision']}%"
    )

    print(
        f"Recall    : {result['recall']}%"
    )

    print(
        f"F1 Score  : {result['f1_score']}%"
    )

    print(
        "\nChart Saved:"
    )

    print(
        result["confidence_chart"]
    )