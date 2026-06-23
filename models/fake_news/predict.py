import os
import joblib
import matplotlib.pyplot as plt

MODEL_PATH = "saved_models/fake_news_model.pkl"
METRICS_PATH = "saved_models/model_metrics.pkl"

# -----------------------------------
# Check Files
# -----------------------------------

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

if not os.path.exists(METRICS_PATH):
    raise FileNotFoundError(
        f"Metrics not found: {METRICS_PATH}"
    )

# -----------------------------------
# Load Model
# -----------------------------------

model = joblib.load(
    MODEL_PATH
)

metrics = joblib.load(
    METRICS_PATH
)

# -----------------------------------
# Confidence Chart
# -----------------------------------

def generate_confidence_chart(
        confidence):

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

    plt.ylabel(
        "Percentage"
    )

    plt.title(
        "Prediction Confidence"
    )

    plt.tight_layout()

    plt.savefig(
        "static/charts/prediction_confidence.png"
    )

    plt.close()

# -----------------------------------
# Prediction Function
# -----------------------------------

def predict_news(news_text):

    prediction = model.predict(
        [news_text]
    )[0]

    probabilities = model.predict_proba(
        [news_text]
    )[0]

    confidence = (
        max(probabilities) * 100
    )

    generate_confidence_chart(
        confidence
    )

    result = {

        "prediction":
            prediction,

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

        "roc_auc":
            metrics.get(
                "roc_auc",
                "N/A"
            ),

        "classification_report":
            metrics.get(
                "classification_report",
                "Not Available"
            )
    }

    return result

# -----------------------------------
# Testing
# -----------------------------------

if __name__ == "__main__":

    sample_news = """
    Government announces new AI policy
    """

    result = predict_news(
        sample_news
    )

    print("\nPrediction Result")

    for key, value in result.items():

        print(
            f"{key} : {value}"
        )