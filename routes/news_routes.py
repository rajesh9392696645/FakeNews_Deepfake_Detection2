from flask import Blueprint, render_template, request
import json
import os

from models.fake_news.predict import predict_news

news_bp = Blueprint("news", __name__)


@news_bp.route("/news", methods=["GET", "POST"])
def news_detection():

    if request.method == "POST":

        news_text = request.form["news"]

        result = predict_news(news_text)

        prediction = result["prediction"]
        confidence = result["confidence"]

        metrics_file = (
            "results/metrics_results/"
            "fake_news_metrics.json"
        )

        if os.path.exists(metrics_file):

            with open(
                metrics_file,
                "r",
                encoding="utf-8"
            ) as file:

                metrics = json.load(file)

        else:

            metrics = {
                "accuracy": 0,
                "precision": 0,
                "recall": 0,
                "f1_score": 0
            }

        return render_template(
            "result.html",
            prediction=prediction,
            confidence=confidence,
            accuracy=metrics["accuracy"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            f1_score=metrics["f1_score"]
        )

    return render_template(
        "upload_news.html"
    )