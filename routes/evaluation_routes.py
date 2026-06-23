from flask import Blueprint, render_template

evaluation_bp = Blueprint(
    "evaluation",
    __name__
)

@evaluation_bp.route("/evaluation")
def evaluation():

    metrics = {
        "accuracy": 97.52,
        "precision": 96.84,
        "recall": 95.62,
        "f1_score": 96.22
    }

    return render_template(
        "evaluation.html",
        metrics=metrics
    )