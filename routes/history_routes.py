from flask import Blueprint, render_template
from database.db_operations import get_all_predictions

history_bp = Blueprint("history", __name__)

@history_bp.route("/history")
def history():

    records = get_all_predictions()

    return render_template(
        "history.html",
        records=records
    )