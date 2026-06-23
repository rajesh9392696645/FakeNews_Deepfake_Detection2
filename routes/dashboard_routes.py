from flask import Blueprint, render_template
from database.db_operations import (
    get_total_predictions,
    get_fake_count,
    get_real_count
)

dashboard_bp = Blueprint("dashboard", __name__)

@dashboard_bp.route("/dashboard")
def dashboard():

    total = get_total_predictions()
    fake = get_fake_count()
    real = get_real_count()

    return render_template(
        "dashboard.html",
        total=total,
        fake=fake,
        real=real
    )