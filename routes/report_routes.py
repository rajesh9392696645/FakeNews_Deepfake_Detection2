from flask import Blueprint
from flask import render_template
from flask import send_file
import os


report_bp = Blueprint(
    "report",
    __name__
)


# REPORT PAGE
@report_bp.route("/reports")
def reports():

    report_content = "No report generated"

    report_path = (
        "results/metrics_results/"
        "fake_news_metrics.txt"
    )

    if os.path.exists(report_path):

        with open(
            report_path,
            "r",
            encoding="utf-8"
        ) as f:

            report_content = f.read()

    return render_template(
        "reports.html",
        report_content=report_content
    )


# PDF DOWNLOAD
@report_bp.route("/download/pdf")
def download_pdf():

    pdf_file = (
        "reports/pdf_reports/"
        "report.pdf"
    )

    return send_file(
        pdf_file,
        as_attachment=True
    )


# EXCEL DOWNLOAD
@report_bp.route("/download/excel")
def download_excel():

    excel_file = (
        "reports/excel_reports/"
        "report.xlsx"
    )

    return send_file(
        excel_file,
        as_attachment=True
    )