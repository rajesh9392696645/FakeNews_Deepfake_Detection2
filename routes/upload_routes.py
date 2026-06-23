from flask import Blueprint, request, jsonify
import os

upload_bp = Blueprint("upload", __name__)

UPLOAD_FOLDER = "static/uploads"

@upload_bp.route("/upload", methods=["POST"])
def upload_file():

    file = request.files["file"]

    if file:

        filepath = os.path.join(
            UPLOAD_FOLDER,
            file.filename
        )

        file.save(filepath)

        return jsonify({
            "message": "Uploaded Successfully",
            "path": filepath
        })

    return jsonify({
        "message": "Upload Failed"
    })