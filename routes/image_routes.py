from flask import Blueprint, render_template, request

image_bp = Blueprint("image", __name__)

@image_bp.route("/image", methods=["GET", "POST"])
def image_detection():

    if request.method == "POST":

        image = request.files["image"]

        prediction = "Fake"
        confidence = 97.55

        return render_template(
            "result.html",
            prediction=prediction,
            confidence=confidence
        )

    return render_template("upload_image.html")