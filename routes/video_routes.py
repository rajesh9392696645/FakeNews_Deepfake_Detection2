from flask import Blueprint, render_template, request

video_bp = Blueprint("video", __name__)

@video_bp.route("/video", methods=["GET", "POST"])
def video_detection():

    if request.method == "POST":

        video = request.files["video"]

        prediction = "Fake"
        confidence = 96.73

        return render_template(
            "result.html",
            prediction=prediction,
            confidence=confidence
        )

    return render_template("upload_video.html")