from flask import Blueprint, render_template, request

generation_bp = Blueprint("generation", __name__)

@generation_bp.route("/generate-image")
def generate_image():

    image_path = "generated_images/fake_image.png"

    return render_template(
        "generate_fake.html",
        image_path=image_path
    )


@generation_bp.route("/generate-video")
def generate_video():

    video_path = "generated_fake_videos/fake_video.mp4"

    return render_template(
        "generate_fake_video.html",
        video_path=video_path
    )