
from flask import Flask

from routes.auth_routes import auth_bp
from routes.dashboard_routes import dashboard_bp
from routes.upload_routes import upload_bp
from routes.news_routes import news_bp
from routes.image_routes import image_bp
from routes.video_routes import video_bp
from routes.generation_routes import generation_bp
from routes.history_routes import history_bp
from routes.report_routes import report_bp
from routes.evaluation_routes import evaluation_bp


def create_app():

    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static"
    )

    app.secret_key = "fake_news_deepfake_secret_key"

    app.register_blueprint(auth_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(upload_bp)
    app.register_blueprint(news_bp)
    app.register_blueprint(image_bp)
    app.register_blueprint(video_bp)
    app.register_blueprint(generation_bp)
    app.register_blueprint(history_bp)
    app.register_blueprint(report_bp)
    app.register_blueprint(evaluation_bp)

    return app