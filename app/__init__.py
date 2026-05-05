from flask import Flask, jsonify
from flask_cors import CORS

from app.config import Config
from app.routes import api_bp
from app.services.model_service import ModelService


def create_app() -> Flask:
    app = Flask(__name__)
    app.config.from_object(Config)
    CORS(app)

    model_service = ModelService(
        models_dir=app.config["MODELS_DIR"],
        hybrid_model_filename=app.config["HYBRID_MODEL_FILENAME"],
        preprocessor_filename=app.config["PREPROCESSOR_FILENAME"],
        label_encoder_filename=app.config["LABEL_ENCODER_FILENAME"],
        summary_filename=app.config["SUMMARY_FILENAME"],
    )
    app.extensions["model_service"] = model_service

    app.register_blueprint(api_bp)
    register_error_handlers(app)
    return app


def register_error_handlers(app: Flask) -> None:
    @app.errorhandler(ValueError)
    def handle_value_error(error: ValueError):
        return jsonify({"error": "Validation error", "details": str(error)}), 400

    @app.errorhandler(FileNotFoundError)
    def handle_file_not_found(error: FileNotFoundError):
        return jsonify({"error": "File not found", "details": str(error)}), 500

    @app.errorhandler(Exception)
    def handle_unexpected(error: Exception):
        return jsonify({"error": "Internal server error", "details": str(error)}), 500
