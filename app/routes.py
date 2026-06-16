from __future__ import annotations

import os
import tempfile

from flask import Blueprint, current_app, jsonify, request
from werkzeug.utils import secure_filename

api_bp = Blueprint("api", __name__)

REQUIRED_FORM_FIELDS = ["density", "ph_value"]


@api_bp.get("/")
def root():
    model_service = current_app.extensions["model_service"]
    return jsonify(model_service.metadata()), 200


@api_bp.get("/health")
def health():
    model_service = current_app.extensions["model_service"]
    payload = model_service.health()
    status_code = 200 if payload.get("models_loaded") else 500
    return jsonify(payload), status_code


@api_bp.post("/predict")
def predict():
    model_service = current_app.extensions["model_service"]

    missing_fields = [
        field for field in REQUIRED_FORM_FIELDS if field not in request.form or not request.form.get(field)
    ]
    if missing_fields:
        return (
            jsonify(
                {
                    "error": "Missing required form-data fields",
                    "missing_fields": missing_fields,
                }
            ),
            400,
        )

    if "image" not in request.files:
        return jsonify({"error": "Image file is required under field name 'image'"}), 400

    image_file = request.files["image"]
    if not image_file or image_file.filename == "":
        return jsonify({"error": "Uploaded image file is empty"}), 400

    safe_name = secure_filename(image_file.filename) or "uploaded_image"
    suffix = os.path.splitext(safe_name)[1] or ".jpg"

    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file_path = temp_file.name
            image_file.save(temp_file_path)

        result = model_service.predict(
            image_path=temp_file_path,
            density=request.form["density"],
            ph_value=request.form["ph_value"],
            image_size=current_app.config["IMAGE_SIZE"],
        )

        return jsonify(result), 200
    except ValueError as error:
        return jsonify({"error": "Invalid input values", "details": str(error)}), 400
    except Exception as error:
        return jsonify({"error": "Prediction failed", "details": str(error)}), 500
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
