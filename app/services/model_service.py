from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from tensorflow.keras.models import load_model

from app.utils.preprocessing import (
    build_tabular_dataframe,
    preprocess_image,
    preprocess_tabular_input,
)


class ModelService:
    def __init__(
        self,
        models_dir: Path,
        hybrid_model_filename: str,
        preprocessor_filename: str,
        label_encoder_filename: str,
        summary_filename: str,
    ) -> None:
        self.models_dir = Path(models_dir)
        self.hybrid_model_path = self.models_dir / hybrid_model_filename
        self.preprocessor_path = self.models_dir / preprocessor_filename
        self.label_encoder_path = self.models_dir / label_encoder_filename
        self.summary_path = self.models_dir / summary_filename

        self.hybrid_model = None
        self.hybrid_preprocessor = None
        self.label_encoder = None
        self.model_summary = {}
        self._load_all()

    def _load_all(self) -> None:
        self._ensure_file(self.hybrid_model_path)
        self._ensure_file(self.preprocessor_path)
        self._ensure_file(self.label_encoder_path)
        self._ensure_file(self.summary_path)

        self.hybrid_model = load_model(self.hybrid_model_path)
        self.hybrid_preprocessor = joblib.load(self.preprocessor_path)
        self.label_encoder = joblib.load(self.label_encoder_path)

        with self.summary_path.open("r", encoding="utf-8") as f:
            self.model_summary = json.load(f)

    @staticmethod
    def _ensure_file(path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Required model file is missing: {path}")

    def metadata(self) -> dict:
        return {
            "status": "ok",
            "model_type": "hybrid",
            "best_hybrid_model": self.model_summary.get("best_hybrid_model", {}),
            "required_fields": ["oil_mass", "density", "ph_value", "oil_type", "image"],
        }

    def health(self) -> dict:
        loaded = all(
            [
                self.hybrid_model is not None,
                self.hybrid_preprocessor is not None,
                self.label_encoder is not None,
            ]
        )
        return {
            "status": "healthy" if loaded else "unhealthy",
            "models_loaded": loaded,
            "model_type": "hybrid",
        }

    def predict(
        self,
        image_path: str,
        oil_mass: str,
        density: str,
        ph_value: str,
        oil_type: str,
        image_size: tuple[int, int],
    ) -> dict:
        image_input = preprocess_image(image_path=image_path, target_size=image_size)

        tabular_df = build_tabular_dataframe(
            oil_mass=oil_mass, density=density, ph_value=ph_value, oil_type=oil_type
        )
        tabular_input = preprocess_tabular_input(self.hybrid_preprocessor, tabular_df)

        prediction = self.hybrid_model.predict([image_input, tabular_input], verbose=0)
        prediction = np.asarray(prediction)

        if prediction.ndim == 2 and prediction.shape[1] == 1:
            raw_score = float(prediction.flatten()[0])
            predicted_class = int(raw_score >= 0.5)
            confidence = raw_score
            if predicted_class == 0:
                confidence = 1.0 - confidence
        elif prediction.ndim == 1:
            raw_score = float(prediction.flatten()[0])
            predicted_class = int(raw_score >= 0.5)
            confidence = raw_score
            if predicted_class == 0:
                confidence = 1.0 - confidence
        else:
            predicted_class = int(np.argmax(prediction[0]))
            confidence = float(np.max(prediction[0]))

        predicted_label = self.label_encoder.inverse_transform([predicted_class])[0]

        return {
            "prediction": str(predicted_label),
            "predicted_class_index": predicted_class,
            "confidence_percentage": round(confidence * 100.0, 2),
            "raw_prediction_values": prediction.tolist(),
            "classification_type": "binary"
            if (prediction.ndim == 1 or (prediction.ndim == 2 and prediction.shape[1] == 1))
            else "multiclass",
        }
