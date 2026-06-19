from __future__ import annotations

import json
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model

from app.utils.preprocessing import preprocess_image, preprocess_tabular


class ModelService:
    def __init__(
        self,
        models_dir: Path,
        hybrid_model_filename: str,
        model_config_filename: str,
    ) -> None:
        self.models_dir = Path(models_dir)
        self.hybrid_model_path = self.models_dir / hybrid_model_filename
        self.model_config_path = self.models_dir / model_config_filename

        self.hybrid_model = None
        self.model_config: dict = {}
        self._load_all()

    def _load_all(self) -> None:
        self._ensure_file(self.hybrid_model_path)
        self._ensure_file(self.model_config_path)

        self.hybrid_model = self._load_hybrid_model_with_compat()

        with self.model_config_path.open("r", encoding="utf-8") as f:
            self.model_config = json.load(f)

        required_keys = ["class_names", "tabular_preprocessing"]
        missing = [key for key in required_keys if key not in self.model_config]
        if missing:
            raise ValueError(f"model_config.json is missing required keys: {missing}")

        tabular_preprocessing = self.model_config["tabular_preprocessing"]
        feature_names = self.model_config.get("all_features")
        if feature_names:
            tabular_preprocessing = {
                **tabular_preprocessing,
                "feature_names": feature_names,
            }
            self.model_config["tabular_preprocessing"] = tabular_preprocessing

        expected_tabular_dim = len(tabular_preprocessing["scale_mean"])
        model_tabular_dim = self._tabular_input_dim()
        if model_tabular_dim is not None and model_tabular_dim != expected_tabular_dim:
            raise ValueError(
                "Model tabular input size "
                f"({model_tabular_dim}) does not match model_config "
                f"({expected_tabular_dim})."
            )

    @staticmethod
    def _ensure_file(path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Required model file is missing: {path}")

    def _load_hybrid_model_with_compat(self):
        try:
            return load_model(self.hybrid_model_path, compile=False)
        except TypeError as error:
            if "quantization_config" not in str(error):
                raise

            class DenseCompat(Dense):
                def __init__(self, *args, **kwargs):
                    kwargs.pop("quantization_config", None)
                    super().__init__(*args, **kwargs)

            try:
                return load_model(
                    self.hybrid_model_path,
                    compile=False,
                    custom_objects={
                        "Dense": DenseCompat,
                        "keras.layers.Dense": DenseCompat,
                        "keras.src.layers.core.dense.Dense": DenseCompat,
                    },
                )
            except TypeError as compat_error:
                if "quantization_config" not in str(compat_error):
                    raise
                return self._load_model_after_stripping_quantization_config()

    def _load_model_after_stripping_quantization_config(self):
        temp_model_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
                temp_model_path = tmp.name

            with zipfile.ZipFile(self.hybrid_model_path, "r") as source_zip:
                with zipfile.ZipFile(temp_model_path, "w", zipfile.ZIP_DEFLATED) as target_zip:
                    for file_info in source_zip.infolist():
                        data = source_zip.read(file_info.filename)
                        if file_info.filename == "config.json":
                            model_config = json.loads(data.decode("utf-8"))
                            self._remove_quantization_config_keys(model_config)
                            data = json.dumps(model_config).encode("utf-8")
                        target_zip.writestr(file_info, data)

            return load_model(temp_model_path, compile=False)
        finally:
            if temp_model_path and os.path.exists(temp_model_path):
                os.remove(temp_model_path)

    def _tabular_input_dim(self) -> int | None:
        if self.hybrid_model is None:
            return None

        try:
            layer = self.hybrid_model.get_layer("tabular_input")
        except ValueError:
            return None

        shape = getattr(layer, "shape", None)
        if shape is None:
            shape = getattr(layer, "batch_input_shape", None)
        if shape and len(shape) >= 2 and shape[-1] is not None:
            return int(shape[-1])

        for model_input in self.hybrid_model.inputs:
            if model_input.name.endswith("tabular_input"):
                if model_input.shape[-1] is not None:
                    return int(model_input.shape[-1])
        return None

    @staticmethod
    def _remove_quantization_config_keys(value):
        if isinstance(value, dict):
            value.pop("quantization_config", None)
            for nested in value.values():
                ModelService._remove_quantization_config_keys(nested)
        elif isinstance(value, list):
            for item in value:
                ModelService._remove_quantization_config_keys(item)

    def metadata(self) -> dict:
        return {
            "status": "ok",
            "model_type": self.model_config.get("model_type", "hybrid"),
            "class_names": self.model_config.get("class_names", []),
            "required_fields": ["density", "ph_value", "image"],
        }

    def health(self) -> dict:
        loaded = self.hybrid_model is not None and bool(self.model_config)
        return {
            "status": "healthy" if loaded else "unhealthy",
            "models_loaded": loaded,
            "model_type": self.model_config.get("model_type", "hybrid"),
        }

    def predict(
        self,
        image_path: str,
        density: str,
        ph_value: str,
        image_size: tuple[int, int],
    ) -> dict:
        image_input = preprocess_image(image_path=image_path, target_size=image_size)
        tabular_input = preprocess_tabular(
            density=float(density),
            ph_value=float(ph_value),
            preprocessing=self.model_config["tabular_preprocessing"],
        )

        prediction = self.hybrid_model.predict([image_input, tabular_input], verbose=0)
        prediction = np.asarray(prediction).reshape(1, -1)
        predicted_class = int(np.argmax(prediction[0]))
        confidence = float(np.max(prediction[0]))

        class_names = self.model_config["class_names"]
        if predicted_class < 0 or predicted_class >= len(class_names):
            raise ValueError(f"Predicted class index {predicted_class} is out of range.")

        predicted_label = class_names[predicted_class]

        return {
            "prediction": str(predicted_label),
            "predicted_class_index": predicted_class,
            "confidence_percentage": round(confidence * 100.0, 2),
            "raw_prediction_values": prediction.tolist(),
            "classification_type": "multiclass",
        }
