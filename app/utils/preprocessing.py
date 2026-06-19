from __future__ import annotations

import numpy as np
from PIL import Image

_EPS = 1e-8


def engineer_features(density: float, ph_value: float) -> dict[str, float]:
    return {
        "density": density,
        "ph_value": ph_value,
        "density_ph_ratio": density / (ph_value + _EPS),
        "ph_density_ratio": ph_value / (density + _EPS),
        "density_sq": density**2,
        "ph_sq": ph_value**2,
        "log_density": float(np.log1p(max(density, 0.0))),
        "log_ph": float(np.log1p(max(ph_value, 0.0))),
        "density_x_ph": density * ph_value,
    }


def preprocess_image(image_path: str, target_size: tuple[int, int]) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img).astype(np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)


def preprocess_tabular(
    density: float,
    ph_value: float,
    preprocessing: dict,
) -> np.ndarray:
    feature_names = preprocessing.get("feature_names") or preprocessing.get("all_features")
    if not feature_names:
        feature_names = ["density", "ph_value"]

    engineered = engineer_features(density, ph_value)
    values = np.array([engineered[name] for name in feature_names], dtype=np.float32)

    impute_values = np.asarray(preprocessing["impute_values"], dtype=np.float32)
    scale_mean = np.asarray(preprocessing["scale_mean"], dtype=np.float32)
    scale_std = np.asarray(preprocessing["scale_std"], dtype=np.float32)

    if len(impute_values) != len(values):
        raise ValueError(
            f"Expected {len(values)} preprocessing stats but got {len(impute_values)}."
        )

    for i, value in enumerate(values):
        if np.isnan(value):
            values[i] = impute_values[i]

    scaled = (values - scale_mean) / scale_std
    return np.expand_dims(scaled, axis=0)
