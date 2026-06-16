from __future__ import annotations

import numpy as np
from PIL import Image


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
    impute_values = preprocessing["impute_values"]
    scale_mean = np.asarray(preprocessing["scale_mean"], dtype=np.float32)
    scale_std = np.asarray(preprocessing["scale_std"], dtype=np.float32)

    values = np.array([density, ph_value], dtype=np.float32)
    for i, value in enumerate(values):
        if np.isnan(value):
            values[i] = impute_values[i]

    scaled = (values - scale_mean) / scale_std
    return np.expand_dims(scaled, axis=0)
