from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image


def preprocess_image(image_path: str, target_size: tuple[int, int]) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img).astype(np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)


def build_tabular_dataframe(
    oil_mass: str, density: str, ph_value: str, oil_type: str
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "oil_mass": float(oil_mass),
                "density": float(density),
                "ph_value": float(ph_value),
                "oil_type": str(oil_type),
            }
        ]
    )


def preprocess_tabular_input(preprocessor, tabular_df: pd.DataFrame) -> np.ndarray:
    transformed = preprocessor.transform(tabular_df)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    return np.asarray(transformed, dtype=np.float32)
