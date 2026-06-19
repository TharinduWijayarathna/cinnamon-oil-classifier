#!/usr/bin/env python3
"""Export model_config.json from Colab training artifacts.

After retraining, copy hybrid_tabular_preprocessor.pkl and label_encoder.pkl
from the notebook export into models/, then run this script. Only
best_hybrid_model.keras and model_config.json are needed at runtime.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib


def export_model_config(models_dir: Path) -> dict:
    preprocessor_path = models_dir / "hybrid_tabular_preprocessor.pkl"
    label_encoder_path = models_dir / "label_encoder.pkl"
    summary_path = models_dir / "model_summary.json"

    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Missing preprocessor: {preprocessor_path}")
    if not label_encoder_path.exists():
        raise FileNotFoundError(f"Missing label encoder: {label_encoder_path}")

    preprocessor = joblib.load(preprocessor_path)
    label_encoder = joblib.load(label_encoder_path)

    numeric_pipeline = preprocessor.named_transformers_["num"]
    imputer = numeric_pipeline.named_steps["imputer"]
    scaler = numeric_pipeline.named_steps["scaler"]
    all_features = list(preprocessor.feature_names_in_)

    config = {
        "model_type": "hybrid",
        "target_column": "result",
        "numerical_features": ["density", "ph_value"],
        "all_features": all_features,
        "class_names": [str(label) for label in label_encoder.classes_],
        "tabular_preprocessing": {
            "impute_strategy": "median",
            "impute_values": imputer.statistics_.tolist(),
            "scale_mean": scaler.mean_.tolist(),
            "scale_std": scaler.scale_.tolist(),
        },
    }

    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        config["target_column"] = summary.get("target_column", config["target_column"])
        config["numerical_features"] = summary.get(
            "numerical_features", config["numerical_features"]
        )

    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "models",
        help="Directory containing hybrid_tabular_preprocessor.pkl and label_encoder.pkl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (defaults to <models-dir>/model_config.json)",
    )
    args = parser.parse_args()

    config = export_model_config(args.models_dir)
    output_path = args.output or (args.models_dir / "model_config.json")
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")

    print(f"Wrote {output_path}")
    print(f"  features: {len(config['all_features'])}")
    print(f"  classes: {len(config['class_names'])}")


if __name__ == "__main__":
    main()
