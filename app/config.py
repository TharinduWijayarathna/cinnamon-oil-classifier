from pathlib import Path


class Config:
    BASE_DIR = Path(__file__).resolve().parent.parent
    MODELS_DIR = BASE_DIR / "models"

    HYBRID_MODEL_FILENAME = "best_hybrid_model.keras"
    PREPROCESSOR_FILENAME = "hybrid_tabular_preprocessor.pkl"
    LABEL_ENCODER_FILENAME = "label_encoder.pkl"
    SUMMARY_FILENAME = "model_summary.json"

    IMAGE_SIZE = (224, 224)
