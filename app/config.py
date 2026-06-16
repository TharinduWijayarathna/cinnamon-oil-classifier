from pathlib import Path


class Config:
    BASE_DIR = Path(__file__).resolve().parent.parent
    MODELS_DIR = BASE_DIR / "models"

    HYBRID_MODEL_FILENAME = "best_hybrid_model.keras"
    MODEL_CONFIG_FILENAME = "model_config.json"

    IMAGE_SIZE = (224, 224)
