# Cinnamon Oil Classifier

Flask API and React frontend for classifying cinnamon oil samples using a trained hybrid model (image + tabular inputs). The model predicts one of seven quality categories from a sample image, density, and pH value.

## Output Classes

| Class | Quality tier |
|-------|--------------|
| `bark_superior`, `bark_special` | High quality |
| `bark_average`, `bark_ordinary`, `leaf_pass` | Moderate |
| `leaf_fail`, `adulterated` | Low quality |

## Folder Structure

```text
predict-cinnamon-oil-purity/
├── Dockerfile
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── routes.py
│   ├── services/
│   │   └── model_service.py
│   └── utils/
│       └── preprocessing.py
├── docker-compose.yml
├── models/
│   ├── best_hybrid_model.keras   # Hybrid neural network (image + tabular)
│   └── model_config.json         # Class labels and tabular preprocessing params
├── notebook/                            # Training notebook and explanations
├── requirements.txt
├── run.py
└── ui/                                  # React + Vite frontend
    ├── src/
    │   ├── App.tsx
    │   ├── constants.ts
    │   ├── types.ts
    │   └── styles.css
    ├── index.html
    ├── package.json
    └── Dockerfile
```

## Local Setup

### Backend

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the API:

```bash
python run.py
```

The server runs at `http://127.0.0.1:8000`.

Production run example:

```bash
gunicorn -w 2 -b 0.0.0.0:8000 run:app
```

### Frontend

1. Install dependencies:

```bash
cd ui
npm install
```

2. Start the dev server:

```bash
npm run dev
```

The UI runs at `http://127.0.0.1:5173` and connects to the API at `http://<hostname>:8000` by default. Override with:

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000 npm run dev
```

## Docker Setup

Run the backend and UI together:

```bash
docker compose up --build
```

Services:

- Backend API: `http://127.0.0.1:8000`
- UI: `http://127.0.0.1:5173`

Stop all containers:

```bash
docker compose down
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | API status, model type, required fields |
| `GET` | `/health` | Health check and model loading status |
| `POST` | `/predict` | Classify a sample (multipart/form-data) |

### `POST /predict` fields

| Field | Type | Example |
|-------|------|---------|
| `density` | float | `1.042` |
| `ph_value` | float | `5.35` |
| `image` | file (jpg/png) | sample photo |

### cURL example

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "density=1.042" \
  -F "ph_value=5.35" \
  -F "image=@/absolute/path/to/sample_image.jpg"
```

### Sample response

```json
{
  "prediction": "bark_superior",
  "predicted_class_index": 4,
  "confidence_percentage": 96.42,
  "raw_prediction_values": [[0.01, 0.02, 0.01, 0.01, 0.94, 0.005, 0.005]],
  "classification_type": "multiclass"
}
```

## Models

The application needs only two files in `models/`:

| File | Purpose |
|------|---------|
| `best_hybrid_model.keras` | Fuses image and tabular features for classification (~19 MB) |
| `model_config.json` | Class label order and density/pH scaling parameters |

`model_config.json` replaces the separate preprocessor and label-encoder pickle files. Tabular preprocessing (median imputation + standard scaling) is applied in code using the values exported from training.

If you retrain the model, regenerate `model_config.json` with the new class names and scaler statistics from the notebook export step.
