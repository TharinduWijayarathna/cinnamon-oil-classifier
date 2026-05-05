# Cinnamon Oil Purity Prediction API

Production-ready Flask backend that loads a trained hybrid model (image + tabular input) and exposes prediction endpoints.

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
│   ├── best_hybrid_model.keras
│   ├── hybrid_tabular_preprocessor.pkl
│   ├── label_encoder.pkl
│   └── model_summary.json
├── requirements.txt
├── run.py
└── ui/
```

## Setup

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

Server runs on `http://127.0.0.1:8000` by default.

Production run example:

```bash
gunicorn -w 2 -b 0.0.0.0:8000 run:app
```

## One-command Docker Setup

Run everything (backend + TypeScript UI):

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

## Endpoints

- `GET /` : API status, model type, required fields.
- `GET /health` : Health check and model loading status.
- `POST /predict` : Hybrid prediction using tabular + image input.

## cURL Test (multipart/form-data)

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "oil_mass=12.5" \
  -F "density=0.98" \
  -F "ph_value=6.2" \
  -F "oil_type=organic" \
  -F "image=@/absolute/path/to/sample_image.jpg"
```

## Sample Response

```json
{
  "prediction": "Pure",
  "predicted_class_index": 1,
  "confidence_percentage": 96.42,
  "raw_prediction_values": [[0.0358, 0.9642]],
  "classification_type": "multiclass"
}
```
