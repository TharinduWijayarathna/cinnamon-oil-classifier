import { DragEvent, FormEvent, useEffect, useRef, useState } from "react";
import {
  API_BASE,
  CLASS_LABELS,
  QUALITY_LABEL,
  QUALITY_MAP,
  QualityTier,
} from "./constants";
import { PredictionResponse, ValidationErrors } from "./types";

function extractProbabilities(raw: number[] | number[][] | undefined): number[] {
  if (!raw) return [];
  if (Array.isArray(raw[0])) return raw[0] as number[];
  if (Array.isArray(raw)) return raw as number[];
  return [];
}

export default function App() {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [density, setDensity] = useState("");
  const [phValue, setPhValue] = useState("");
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [loading, setLoading] = useState(false);
  const [errors, setErrors] = useState<ValidationErrors>({
    image: false,
    density: false,
    ph: false,
  });
  const [apiError, setApiError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [apiHealthy, setApiHealthy] = useState<boolean | null>(null);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  useEffect(() => {
    async function checkHealth() {
      try {
        const res = await fetch(`${API_BASE}/health`);
        const data = await res.json();
        setApiHealthy(res.ok && data.models_loaded !== false);
      } catch {
        setApiHealthy(false);
      }
    }
    checkHealth();
  }, []);

  function setImage(file: File | null) {
    setPreviewUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return file ? URL.createObjectURL(file) : null;
    });
    setImageFile(file);
    setErrors((e) => ({ ...e, image: false }));
  }

  function handleFileChange(file: File | null) {
    if (file && !file.type.startsWith("image/")) return;
    setImage(file);
  }

  function handleDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file?.type.startsWith("image/")) {
      handleFileChange(file);
      if (fileInputRef.current) {
        const dt = new DataTransfer();
        dt.items.add(file);
        fileInputRef.current.files = dt.files;
      }
    }
  }

  function validate(): boolean {
    const next: ValidationErrors = {
      image: !imageFile,
      density: density === "" || isNaN(parseFloat(density)),
      ph: phValue === "" || isNaN(parseFloat(phValue)),
    };
    setErrors(next);
    return !next.image && !next.density && !next.ph;
  }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setApiError(null);
    if (!validate() || !imageFile) return;

    const formData = new FormData();
    formData.append("density", density);
    formData.append("ph_value", phValue);
    formData.append("image", imageFile);

    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/predict`, { method: "POST", body: formData });
      const data = await res.json();

      if (!res.ok) {
        const msg = data.details
          ? `${data.error}: ${data.details}`
          : data.error || "Prediction request failed.";
        setApiError(msg);
        setResult(null);
        return;
      }

      setResult(data as PredictionResponse);
    } catch {
      setApiError(
        `Could not reach the API. Make sure the backend is running at ${API_BASE}.`
      );
      setApiHealthy(false);
      setResult(null);
    } finally {
      setLoading(false);
    }
  }

  function handleReset() {
    setDensity("");
    setPhValue("");
    setImage(null);
    setErrors({ image: false, density: false, ph: false });
    setApiError(null);
    setResult(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  const tier: QualityTier | undefined = result
    ? QUALITY_MAP[result.prediction]
    : undefined;
  const probs = result
    ? extractProbabilities(result.raw_prediction_values)
    : [];

  return (
    <div className="card">
      <div
        className="status-indicator"
        title={apiHealthy === null ? "Checking API…" : undefined}
      >
        <span
          className={`status-dot${apiHealthy === true ? " healthy" : apiHealthy === false ? " offline" : ""}`}
        />
        <span>
          {apiHealthy === null
            ? "Checking…"
            : apiHealthy
              ? "API Online"
              : "API Offline"}
        </span>
      </div>

      <div className="card-header">
        <h1>Cinnamon Oil Classifier</h1>
        <p>Upload a sample image and enter lab measurements to classify oil quality.</p>
      </div>

      <form onSubmit={handleSubmit} noValidate>
        <div className="field">
          <label className="field-label" htmlFor="imageInput">
            Sample Image
          </label>
          <div
            className={`upload-zone${imageFile ? " has-image" : ""}${dragOver ? " dragover" : ""}`}
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleDrop}
          >
            <input
              ref={fileInputRef}
              type="file"
              id="imageInput"
              accept="image/jpeg,image/png,image/jpg"
              onChange={(e) => handleFileChange(e.target.files?.[0] ?? null)}
            />
            <svg
              className="upload-icon"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="17 8 12 3 7 8" />
              <line x1="12" y1="3" x2="12" y2="15" />
            </svg>
            <span className="upload-text">Drop image here or click to browse</span>
            <span className="upload-hint">JPG or PNG</span>
            {previewUrl && (
              <img className="upload-preview" src={previewUrl} alt="Sample preview" />
            )}
          </div>
          <p className={`field-error${errors.image ? " visible" : ""}`}>
            Please upload a sample image.
          </p>
        </div>

        <div className="input-row">
          <div className="field">
            <label className="field-label" htmlFor="density">
              Density
            </label>
            <input
              type="number"
              id="density"
              step="0.001"
              placeholder="1.042"
              value={density}
              onChange={(e) => {
                setDensity(e.target.value);
                setErrors((prev) => ({ ...prev, density: false }));
              }}
            />
            <p className={`field-error${errors.density ? " visible" : ""}`}>
              Please enter a density value.
            </p>
          </div>
          <div className="field">
            <label className="field-label" htmlFor="phValue">
              pH Value
            </label>
            <input
              type="number"
              id="phValue"
              step="0.01"
              placeholder="5.35"
              value={phValue}
              onChange={(e) => {
                setPhValue(e.target.value);
                setErrors((prev) => ({ ...prev, ph: false }));
              }}
            />
            <p className={`field-error${errors.ph ? " visible" : ""}`}>
              Please enter a pH value.
            </p>
          </div>
        </div>

        <button
          type="submit"
          className={`btn-primary${loading ? " loading" : ""}`}
          disabled={loading}
        >
          <span className="spinner" />
          <span>{loading ? "Analysing…" : "Analyse Sample"}</span>
        </button>
      </form>

      {apiError && <div className="error-section visible">{apiError}</div>}

      {result && (
        <div className="result-section visible">
          <p className="result-label">Predicted Classification</p>
          <div className="prediction-row">
            <span className="prediction-name">{result.prediction}</span>
            {tier && (
              <span className={`badge ${tier}`}>
                {QUALITY_LABEL[tier]}
              </span>
            )}
          </div>

          <div className="confidence-block">
            <div className="confidence-header">
              <span>Confidence</span>
              <span>{result.confidence_percentage.toFixed(2)}%</span>
            </div>
            <div className="confidence-bar-track">
              <div
                className="confidence-bar-fill"
                style={{ width: `${result.confidence_percentage}%` }}
              />
            </div>
          </div>

          <p className="breakdown-title">Class Probability Breakdown</p>
          <div>
            {CLASS_LABELS.map((label, i) => {
              const pct = probs[i] != null ? probs[i] * 100 : 0;
              const barTier = QUALITY_MAP[label] ?? "neutral";
              return (
                <div className="prob-row" key={label}>
                  <span className="prob-label">{label}</span>
                  <div className="prob-bar-track">
                    <div
                      className={`prob-bar-fill ${barTier}`}
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                  <span className="prob-pct">{pct.toFixed(1)}%</span>
                </div>
              );
            })}
          </div>

          <button type="button" className="btn-reset" onClick={handleReset}>
            Reset
          </button>
        </div>
      )}
    </div>
  );
}
