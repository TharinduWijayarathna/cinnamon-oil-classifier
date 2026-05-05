import { ChangeEvent, FormEvent, useMemo, useState } from "react";

type PredictionResponse = {
  prediction: string;
  predicted_class_index: number;
  confidence_percentage: number;
  raw_prediction_values: number[] | number[][];
  classification_type: "binary" | "multiclass";
};

const DEFAULT_API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

export default function App() {
  const [apiBase, setApiBase] = useState(DEFAULT_API_BASE);
  const [oilMass, setOilMass] = useState("12.5");
  const [density, setDensity] = useState("0.98");
  const [phValue, setPhValue] = useState("6.2");
  const [oilType, setOilType] = useState("organic");
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);

  const imageName = useMemo(() => imageFile?.name ?? "No file selected", [imageFile]);

  const onImageChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0] ?? null;
    setImageFile(file);
  };

  const onSubmit = async (event: FormEvent) => {
    event.preventDefault();
    setError(null);
    setResult(null);

    if (!imageFile) {
      setError("Please upload an image.");
      return;
    }

    const formData = new FormData();
    formData.append("oil_mass", oilMass);
    formData.append("density", density);
    formData.append("ph_value", phValue);
    formData.append("oil_type", oilType);
    formData.append("image", imageFile);

    setLoading(true);
    try {
      const response = await fetch(`${apiBase}/predict`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data?.details || data?.error || "Prediction failed.");
      }

      setResult(data as PredictionResponse);
    } catch (submitError) {
      const message =
        submitError instanceof Error ? submitError.message : "Unexpected request error.";
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="container">
      <h1>Cinnamon Oil Purity Checker</h1>
      <p className="subtitle">TypeScript UI for testing the Flask hybrid model API.</p>

      <form className="card" onSubmit={onSubmit}>
        <label>
          API Base URL
          <input value={apiBase} onChange={(e) => setApiBase(e.target.value)} />
        </label>

        <label>
          Oil Mass
          <input type="number" step="any" value={oilMass} onChange={(e) => setOilMass(e.target.value)} />
        </label>

        <label>
          Density
          <input type="number" step="any" value={density} onChange={(e) => setDensity(e.target.value)} />
        </label>

        <label>
          pH Value
          <input type="number" step="any" value={phValue} onChange={(e) => setPhValue(e.target.value)} />
        </label>

        <label>
          Oil Type
          <input value={oilType} onChange={(e) => setOilType(e.target.value)} />
        </label>

        <label>
          Image File
          <input type="file" accept="image/*" onChange={onImageChange} />
        </label>

        <p className="file-name">{imageName}</p>

        <button type="submit" disabled={loading}>
          {loading ? "Predicting..." : "Predict Purity"}
        </button>
      </form>

      {error && <section className="card error">{error}</section>}

      {result && (
        <section className="card result">
          <h2>Prediction Result</h2>
          <p>
            <strong>Label:</strong> {result.prediction}
          </p>
          <p>
            <strong>Confidence:</strong> {result.confidence_percentage}%
          </p>
          <p>
            <strong>Class Index:</strong> {result.predicted_class_index}
          </p>
          <p>
            <strong>Type:</strong> {result.classification_type}
          </p>
          <pre>{JSON.stringify(result.raw_prediction_values, null, 2)}</pre>
        </section>
      )}
    </main>
  );
}
