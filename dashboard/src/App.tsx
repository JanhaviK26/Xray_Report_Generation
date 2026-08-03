import { useCallback, useEffect, useState } from "react";
import "./App.css";

const API_BASE = "/api";

type Result = {
  report: string;
  heatmap_b64?: string;
  trained?: boolean;
} | null;

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<Result>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [apiReachable, setApiReachable] = useState<boolean | null>(null);

  useEffect(() => {
    fetch(`${API_BASE}/health`)
      .then((r) => r.json())
      .then((d) => setApiReachable(d?.status === "ok"))
      .catch(() => setApiReachable(false));
  }, []);

  const onFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    setFile(f ?? null);
    setResult(null);
    setError(null);
    if (f) {
      const url = URL.createObjectURL(f);
      setPreview(url);
      return () => URL.revokeObjectURL(url);
    } else {
      setPreview(null);
    }
  }, []);

  const callApi = useCallback(async (endpoint: string) => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    const form = new FormData();
    form.append("file", file);
    try {
      const res = await fetch(`${API_BASE}${endpoint}`, { method: "POST", body: form });
      const t = await res.text();
      if (!res.ok) {
        let msg = t || res.statusText;
        try {
          const j = JSON.parse(t);
          if (j.detail) msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
        } catch (_) {}
        throw new Error(msg);
      }
      const data = JSON.parse(t);
      if (endpoint === "/predict") {
        setResult({ report: data.report, heatmap_b64: undefined, trained: data.trained });
      } else {
        setResult({ report: data.report, heatmap_b64: data.heatmap_b64, trained: data.trained });
      }
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Request failed";
      if (
        msg === "Failed to fetch" ||
        msg.includes("NetworkError") ||
        msg.includes("Load failed") ||
        msg.includes("connection") ||
        msg.includes("refused")
      ) {
        setError(
          "Backend not running. Open a terminal, go to the project folder, and run: ./start_backend.sh  (Then start the dashboard with: cd dashboard && npm run dev)"
        );
      } else {
        setError(msg);
      }
    } finally {
      setLoading(false);
    }
  }, [file]);

  return (
    <div className="layout">
      {apiReachable === false && (
        <div className="banner">
          Backend not running. In a terminal run: <code>./start_backend.sh</code> from the project folder (port 8000).
        </div>
      )}
      <header className="header">
        <h1>COVID-19 CXR Report Generation</h1>
        <p className="subtitle">ResNet-18 + DistilGPT-2 · Grad-CAM · BLEU-4 / ROUGE-L</p>
      </header>

      <main className="main">
        <section className="upload-section">
          <label className="upload-label">
            <input
              type="file"
              accept="image/*"
              onChange={onFileChange}
              className="upload-input"
            />
            <span className="upload-box">
              {preview ? (
                <img src={preview} alt="Preview" className="preview-img" />
              ) : (
                <>Drop chest X-ray or click to select</>
              )}
            </span>
          </label>
          <div className="btn-row">
            <button
              className="btn primary"
              onClick={() => callApi("/predict")}
              disabled={!file || loading}
            >
              {loading ? "…" : "Report only"}
            </button>
            <button
              className="btn primary"
              onClick={() => callApi("/predict_with_gradcam")}
              disabled={!file || loading}
            >
              {loading ? "…" : "Report + Grad-CAM"}
            </button>
          </div>
        </section>

        {error && <div className="error">{error}</div>}

        {result && (
          <section className="result-section">
            {result.trained === false && (
              <div className="banner demo-banner">
                Demo mode — model not trained yet. Run <code>./train_covid19.sh</code> then restart the backend.
              </div>
            )}
            <div className="result-grid">
              <div className="card">
                <h3>Generated report</h3>
                <div className="report-text">{result.report || "—"}</div>
              </div>
              {result.heatmap_b64 && (
                <div className="card">
                  <h3>Grad-CAM attention</h3>
                  <img
                    src={`data:image/png;base64,${result.heatmap_b64}`}
                    alt="Grad-CAM heatmap"
                    className="heatmap-img"
                  />
                </div>
              )}
            </div>
            {preview && (
              <div className="card">
                <h3>Input image</h3>
                <img src={preview} alt="Input" className="preview-result" />
              </div>
            )}
          </section>
        )}
      </main>

      <footer className="footer">
        COVID-19 Radiography · ResNet-18 + DistilGPT-2 · FastAPI + React
      </footer>
    </div>
  );
}
