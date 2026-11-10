"""FastAPI page: upload a frontal chest X-ray, get a findings draft."""

from __future__ import annotations

import io
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from app.engine import MODELS, ROOT, get_engine, sample_by_id

STATIC = Path(__file__).resolve().parent / "static"
SAMPLES = ROOT / "samples"
MAX_BYTES = 12 * 1024 * 1024

app = FastAPI(title="IU-Xray report demo")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
app.mount("/samples", StaticFiles(directory=SAMPLES), name="samples")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    engine = get_engine()
    return {
        "ok": True,
        "device": str(engine.device),
        "models": {
            key: {
                "label": spec["label"],
                "run": spec["run"],
                "checkpoint_present": engine.checkpoint_ready(key),
            }
            for key, spec in MODELS.items()
        },
    }


@app.get("/api/samples")
def samples():
    from app.engine import load_samples

    rows = []
    for row in load_samples():
        rows.append(
            {
                "id": row["id"],
                "file": row["file"],
                "normal": row["normal"],
                "split": row["split"],
            }
        )
    return rows


@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    model: str = Form("norag"),
    sample_id: str = Form(""),
):
    if model not in MODELS:
        raise HTTPException(status_code=400, detail="Unknown model")
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(raw) > MAX_BYTES:
        raise HTTPException(status_code=400, detail="Image is larger than 12 MB")
    try:
        pil = Image.open(io.BytesIO(raw))
        pil.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="That file is not a readable image")

    engine = get_engine()
    try:
        result = engine.predict(pil, model)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    sample = sample_by_id(sample_id) if sample_id else None
    result["reference"] = sample["reference"] if sample else None
    result["sample_id"] = sample["id"] if sample else None
    result["sample_normal"] = sample["normal"] if sample else None
    result["disclaimer"] = (
        "Research prototype on IU-Xray. Not for diagnosis or clinical use. "
        "Test CheXbert F1 for the no-RAG models is about 0.32."
    )
    return result
