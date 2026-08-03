"""FastAPI backend: COVID-19 CXR report generation + Grad-CAM."""

import base64
import io
from pathlib import Path

import torch
import uvicorn
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision import transforms
from transformers import GPT2Tokenizer

# Add project root
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import ReportGenerator, GradCAM


def load_config():
    p = Path(__file__).resolve().parent.parent / "config.yaml"
    with open(p) as f:
        return yaml.safe_load(f)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


cfg = load_config()
device = get_device()


def post_process_report(text: str) -> str:
    """Clean up generated report: spacing, truncate to short template length, fix glues."""
    if not text or not text.strip():
        return text
    import re
    text = text.strip()
    # Remove recurring bad opening fragment (COVID/Lung reports)
    text = re.sub(r"^\s*Ung\s+present\s+Npe\.?\s*", "", text, flags=re.IGNORECASE)
    text = text.strip()
    # COVID-19 reports are 1–2 short sentences; keep only first 2 sentences
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) > 2:
        text = " ".join(sentences[:2]).strip()
    if not text:
        return text
    # Fix common BPE glues that produce gibberish (add space so words separate)
    glue_fixes = [
        (r"linalourarkable", "unremarkable"),
        (r"Opities", "Opacities"),
        (r"\bIngs\b", "Findings"),
        (r"viralon", "viral"),
        (r"Onscific", "Nonspecific"),
        (r"pneumitis", "pneumonitis"),
        (r"No consolidation eff[,.]", "No consolidation, effusion."),
        (r"\beff[,.]", "effusion."),
        (r"^Size\.\s+", "Heart size normal. "),
        (r"Noothax", "No pleural"),
        (r"cardomedast", "cardiac mediastinal"),
        (r"pulmonarycul", "pulmonary"),
        (r"mediinalour", "mediastinal contour"),
        (r"pleuraleff", "pleural effusion"),
        (r"Lungarensisin", "Lung aeration is in"),
        (r"Cardomealy", "Cardiomegaly"),
        (r"vascularification", "vascular marking"),
        (r"athecler", "atherosclerotic"),
        (r"acuteiopulmonary", "acute pulmonary"),
        (r"orothax", "or thoracic"),
        (r"granoma", "granuloma"),
        (r"nouralusion", "no consolidation"),
        (r"oruralusion", "or consolidation"),
        (r"pneumor", "pneumothorax"),
        (r"hilaren", "hilar"),
        (r"atasis", "atelectasis"),
        (r"pulmonaryity theort", "pulmonary, thorax"),
        (r"upperateelect", "upper lobe atelectasis"),
        (r"versusphymatous", "versus emphysematous"),
        (r"arearkable", "are remarkable"),
        (r"chronicstitialema", "chronic interstitial"),
        (r"Mildativeiosis", "Mild atelectasis"),
        (r"thorolarifications", "thoracic calcifications"),
        (r"previousul exam", "previous exam"),
        (r"bronov crowd", "bronchovascular crowding"),
        (r"airspaceac", "airspace"),
        (r"St appearance", "Soft tissue appearance"),
        (r"\.able ", ". Able "),
        (r"mildu ", "mild "),
        (r"aicotomy", "history"),
        (r"Visual osus", "Visual assessment"),
        (r"ple eff", "pleural effusion"),
        (r"orusions", "effusions"),
        (r"heart andastum", "heart and mediastinum"),
        (r"lobeyibilarelect", "lobe bilateral atelectasis"),
        (r"versusasis", "versus"),
        (r"pulmonaryity", "pulmonary"),
        (r"thanifiedulotomy", "status"),
        (r"Leftlyrexanded", "Left lung expanded"),
        (r"aticalement", "segment"),
        (r"bilateralophicchascularing", "bilateral vascular"),
        (r"Normalizedstitialema", "No interstitial edema"),
        (r"acuteiopmonaryality", "acute pulmonary"),
        (r"emsealringent", "evidence"),
        (r"cardomedar", "cardiac mediastinal"),
        (r"uralusion", "consolidation"),
        (r"theulailaterally", "bilaterally"),
        (r"yearold", "year old"),
        (r"leftsidedac", "left sided"),
        (r"St cardomedar", "Soft tissue cardiac mediastinal"),
        (r"Mildative ", "Mild atelectasis "),
    ]
    for pattern, replacement in glue_fixes:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    # Remove stray punctuation glued in words (e.g. are!"s -> are. )
    text = re.sub(r'([a-z])["\'!&]+\s*([a-z])', r"\1 \2", text)
    text = re.sub(r"\)([a-zA-Z])", r") \1", text)
    text = re.sub(r"\(([a-zA-Z])", r"( \1", text)
    # Remove placeholder tokens and stray symbols (# $ % mid-text)
    text = re.sub(r"#\s*X+\s*", " ", text)
    text = re.sub(r"\s+[#$%]\s+", " ", text)
    text = re.sub(r"[#$%](?=\S)", " ", text)
    # Ensure space after punctuation
    text = re.sub(r"([.,;:])([^\s])", r"\1 \2", text)
    # Space between lowercase followed by uppercase (fixes run-ons like "normalThere")
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    # Capitalize first letter of each sentence
    text = re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)
    # Collapse multiple spaces
    text = re.sub(r" +", " ", text)
    # Final trim: at most 2 sentences again after cleanup
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if len(sentences) > 2:
        text = " ".join(sentences[:2]).strip()
    # Truncate at last proper sentence end; drop trailing fragment (often gibberish)
    last_period = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
    if last_period > 10:
        text = text[: last_period + 1].strip()
    # Drop words that look like gibberish: very long, digits mixed with letters, 4+ same char
    junk_tokens = {"ung", "npe", "llateralac", "cloningon", "lateralac"}
    words = text.split()
    kept = []
    for w in words:
        if w.lower() in junk_tokens:
            continue
        if len(w) > 22:
            continue
        if re.search(r"[a-zA-Z].*\d|\d.*[a-zA-Z]", w):
            continue
        if re.search(r"(.)\1{3,}", w):
            continue
        kept.append(w)
    text = " ".join(kept)
    return text.strip()


image_size = cfg["data"]["image_size"]
max_seq_len = cfg["data"]["max_seq_len"]
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

model = None
tokenizer = None
gradcam = None
_load_error = None
_checkpoint_loaded = False
_loaded_checkpoint_path = None


def _ensure_model():
    global model, tokenizer, gradcam, _load_error, _checkpoint_loaded, _loaded_checkpoint_path
    if _load_error:
        raise RuntimeError(_load_error)
    if model is not None:
        return
    project_root = Path(__file__).resolve().parent.parent
    ckpt_dir = project_root / cfg.get("train", {}).get("checkpoint_dir", "checkpoints")
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(cfg["model"]["decoder"])
        tokenizer.pad_token = tokenizer.eos_token
        model = ReportGenerator(
            tokenizer=tokenizer,
            encoder_embed_dim=cfg["model"]["embed_dim"],
            decoder_name=cfg["model"]["decoder"],
            pretrained_encoder=cfg["model"]["densenet_pretrained"],
            dropout=cfg["model"]["dropout"],
            encoder_name=cfg["model"].get("encoder", "resnet18"),
        ).to(device)
        best_ckpt = ckpt_dir / "cxr_best.pt"
        if best_ckpt.exists():
            ckpt_path = best_ckpt
        else:
            candidates = list(ckpt_dir.glob("cxr_epoch_*.pt"))
            ckpt_path = max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None
        if ckpt_path is not None and ckpt_path.exists():
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state["model_state_dict"], strict=False)
            _checkpoint_loaded = True
            _loaded_checkpoint_path = ckpt_path.name
            print("Loaded checkpoint:", ckpt_path.name)
        else:
            _loaded_checkpoint_path = None
            print("No checkpoint found; using untrained model for demo.")
        model.eval()
        gradcam = GradCAM(model.encoder)
    except Exception as e:
        import traceback
        _load_error = traceback.format_exc()
        print("Model load failed:", _load_error)
        raise

# Load model at startup (so first request gets the same error as before if something fails)
try:
    _ensure_model()
except Exception:
    pass  # _load_error set; endpoints will raise it on first use

app = FastAPI(title="COVID-19 CXR Report Generation API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception(request, exc):
    from fastapi import HTTPException as FastAPIHTTPException
    if isinstance(exc, FastAPIHTTPException):
        raise exc
    import traceback
    tb = traceback.format_exc()
    print("Unhandled exception:", tb)
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "error": str(exc)},
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "trained": _checkpoint_loaded,
        "checkpoint": _loaded_checkpoint_path,
    }


def _predict_report_from_image(img_t: torch.Tensor) -> str:
    """Generate report from image using encoder-decoder (GenAI)."""
    with torch.no_grad():
        rep_penalty = cfg["api"].get("repetition_penalty", 2.5)
        temp = cfg["api"].get("temperature", 1.0)
        top_p = cfg["api"].get("top_p", 1.0)
        top_k = cfg["api"].get("top_k", 50)
        max_gen = cfg["api"].get("max_gen_tokens", 48)
        no_repeat = cfg["api"].get("no_repeat_ngram_size", 4)
        num_beams = int(cfg["api"].get("num_beams", 3))
        length_penalty = float(cfg["api"].get("length_penalty", 1.0))
        gen = model.generate(
            img_t,
            max_length=max_gen,
            num_beams=num_beams,
            length_penalty=length_penalty,
            repetition_penalty=rep_penalty,
            temperature=temp,
            top_p=top_p,
            top_k=top_k,
            no_repeat_ngram_size=no_repeat,
        )
    report = tokenizer.decode(gen[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return post_process_report(report.strip())


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    _ensure_model()
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(400, "Upload an image file.")
    contents = await file.read()
    if len(contents) > (cfg["api"].get("max_file_size_mb", 10) * 1024 * 1024):
        raise HTTPException(400, "File too large.")
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")
    img_t = transform(img).unsqueeze(0).to(device)
    report = _predict_report_from_image(img_t)
    return {"report": report, "trained": _checkpoint_loaded}


@app.post("/gradcam")
async def gradcam_endpoint(file: UploadFile = File(...)):
    _ensure_model()
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(400, "Upload an image file.")
    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")
    img_t = transform(img).unsqueeze(0).to(device)
    img_t.requires_grad_(True)
    heatmap = gradcam.generate_heatmap(img_t, size=(image_size, image_size))
    buf = io.BytesIO()
    from PIL import Image as PILImage
    PILImage.fromarray(heatmap).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return {"heatmap_b64": b64, "format": "png"}


@app.post("/predict_with_gradcam")
async def predict_with_gradcam(file: UploadFile = File(...)):
    _ensure_model()
    try:
        contents = await file.read()
    except Exception as e:
        raise HTTPException(400, f"Failed to read file: {e}")
    if not contents:
        raise HTTPException(400, "Empty file.")
    # Allow missing content_type (some clients omit it)
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(400, "Upload an image file.")
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")

    # Report generation (required)
    img_t = transform(img).unsqueeze(0).to(device)
    try:
        report = _predict_report_from_image(img_t)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print("Report generation failed:", tb)
        raise HTTPException(500, f"Report generation failed: {e}")

    # Grad-CAM (optional): if it fails, still return report without heatmap
    heatmap_b64 = None
    try:
        img_t_grad = transform(img).unsqueeze(0).to(device)
        img_t_grad.requires_grad_(True)
        heatmap = gradcam.generate_heatmap(img_t_grad, size=(image_size, image_size))
        buf = io.BytesIO()
        from PIL import Image as PILImage
        PILImage.fromarray(heatmap).save(buf, format="PNG")
        heatmap_b64 = base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        import traceback
        print("Grad-CAM failed (report still returned):", traceback.format_exc())

    return {"report": report, "heatmap_b64": heatmap_b64, "trained": _checkpoint_loaded}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=cfg["api"].get("host", "0.0.0.0"),
        port=cfg["api"].get("port", 8000),
        reload=False,
    )
