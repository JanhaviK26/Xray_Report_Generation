"""Load the IU-Xray index and run the notebook's upload demo."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import GPT2TokenizerFast

from app.model import RagCaptioner

ROOT = Path(__file__).resolve().parent.parent
STUDIES = ROOT / "data" / "iu_xray" / "Kaggle" / "studies.csv"
POOLED = ROOT / "data" / "iu_xray" / "Kaggle" / "feat_pooled.npy"
SAMPLES = ROOT / "samples" / "manifest.json"
DRIVE_CKPT = "MyDrive/cxr_report_gen/checkpoints"

CTX_TOKENS = 60
K_RAG = 3

# Seed of each family with the highest validation CheXbert F1.
MODELS = {
    "norag": {
        "label": "No RAG",
        "run": "norag_s1",
        "k": 0,
        "beams": 1,
        "ngram": 4,
        "file": "norag_s1.pt",
    },
    "rag": {
        "label": "RAG, 3 retrieved reports",
        "run": "rag3_s3",
        "k": 3,
        "beams": 1,
        "ngram": 4,
        "file": "rag3_s3.pt",
    },
}


def checkpoint_dirs() -> list[Path]:
    """Places the app may read weights from. Nothing here is copied into the repo."""
    found: list[Path] = []
    env = os.environ.get("CXR_CKPT_DIR", "").strip()
    if env:
        found.append(Path(env).expanduser())
    found.append(ROOT / "checkpoints")
    cloud = Path.home() / "Library" / "CloudStorage"
    if cloud.is_dir():
        for drive in sorted(cloud.glob("GoogleDrive-*")):
            for folder in ("My Drive", "MyDrive"):
                found.append(drive / folder / "cxr_report_gen" / "checkpoints")
    found.append(Path.home() / "Google Drive" / "My Drive" / "cxr_report_gen" / "checkpoints")
    return found


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class Neighbor:
    rank: int
    uid: int
    similarity: float
    report: str


class Engine:
    def __init__(self):
        self.device = pick_device()
        self.df: pd.DataFrame | None = None
        self.tr_idx: np.ndarray | None = None
        self.pn_train: np.ndarray | None = None
        self.enc_body = None
        self.enc_top = None
        self.crop = None
        self.tok = None
        self.eos = None
        self.sep = None
        self.prompt = None
        self.generators: dict[str, RagCaptioner] = {}
        self._index_ready = False

    def checkpoint_path(self, key: str) -> Path | None:
        name = MODELS[key]["file"]
        for folder in checkpoint_dirs():
            path = folder / name
            if path.is_file():
                return path
        return None

    def checkpoint_ready(self, key: str) -> bool:
        return self.checkpoint_path(key) is not None

    def ensure_index(self):
        if self._index_ready:
            return
        if not STUDIES.is_file() or not POOLED.is_file():
            raise FileNotFoundError(
                "Missing data/iu_xray/Kaggle/studies.csv or feat_pooled.npy"
            )
        import torchxrayvision as xrv

        df = pd.read_csv(STUDIES)
        pooled = np.load(POOLED).astype(np.float32)
        if len(df) != len(pooled):
            raise RuntimeError(
                f"studies.csv has {len(df)} rows but feat_pooled.npy has {len(pooled)}"
            )
        train = df["split"].eq("train").to_numpy()
        tr_idx = np.flatnonzero(train)
        pn = pooled[tr_idx]
        pn = pn / np.linalg.norm(pn, axis=1, keepdims=True)

        enc = xrv.models.DenseNet(weights="densenet121-res224-all").to(self.device).eval()
        self.enc_body = enc.features[:-2]
        self.enc_top = enc.features[-2:]
        self.crop = xrv.datasets.XRayCenterCrop()
        self.df = df
        self.tr_idx = tr_idx
        self.pn_train = pn
        self._index_ready = True

    def ensure_tokenizer(self):
        if self.tok is not None:
            return
        self.tok = GPT2TokenizerFast.from_pretrained("distilgpt2")
        self.eos = self.tok.eos_token_id
        self.sep = self.tok(" |").input_ids
        self.prompt = self.tok(" findings:").input_ids

    def ensure_generator(self, key: str) -> RagCaptioner | None:
        if key in self.generators:
            return self.generators[key]
        path = self.checkpoint_path(key)
        if path is None:
            return None
        self.ensure_tokenizer()
        spec = MODELS[key]
        model = RagCaptioner(spec["k"], finetune_enc=True)
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.to(self.device).eval()
        self.generators[key] = model
        return model

    def preprocess(self, pil: Image.Image) -> torch.Tensor:
        self.ensure_index()
        a = np.array(pil).astype(np.float32)
        if a.ndim == 3:
            a = a[..., :3].mean(-1)
        if a.max() > 255:
            a = a / a.max() * 255
        import torchxrayvision as xrv

        a = self.crop(xrv.datasets.normalize(a, 255)[None])
        x = torch.from_numpy(np.ascontiguousarray(a))[None]
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return x.to(self.device)

    @torch.no_grad()
    def embed(self, x: torch.Tensor):
        mid = self.enc_body(x)
        pooled = F.relu(self.enc_top(mid)).mean((2, 3))[0].float().cpu().numpy()
        pooled = pooled / np.linalg.norm(pooled)
        return mid, pooled

    def neighbors(self, pooled: np.ndarray, k: int = K_RAG) -> list[Neighbor]:
        sims = self.pn_train @ pooled
        pos = np.argsort(-sims)[:k]
        rows = []
        for rank, p in enumerate(pos, start=1):
            i = int(self.tr_idx[int(p)])
            rows.append(
                Neighbor(
                    rank=rank,
                    uid=int(self.df.uid.iloc[i]),
                    similarity=float(sims[int(p)]),
                    report=str(self.df.target.iloc[i]),
                )
            )
        return rows

    def context(self, neigh: list[Neighbor]) -> torch.Tensor:
        self.ensure_tokenizer()
        ctx: list[int] = []
        uid_to_i = {int(u): i for i, u in enumerate(self.df.uid.tolist())}
        for nb in neigh:
            i = uid_to_i[nb.uid]
            ids = self.tok(str(self.df.target.iloc[i])).input_ids[:CTX_TOKENS]
            ctx += ids + [self.eos] * (CTX_TOKENS - len(ids)) + self.sep
        return torch.tensor([ctx + self.prompt], device=self.device)

    @torch.no_grad()
    def generate(self, key: str, mid: torch.Tensor, neigh: list[Neighbor]) -> str | None:
        model = self.ensure_generator(key)
        if model is None:
            return None
        spec = MODELS[key]
        ctx = self.context(neigh) if spec["k"] else None
        out = model.generate(
            mid.float(),
            ctx,
            eos_id=self.eos,
            beams=spec["beams"],
            ngram=spec["ngram"],
        )
        return self.tok.decode(out[0], skip_special_tokens=True).strip()

    def predict(self, pil: Image.Image, key: str) -> dict:
        if key not in MODELS:
            raise KeyError(key)
        self.ensure_index()
        x = self.preprocess(pil)
        mid, pooled = self.embed(x)
        neigh = self.neighbors(pooled)
        spec = MODELS[key]
        report = self.generate(key, mid, neigh)
        return {
            "model": key,
            "model_label": spec["label"],
            "run": spec["run"],
            "beams": spec["beams"],
            "ngram": spec["ngram"],
            "report": report,
            "report_ready": report is not None,
            "checkpoint": str(self.checkpoint_path(key) or f"{DRIVE_CKPT}/{spec['file']}"),
            "checkpoint_present": self.checkpoint_ready(key),
            "weights_note": (
                None
                if report is not None
                else (
                    "The trained weights stay on Google Drive "
                    f"({DRIVE_CKPT}/{spec['file']}) because that file is too large "
                    "to keep in this project. Closest training reports are below."
                )
            ),
            "neighbors": [
                {
                    "rank": n.rank,
                    "uid": n.uid,
                    "similarity": round(n.similarity, 3),
                    "report": n.report,
                }
                for n in neigh
            ],
        }


def load_samples() -> list[dict]:
    if not SAMPLES.is_file():
        return []
    return json.loads(SAMPLES.read_text())


def sample_by_id(sample_id: str) -> dict | None:
    for row in load_samples():
        if str(row["id"]) == str(sample_id):
            return row
    return None


_ENGINE: Engine | None = None


def get_engine() -> Engine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = Engine()
    return _ENGINE
