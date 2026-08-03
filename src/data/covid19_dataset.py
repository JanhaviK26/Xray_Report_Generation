"""COVID-19 Radiography dataset: class folders -> template reports. Small, CPU-friendly."""

# Preprocessing: folder scan -> (path, template); Dataset __getitem__ -> tensor + tokens

from pathlib import Path  # typed paths for image files
from typing import List, Tuple  # type hints for pairs list

import torch  # reserved import (some setups use it; safe for future)
from PIL import Image  # load PNG/JPG as RGB
from torch.utils.data import Dataset  # PyTorch dataset interface
from torchvision import transforms  # resize, normalize, ToTensor

# Class indices 0..3 — only for WeightedRandomSampler in train.py (not used at inference)
CLASS_ORDER = ["normal", "covid", "viral", "lung_opacity"]


def template_to_class_idx(text: str) -> int:
    """Which template (0..3) matches this exact report string."""
    text = text.strip()  # normalize whitespace
    for i, key in enumerate(CLASS_ORDER):  # scan fixed order
        if COVID19_TEMPLATES[key].strip() == text:  # exact match to a template
            return i  # found class index
    return 0  # unknown text defaults to 0 (normal slot)


# One reference paragraph per class — all images in that folder share this text
COVID19_TEMPLATES = {
    "normal": (
        "Heart size normal. Mediastinal contour unremarkable. "
        "Lungs clear. No focal consolidation, effusion, or pneumothorax."
    ),
    "covid": (
        "Bilateral opacities noted. Findings may correlate with viral pneumonitis. "
        "Clinical correlation and PCR testing recommended."
    ),
    "viral": (
        "Patchy opacities present. Findings consistent with viral pneumonia. "
        "Recommend clinical correlation."
    ),
    "lung_opacity": (
        "Lung opacity present. Nonspecific. Recommend correlation with clinical history and prior studies."
    ),
}
COVID19_TEMPLATES["covid-19"] = COVID19_TEMPLATES["covid"]  # alias folder spelling
COVID19_TEMPLATES["covid19"] = COVID19_TEMPLATES["covid"]  # alias
COVID19_TEMPLATES["viral pneumonia"] = COVID19_TEMPLATES["viral"]  # alias
COVID19_TEMPLATES["lung opacity"] = COVID19_TEMPLATES["lung_opacity"]  # alias


def _class_key(folder_name: str) -> str:
    """Infer template key from subfolder name (case-insensitive)."""
    s = folder_name.lower().strip()  # normalize folder label
    if "covid" in s:  # COVID folder
        return "covid"
    if "normal" in s:  # normal chest
        return "normal"
    if "viral" in s or ("pneumonia" in s and "viral" in s):  # viral pneumonia variants
        return "viral"
    if "lung" in s and "opacity" in s:  # lung opacity class
        return "lung_opacity"
    return s.replace(" ", "_")[:20]  # fallback truncated key


def build_covid19_pairs(root_dir: Path) -> List[Tuple[str, str]]:
    """Return [(absolute_path, template_report), ...] for every image under root_dir."""
    root_dir = Path(root_dir)  # ensure Path
    if not root_dir.exists():  # missing dataset
        return []  # nothing to train on
    pairs = []  # accumulator
    exts = {".png", ".jpg", ".jpeg"}  # allowed image extensions
    for subdir in root_dir.iterdir():  # each class folder
        if not subdir.is_dir():  # skip files at root
            continue  # next entry
        key = _class_key(subdir.name)  # map folder name -> template key
        template = COVID19_TEMPLATES.get(key)  # look up paragraph
        if template is None:  # unknown folder
            if key == "covid" and "covid" in subdir.name.lower():  # extra covid fallback
                template = COVID19_TEMPLATES["covid"]  # use covid template
            else:
                continue  # skip unrecognized folders
        search_dirs = [subdir]  # images may be here
        if (subdir / "images").is_dir():  # or in images/ subfolder
            search_dirs.append(subdir / "images")  # add second search root
        for d in search_dirs:  # scan each
            for f in d.iterdir():  # each file
                if f.suffix.lower() in exts:  # image file
                    pairs.append((str(f.resolve()), template))  # absolute path + text
    return pairs  # full list


class CXRReportDataset(Dataset):
    """PyTorch Dataset: __getitem__ returns tensor image + tokenized report."""

    def __init__(
        self,
        pairs: List[Tuple[str, str]],  # built by build_covid19_pairs
        tokenizer,  # GPT2Tokenizer (must match decoder)
        image_size: int = 224,  # H=W for ResNet input
        max_seq_len: int = 128,  # max BPE tokens per report
    ):
        self.pairs = pairs  # store all (path, text) pairs
        self.tokenizer = tokenizer  # for encoding reports
        self.max_seq_len = max_seq_len  # pad/truncate length
        self.transform = transforms.Compose([  # vision preprocessing pipeline
            transforms.Resize((image_size, image_size)),  # square resize
            transforms.ToTensor(),  # HWC [0,1] float
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # ImageNet stats
        ])
        self.class_ids = [template_to_class_idx(text) for _, text in pairs]  # 0..3 per row

    def __len__(self):
        return len(self.pairs)  # dataset size

    def __getitem__(self, idx):
        img_path, text = self.pairs[idx]  # one pair
        img = Image.open(img_path).convert("RGB")  # load; 3 channels for CNN
        img_t = self.transform(img)  # tensor (3,224,224)
        class_id = self.class_ids[idx]  # for balanced sampling only
        enc = self.tokenizer(  # BPE encode report
            text,  # raw string
            max_length=self.max_seq_len,  # cap length
            padding="max_length",  # pad to max_seq_len
            truncation=True,  # cut if too long
            return_tensors="pt",  # batch dim kept for squeeze below
        )
        input_ids = enc["input_ids"].squeeze(0)  # (max_seq_len,) long tensor
        attention_mask = enc["attention_mask"].squeeze(0)  # 1=real token, 0=pad
        return {
            "image": img_t,  # model input
            "input_ids": input_ids,  # decoder targets / teacher forcing
            "attention_mask": attention_mask,  # ignore pads in attention
            "report": text,  # reference string for metrics
            "image_path": img_path,  # debug path
            "class_id": class_id,  # class balance (train only)
        }
