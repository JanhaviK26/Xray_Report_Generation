#!/usr/bin/env python3
"""Train 4-class CXR classifier for template-based reports. Saves checkpoints/cxr_classifier.pt."""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from src.model.cxr_classifier import CXRClassifier


def get_class_to_idx(root_dir: Path):
    """Scan root_dir for class folders; return class_to_idx and class_names in fixed order for template mapping."""
    root_dir = Path(root_dir)
    if not root_dir.exists():
        return {}, []
    # Fixed order: index 0=covid, 1=lung_opacity, 2=normal, 3=viral (matches COVID19_TEMPLATES keys)
    canonical = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]
    dirs = {d.name: d for d in root_dir.iterdir() if d.is_dir() and not d.name.startswith(".")}
    def match(name):
        s = name.lower()
        if "covid" in s and "lung" not in s: return "COVID"
        if "lung" in s and "opacity" in s: return "Lung_Opacity"
        if "normal" in s: return "Normal"
        if "viral" in s or "pneumonia" in s: return "Viral Pneumonia"
        return None
    class_to_idx = {}
    class_names = []
    for want in canonical:
        for folder_name, d in dirs.items():
            if match(folder_name) == want:
                class_to_idx[folder_name] = len(class_names)
                class_names.append(folder_name)
                break
    return class_to_idx, class_names


class COVID19ImageDataset(Dataset):
    """Load images from COVID-19 folder structure with class labels."""

    def __init__(self, root_dir: Path, class_to_idx: dict, image_size: int = 224, transform=None):
        self.root_dir = Path(root_dir)
        self.class_to_idx = class_to_idx
        self.image_size = image_size
        self.transform = transform or transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.samples = []  # (path, class_idx)
        exts = {".png", ".jpg", ".jpeg"}
        for class_name, idx in class_to_idx.items():
            folder = self.root_dir / class_name
            dirs_to_scan = [folder]
            if (folder / "images").is_dir():
                dirs_to_scan.append(folder / "images")
            for d in dirs_to_scan:
                if not d.exists():
                    continue
                for f in d.iterdir():
                    if f.suffix.lower() in exts:
                        self.samples.append((str(f.resolve()), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=None, help="Cap total samples (for quick run)")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    data_cfg = cfg["data"]
    root = Path(args.data_dir or data_cfg.get("covid19_dir", "data/COVID-19_Radiography_Dataset"))
    if not root.exists():
        root = Path(data_cfg.get("data_dir", "data")) / "COVID-19_Radiography_Dataset"
    if not root.exists():
        print("COVID-19 data not found at", root)
        return

    class_to_idx, class_names = get_class_to_idx(root)
    if len(class_to_idx) != 4:
        print("Expected 4 classes (COVID, Lung_Opacity, Normal, Viral Pneumonia). Found:", list(class_to_idx.keys()))
        return
    print("Classes:", class_names)

    image_size = data_cfg.get("image_size", 224)
    dataset = COVID19ImageDataset(root, class_to_idx, image_size=image_size)
    n = len(dataset)
    if args.max_samples and n > args.max_samples:
        dataset = torch.utils.data.Subset(dataset, range(args.max_samples))
        n = len(dataset)
    print("Total samples:", n)

    train_size = int(0.8 * n)
    val_size = n - train_size
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CXRClassifier(num_classes=4, pretrained=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    ckpt_dir = Path(cfg.get("train", {}).get("checkpoint_dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            opt.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        train_loss = total_loss / len(train_loader)
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                pred = model(images).argmax(dim=1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total if total else 0
        print(f"Epoch {epoch+1}  train_loss: {train_loss:.4f}  val_acc: {val_acc:.4f}")

    ckpt_path = ckpt_dir / "cxr_classifier.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "class_to_idx": class_to_idx,
        "image_size": image_size,
    }, ckpt_path)
    print("Saved", ckpt_path)


if __name__ == "__main__":
    main()
