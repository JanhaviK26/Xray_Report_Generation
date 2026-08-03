#!/usr/bin/env python3
"""Evaluate trained model: BLEU-4 and ROUGE-L on test set."""

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer

from src.data import build_covid19_pairs, CXRReportDataset
from src.model import ReportGenerator
from src.eval import compute_metrics


def load_config(path: str = "config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to cxr_epoch_*.pt")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--max_samples", type=int, default=500)
    args = parser.parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

    data_dir = Path(args.data_dir or data_cfg["data_dir"])
    covid_dir = Path(data_cfg.get("covid19_dir", data_dir))
    pairs = build_covid19_pairs(covid_dir)
    if not pairs:
        print("No COVID-19 data. Put data in data/COVID-19_Radiography_Dataset/ (COVID, Normal, Viral Pneumonia, Lung_Opacity).")
        return

    tokenizer = GPT2Tokenizer.from_pretrained(model_cfg["decoder"])
    tokenizer.pad_token = tokenizer.eos_token
    dataset = CXRReportDataset(
        pairs=pairs,
        tokenizer=tokenizer,
        image_size=data_cfg["image_size"],
        max_seq_len=data_cfg["max_seq_len"],
    )
    n = len(dataset)
    t_len = int(n * data_cfg["train_ratio"])
    v_len = int(n * data_cfg["val_ratio"])
    s_len = n - t_len - v_len
    _, _, test_ds = torch.utils.data.random_split(
        dataset, [t_len, v_len, s_len], generator=torch.Generator().manual_seed(42)
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=8,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ReportGenerator(
        tokenizer=tokenizer,
        encoder_embed_dim=model_cfg["embed_dim"],
        decoder_name=model_cfg["decoder"],
        pretrained_encoder=False,
        dropout=model_cfg["dropout"],
        encoder_name=model_cfg.get("encoder", "resnet18"),
    )
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model = model.to(device)
    model.eval()

    references = []
    hypotheses = []
    count = 0
    with torch.no_grad():
        for batch in test_loader:
            if count >= args.max_samples:
                break
            img = batch["image"].to(device)
            for r in batch["report"]:
                references.append(r)
            gen = model.generate(img, max_length=data_cfg["max_seq_len"], num_beams=4)
            for i in range(gen.size(0)):
                if count >= args.max_samples:
                    break
                t = tokenizer.decode(gen[i], skip_special_tokens=True)
                hypotheses.append(t)
                count += 1

    if len(references) != len(hypotheses):
        hypotheses = hypotheses[: len(references)]
    metrics = compute_metrics(references, hypotheses)
    print("Evaluation (BLEU-4, ROUGE-L):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
