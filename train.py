#!/usr/bin/env python3
"""Train encoder-decoder; data preprocessing lives in covid19_dataset.CXRReportDataset."""

import argparse  # CLI --config, --data_dir, --max_samples
import random  # (available if you add manual shuffling later)
from collections import Counter  # count samples per class for weights
from pathlib import Path  # paths to config and data

import torch  # tensors, device, save
import yaml  # read config.yaml
from torch.utils.data import DataLoader, Dataset, random_split  # loaders and split
from torchvision import transforms  # used in TrainAugmentWrapper
from transformers import GPT2Tokenizer  # must match decoder name in config

from src.data import build_covid19_pairs, CXRReportDataset  # data pipeline
from src.model import ReportGenerator  # encoder + GPT-2
from src.eval import compute_metrics  # BLEU-4, ROUGE-L


class TrainAugmentWrapper(Dataset):
    """Adds random horizontal flip only on training Subset indices."""

    def __init__(self, subset):
        self.subset = subset  # random_split Subset; .indices -> original dataset indices
        self.flip = transforms.RandomHorizontalFlip(p=0.5)  # 50% mirror CXR

    def __len__(self):
        return len(self.subset)  # same length as train split

    def __getitem__(self, idx):
        out = dict(self.subset[idx])  # copy base sample dict
        out["image"] = self.flip(out["image"])  # augment image tensor only
        return out  # batch collates this


def load_config(path: str = "config.yaml"):
    with open(path) as f:  # read YAML file
        return yaml.safe_load(f)  # nested dict


def main():
    parser = argparse.ArgumentParser()  # CLI
    parser.add_argument("--config", default="config.yaml")  # hyperparameters file
    parser.add_argument("--data_dir", default=None)  # override data root
    parser.add_argument("--max_samples", type=int, default=None, help="Max train+val+test samples (for quick run)")  # debug
    args = parser.parse_args()  # parse
    cfg = load_config(args.config)  # full config dict
    data_cfg = cfg["data"]  # paths, ratios, image size
    model_cfg = cfg["model"]  # encoder, decoder, dropout
    train_cfg = cfg["train"]  # lr, epochs, checkpoint dir
    eval_cfg = cfg["eval"]  # reserved for eval settings (metrics use defaults in code)

    data_dir = Path(args.data_dir or data_cfg["data_dir"])  # default from yaml
    covid_dir = Path(data_cfg.get("covid19_dir", data_dir))  # COVID-19 Radiography root
    pairs = build_covid19_pairs(covid_dir)  # list of (path, template text)
    if not pairs:  # empty dataset
        print("No COVID-19 images found. Put data in data/COVID-19_Radiography_Dataset/ with subdirs: COVID, Normal, Viral Pneumonia, Lung_Opacity (see README).")
        return  # exit
    print(f"Using COVID-19 Radiography: {len(pairs)} image-report pairs.")  # log size

    tokenizer = GPT2Tokenizer.from_pretrained(model_cfg["decoder"])  # load DistilGPT-2 tokenizer
    tokenizer.pad_token = tokenizer.eos_token  # padding id = eos id for batching

    dataset = CXRReportDataset(  # lazy preprocessing in __getitem__
        pairs=pairs,  # all samples
        tokenizer=tokenizer,  # encode text
        image_size=data_cfg["image_size"],  # e.g. 224
        max_seq_len=data_cfg["max_seq_len"],  # e.g. 128
    )
    n = len(dataset)  # total samples
    if args.max_samples is not None and n > args.max_samples:  # optional cap
        dataset = torch.utils.data.Subset(dataset, range(args.max_samples))  # first N only
        n = len(dataset)  # new size

    tr, val, te = data_cfg["train_ratio"], data_cfg["val_ratio"], data_cfg["test_ratio"]  # must sum to 1
    t_len = int(n * tr)  # train count
    v_len = int(n * val)  # val count
    s_len = n - t_len - v_len  # test count (remainder)
    train_ds, val_ds, test_ds = random_split(  # disjoint indices
        dataset, [t_len, v_len, s_len], generator=torch.Generator().manual_seed(42)  # reproducible
    )

    train_ds = TrainAugmentWrapper(train_ds)  # flip only train

    train_indices = train_ds.subset.indices  # list of indices into full dataset
    class_ids = [dataset.class_ids[i] for i in train_indices]  # 0..3 per train sample
    class_counts = Counter(class_ids)  # how many per class in train
    weights = [1.0 / class_counts[c] for c in class_ids]  # inverse freq weight per sample
    sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights))  # balanced draws

    train_loader = DataLoader(  # training batches
        train_ds,  # augmented train subset
        batch_size=train_cfg["batch_size"],  # e.g. 16
        sampler=sampler,  # balanced; cannot combine with shuffle=True
        num_workers=0,  # single-process loading (simple on Mac)
        pin_memory=True,  # faster host->GPU copy if CUDA
    )
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)  # fixed order for val

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU or CPU
    model = ReportGenerator(  # build nn.Module
        tokenizer=tokenizer,  # for generate() decode
        encoder_embed_dim=model_cfg["embed_dim"],  # 768 matches DistilGPT-2
        decoder_name=model_cfg["decoder"],  # HF model id
        pretrained_encoder=model_cfg["densenet_pretrained"],  # ImageNet weights flag
        dropout=model_cfg["dropout"],  # regularization
        encoder_name=model_cfg.get("encoder", "resnet18"),  # CNN backbone name
    ).to(device)  # move parameters to device
    lr = float(train_cfg["lr"])  # learning rate
    weight_decay = float(train_cfg["weight_decay"])  # L2 penalty
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)  # optimizer

    ckpt_dir = Path(train_cfg["checkpoint_dir"])  # e.g. checkpoints/
    ckpt_dir.mkdir(parents=True, exist_ok=True)  # create if missing

    best_val_bleu = -1.0  # best BLEU so far (higher is better)
    best_epoch = 0  # 1-based epoch index of best
    patience = int(train_cfg.get("early_stop_patience", 2))  # epochs without improvement
    epochs_no_improve = 0  # counter

    for epoch in range(int(train_cfg["epochs"])):  # outer epoch loop
        model.train()  # dropout on, batchnorm train mode
        total_loss = 0.0  # accumulate batch losses
        for batch in train_loader:  # one batch
            img = batch["image"].to(device)  # (B,3,224,224)
            input_ids = batch["input_ids"].to(device)  # (B, seq)
            attn = batch["attention_mask"].to(device)  # (B, seq)
            labels = input_ids.clone()  # same shape as targets for HF
            labels[labels == tokenizer.pad_token_id] = -100  # ignore pad in loss
            opt.zero_grad()  # clear grads
            out = model(image=img, input_ids=input_ids, attention_mask=attn, labels=labels)  # forward + loss
            loss = out.loss  # scalar cross-entropy (next-token)
            loss.backward()  # backprop
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["gradient_clip"])  # stability
            opt.step()  # update weights
            total_loss += loss.item()  # float for logging
        avg = total_loss / len(train_loader)  # mean train loss this epoch
        print(f"Epoch {epoch+1} train loss: {avg:.4f}", flush=True)  # stdout

        model.eval()  # dropout off for generation
        refs, hyps = [], []  # reference and hypothesis strings
        with torch.no_grad():  # no gradients during val
            for batch in list(val_loader)[:20]:  # first 20 batches only (speed)
                img = batch["image"].to(device)  # val images
                for r in batch["report"]:  # ground-truth template strings
                    refs.append(r)  # one ref per image in batch
                gen = model.generate(img, max_length=data_cfg["max_seq_len"], num_beams=2)  # beam decode
                for i in range(gen.size(0)):  # each sample in batch
                    t = tokenizer.decode(gen[i], skip_special_tokens=True)  # ids -> string
                    hyps.append(t)  # predicted text
        val_bleu = 0.0  # default if metrics fail
        if refs and hyps and len(refs) == len(hyps):  # valid pairs
            metrics = compute_metrics(refs, hyps)  # dict with BLEU-4, ROUGE-L
            val_bleu = metrics["BLEU-4"]  # scalar for early stopping
            print(f"  Val BLEU-4: {val_bleu:.4f}  ROUGE-L: {metrics['ROUGE-L']:.4f}", flush=True)  # log

        if val_bleu > best_val_bleu:  # improved
            best_val_bleu = val_bleu  # update best
            best_epoch = epoch + 1  # human-readable epoch number
            epochs_no_improve = 0  # reset patience
            torch.save(  # write best weights
                {
                    "epoch": epoch + 1,  # saved epoch
                    "model_state_dict": model.state_dict(),  # all parameters
                    "config": cfg,  # full yaml for reproducibility
                },
                ckpt_dir / "cxr_best.pt",  # filename
            )
        else:  # no improvement
            epochs_no_improve += 1  # bump counter

        if (epoch + 1) % int(train_cfg["save_every"]) == 0:  # every N epochs
            torch.save(  # snapshot even if not best
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                },
                ckpt_dir / f"cxr_epoch_{epoch+1}.pt",  # per-epoch file
            )

        if epochs_no_improve >= patience:  # early stop condition
            print(f"Early stop: no BLEU-4 improvement for {patience} epochs (best: {best_val_bleu:.4f} @ epoch {best_epoch}).", flush=True)
            break  # exit epoch loop

    print("Training done. Best checkpoint:", ckpt_dir / "cxr_best.pt", flush=True)  # final message


if __name__ == "__main__":
    main()  # entry point when run as script
