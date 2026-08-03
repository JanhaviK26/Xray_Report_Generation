# COVID-19 CXR Report Generation

Chest X-ray report generation using the **COVID-19 Radiography** dataset and **template reports**. **ResNet-18** encoder + **DistilGPT-2** decoder, with **Grad-CAM** visualization. FastAPI backend + React dashboard.

## Stack

- **Dataset**: COVID-19 Radiography (COVID, Normal, Viral Pneumonia, Lung Opacity) — template reports per class
- **Model**: ResNet-18 encoder + DistilGPT-2 decoder, PyTorch
- **Visualization**: Grad-CAM on ResNet-18
- **Eval**: BLEU-4, ROUGE-L
- **Deploy**: FastAPI backend, React (Vite) dashboard

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

**Data**: Download [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) and place it so that **`data/COVID-19_Radiography_Dataset/`** contains subfolders **COVID**, **Normal**, **Viral Pneumonia**, **Lung_Opacity** (each with an `images/` subfolder or images directly inside).

## Template-based reports (recommended)

The app can run in **template mode**: a 4-class classifier (Normal, COVID, Viral Pneumonia, Lung_Opacity) predicts the class, and the **exact template** for that class is returned. No gibberish—reports are always one of the four short, clean sentences.

**Train the classifier** (after data is in place):

```bash
python train_classifier.py --epochs 5
# Quick run: python train_classifier.py --epochs 3 --max_samples 5000
```

This saves `checkpoints/cxr_classifier.pt`. The backend **prefers** this file when present and uses template mode; otherwise it falls back to the generative model (`cxr_epoch_*.pt`).

## Training (generative model)

```bash
./train_covid19.sh
```

Or:

```bash
python train.py --config config_covid19.yaml
```

Checkpoints are saved under `checkpoints/` (e.g. `cxr_epoch_5.pt`). The backend uses `cxr_classifier.pt` if present, else the latest `cxr_epoch_*.pt`.

## Evaluation

```bash
python eval.py --config config_covid19.yaml --checkpoint checkpoints/cxr_epoch_5.pt --max_samples 500
```

## Run the app

**Terminal 1 – Backend** (from project root):

```bash
./start_backend.sh
```

**Terminal 2 – Dashboard**:

```bash
cd dashboard && npm install && npm run dev
```

Open **http://localhost:3000**. Upload a chest X-ray to get a generated report and optional Grad-CAM heatmap.

## API Endpoints

- `POST /predict` — image file → generated report (JSON `{ "report": "..." }`).
- `POST /gradcam` — image file → Grad-CAM heatmap (JSON `{ "heatmap_b64": "..." }`).
- `POST /predict_with_gradcam` — image file → report + heatmap in one response.
- `GET /health` — health check.

## Project layout

```
├── config.yaml              # COVID-19 default config
├── config_covid19.yaml      # COVID-19 training config
├── config_mac.yaml         # macOS quick train (2 epochs)
├── train_covid19.sh        # Train on COVID-19 Radiography
├── train_mac.sh            # Train on Mac (small subset)
├── start_backend.sh
├── requirements.txt
├── train.py
├── train_classifier.py   # 4-class classifier for template-based reports
├── eval.py
├── src/
│   ├── data/
│   │   └── covid19_dataset.py   # COVID-19 + template reports, CXRReportDataset
│   ├── model/
│   │   ├── encoder_decoder.py    # ResNet-18 + DistilGPT-2
│   │   └── gradcam.py
│   └── eval/
├── backend/
│   └── main.py
└── dashboard/
```

## License

COVID-19 Radiography data: see Kaggle/dataset terms. Code: use as you like.
