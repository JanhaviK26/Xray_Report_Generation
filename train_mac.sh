#!/usr/bin/env bash
# Train on macOS with a small dataset (quick run). Run from project root.
set -e
cd "$(dirname "$0")"
export PYTHONPATH="$PWD"

# Default: 2000 samples (CPU-friendly, not too small/large), 2 epochs. Override: TRAIN_SAMPLES=1000 ./train_mac.sh
SAMPLES="${TRAIN_SAMPLES:-2000}"

if [ -d ".venv" ]; then
  source .venv/bin/activate
elif [ -d "venv" ]; then
  source venv/bin/activate
fi

if [ ! -d "data/COVID-19_Radiography_Dataset" ] && [ ! -d "data/covid19" ]; then
  echo "Error: COVID-19 data not found. Put dataset in data/COVID-19_Radiography_Dataset/ (COVID, Normal, Viral Pneumonia, Lung_Opacity)."
  exit 1
fi

echo "Training on COVID-19 (${SAMPLES} samples, config_mac.yaml: 2 epochs, batch 8)..."
python3 train.py --config config_mac.yaml --max_samples "$SAMPLES"

echo ""
echo "Training done. Checkpoints saved in checkpoints/"
echo "Next:"
echo "  1. Start backend:  ./start_backend.sh"
echo "  2. Start dashboard:  cd dashboard && npm run dev"
echo "  3. Open http://localhost:3000 and upload an X-ray to test."
