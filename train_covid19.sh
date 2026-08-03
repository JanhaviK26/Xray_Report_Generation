#!/usr/bin/env bash
# Train on COVID-19 Radiography (template reports). Run from project root.
# Download: https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
# Place so that data/covid19/ contains: COVID/, Normal/, Viral Pneumonia/
set -e
cd "$(dirname "$0")"
export PYTHONPATH="$PWD"

if [ -d ".venv" ]; then
  source .venv/bin/activate
elif [ -d "venv" ]; then
  source venv/bin/activate
fi

if [ ! -d "data/COVID-19_Radiography_Dataset" ] && [ ! -d "data/covid19" ]; then
  echo "Put COVID-19 Radiography data in data/COVID-19_Radiography_Dataset (COVID, Normal, Viral Pneumonia, Lung_Opacity)."
  echo "Download from: https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database"
  exit 1
fi

echo "Training on COVID-19 Radiography (template reports)..."
python3 train.py --config config_covid19.yaml

echo ""
echo "Done. Start backend: ./start_backend.sh   Dashboard: cd dashboard && npm run dev"
