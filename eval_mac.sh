#!/usr/bin/env bash
# Evaluate the latest checkpoint on a small test set (macOS). Run after train_mac.sh.
set -e
cd "$(dirname "$0")"
export PYTHONPATH="$PWD"

if [ -d ".venv" ]; then
  source .venv/bin/activate
elif [ -d "venv" ]; then
  source venv/bin/activate
fi

CKPT="$(ls -t checkpoints/cxr_epoch_*.pt 2>/dev/null | head -1)"
if [ -z "$CKPT" ]; then
  echo "No checkpoint found. Run ./train_mac.sh first."
  exit 1
fi

echo "Evaluating $CKPT on small test set (max 300 samples)..."
python3 eval.py --config config_mac.yaml --checkpoint "$CKPT" --max_samples 300
