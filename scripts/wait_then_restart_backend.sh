#!/usr/bin/env bash
# Wait for cxr_epoch_5.pt to exist, then restart the backend.
set -e
CD="/Users/janhavi/Desktop/New"
CKPT="$CD/checkpoints/cxr_epoch_5.pt"
echo "Waiting for $CKPT (checking every 2 min)..."
while [ ! -f "$CKPT" ]; do
  sleep 120
done
echo "Found $CKPT. Restarting backend..."
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
sleep 2
cd "$CD" && PYTHONPATH="$CD" .venv/bin/python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 &
echo "Backend restarted on port 8000."
