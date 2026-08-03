#!/bin/bash
# Run from project root: ./start_backend.sh
cd "$(dirname "$0")"
export PYTHONPATH="$PWD"

# Use venv if present
if [ -d ".venv" ]; then
  source .venv/bin/activate
elif [ -d "venv" ]; then
  source venv/bin/activate
fi

if ! python3 -c "import uvicorn" 2>/dev/null; then
  echo "Error: uvicorn not installed. Install backend deps first:"
  echo "  python3 -m venv .venv"
  echo "  source .venv/bin/activate   # or: .venv\\Scripts\\activate on Windows"
  echo "  pip install -r requirements.txt"
  exit 1
fi

echo "Starting API on http://localhost:8000 ..."
echo "Dashboard: run 'cd dashboard && npm run dev' and open http://localhost:3000"
exec python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
