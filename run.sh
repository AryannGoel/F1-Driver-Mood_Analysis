#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
python3 -m pip install --quiet fastapi "uvicorn[standard]" pydantic
echo ""
echo "  Pit Wall running -> open http://localhost:8000   (Ctrl+C to stop)"
echo ""
python3 -m uvicorn app:app --port 8000
