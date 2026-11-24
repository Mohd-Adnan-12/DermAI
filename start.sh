#!/bin/bash

# DermAI Application Launcher
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "========================================="
echo "   Starting DermAI Full-Stack System     "
echo "========================================="

# 1. Check PostgreSQL
if ! pg_isready -q 2>/dev/null; then
    echo "⚠️ PostgreSQL does not seem to be responding on default socket."
    echo "Attempting to start postgresql@16 via Homebrew..."
    brew services start postgresql@16 2>/dev/null || true
    sleep 2
fi

# 2. Check virtual environment
if [ -d "$DIR/venv" ]; then
    UVICORN="$DIR/venv/bin/uvicorn"
elif [ -d "$DIR/.venv" ]; then
    UVICORN="$DIR/.venv/bin/uvicorn"
else
    echo "❌ Python virtual environment not found in ./venv or ./.venv"
    exit 1
fi

# 3. Clean up on exit
cleanup() {
    echo ""
    echo "Shutting down DermAI services..."
    kill "$BACKEND_PID" 2>/dev/null || true
    kill "$FRONTEND_PID" 2>/dev/null || true
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# 4. Start Backend
echo "Starting Backend (FastAPI on http://127.0.0.1:8000)..."
"$UVICORN" main:app --host 127.0.0.1 --port 8000 --reload &
BACKEND_PID=$!

# Wait for backend to initialize
sleep 2

# 5. Start Frontend
echo "Starting Frontend (Vite on http://localhost:5173)..."
(cd "$DIR/frontend" && npm run dev) &
FRONTEND_PID=$!

echo ""
echo "========================================="
echo "🚀 DermAI is running!"
echo "   - Web UI:   http://localhost:5173"
echo "   - API:      http://127.0.0.1:8000"
echo "   - API Docs: http://127.0.0.1:8000/docs"
echo "========================================="
echo "Press Ctrl+C to stop all servers."
echo ""

wait

