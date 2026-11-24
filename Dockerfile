# Multi-stage Dockerfile for DermAI Full-Stack Platform

# Stage 1: Build the React + Vite frontend
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci --silent
COPY frontend/ ./
RUN npm run build

# Stage 2: Python production runtime
FROM python:3.11-slim
WORKDIR /app

# Install system dependencies for OpenCV and image operations
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements (using CPU-optimized PyTorch wheels for fast, lightweight cloud builds)
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code and ML artifacts
COPY main.py schema.sql dermai_setup.sql ./
COPY DermAI_Artifacts/ ./DermAI_Artifacts/
COPY uploads/ ./uploads/

# Copy built frontend assets from Stage 1
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Expose port (default 8000, compatible with cloud $PORT variable)
EXPOSE 8000

ENV PYTHONUNBUFFERED=1

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]

