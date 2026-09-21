FROM node:20-alpine AS frontend-build
WORKDIR /frontend
COPY frontend/package.json ./
RUN npm install --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app ARTIFACT_ROOT=/app/artifacts
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/app ./app
COPY scripts ./scripts
COPY data/metadata ./data/metadata
COPY --from=frontend-build /frontend/dist ./frontend/dist
RUN python scripts/download_datasets.py
CMD ["sh", "-c", "python scripts/load_datasets.py --only-missing && exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
