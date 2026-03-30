FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# COPY MODEL
COPY ../training/artifacts/model.onnx /app/model.onnx

COPY ../feature_store/fraud_feature_repo /app/feature_repo

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]