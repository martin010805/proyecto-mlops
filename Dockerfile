FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY serving/ ./serving/
COPY mlflow.db .
COPY mlruns/ ./mlruns/

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]

COPY monitoring/ ./monitoring/
