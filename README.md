# Proyecto MLOps: Clasificación Binaria

## Descripción
Proyecto de clasificación binaria con machine learning, MLflow, FastAPI, Docker y monitoreo de predicciones.

## Instalación
Activar el entorno virtual con: `.\.venv\Scripts\Activate.ps1`
Instalar dependencias con: `pip install -r requirements.txt`

## Entrenamiento
Ejecutar el pipeline con: `python .\pipeline\run_pipeline.py`

## API
Iniciar localmente con: `python -m uvicorn serving.app:app --host 127.0.0.1 --port 8002`
Documentación: http://127.0.0.1:8002/docs

## Docker
Construir imagen: `docker build -t proyecto-mlops-api .`
Ejecutar contenedor con monitoreo compartido: `docker run --rm -p 8003:8000 -v "${PWD}\monitoring:/app/monitoring" proyecto-mlops-api`

## Monitoreo
Las predicciones se guardan en `monitoring/prediction_log.csv`.

## Pruebas
Ejecutar las pruebas automatizadas con: `python -m pytest -q`.
