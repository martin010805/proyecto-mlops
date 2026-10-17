import os
from pathlib import Path

import mlflow
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from monitoring.logger import log_prediction

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATABASE_PATH = (PROJECT_ROOT / "mlflow.db").as_posix()

mlflow.set_tracking_uri(f"sqlite:///{DATABASE_PATH}")

MODEL_URI = os.getenv(
    "MLFLOW_MODEL_PATH",
    str(PROJECT_ROOT / "mlruns" / "1" / "models" / "m-f85504f3b2b3463d9baf5b850c9e98fa" / "artifacts"),
)

model = mlflow.sklearn.load_model(MODEL_URI)

app = FastAPI(
    title="API de Clasificación Binaria",
    description="Servicio de predicciones para nuestro modelo MLOps",
    version="1.0.0",
)


class PredictionInput(BaseModel):
    features: list[float] = Field(
        ...,
        min_length=20,
        max_length=20,
        description="Lista de exactamente 20 características numéricas",
    )


@app.get("/")
def home():
    return {
        "message": "API de clasificación binaria funcionando",
        "model_path": MODEL_URI,
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
    }


@app.post("/predict")
def predict(data: PredictionInput):
    try:
        prediction = int(model.predict([data.features])[0])
        probabilities = model.predict_proba([data.features])[0]

        probability_class_0 = round(float(probabilities[0]), 4)
        probability_class_1 = round(float(probabilities[1]), 4)

        log_prediction(
            prediction,
            probability_class_0,
            probability_class_1,
        )

        return {
            "prediction": prediction,
            "probability_class_0": probability_class_0,
            "probability_class_1": probability_class_1,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error al generar la predicción: {exc}",
        ) from exc


