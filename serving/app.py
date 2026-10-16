import os
from pathlib import Path

import mlflow
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATABASE_PATH = PROJECT_ROOT / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{DATABASE_PATH.as_posix()}")

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "mlruns"
    / "1"
    / "models"
    / "m-bee73561f52d446db4f6b26fb7fbf650"
    / "artifacts"
)

MODEL_PATH = Path(
    os.getenv("MLFLOW_MODEL_PATH", str(DEFAULT_MODEL_PATH))
)

model = mlflow.sklearn.load_model(str(MODEL_PATH))

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
        "model_path": str(MODEL_PATH),
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(data: PredictionInput):
    try:
        prediction = int(model.predict([data.features])[0])
        probabilities = model.predict_proba([data.features])[0]

        return {
            "prediction": prediction,
            "probability_class_0": round(float(probabilities[0]), 4),
            "probability_class_1": round(float(probabilities[1]), 4),
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error al generar la predicción: {exc}",
        ) from exc
