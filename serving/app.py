import csv
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from monitoring.logger import log_prediction

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATABASE_PATH = (PROJECT_ROOT / "mlflow.db").as_posix()
LOG_PATH = PROJECT_ROOT / "monitoring" / "prediction_log.csv"

mlflow.set_tracking_uri(
    os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{DATABASE_PATH}")
)

MODEL_URI = os.getenv(
    "MLFLOW_MODEL_PATH",
    str(
        PROJECT_ROOT
        / "mlruns"
        / "1"
        / "models"
        / "m-f85504f3b2b3463d9baf5b850c9e98fa"
        / "artifacts"
    ),
)

model = mlflow.sklearn.load_model(MODEL_URI)

app = FastAPI(
    title="API de Clasificación Binaria",
    description="Servicio de predicciones para nuestro modelo MLOps",
    version="1.1.0",
)


class PredictionInput(BaseModel):
    features: list[float] = Field(
        ...,
        min_length=20,
        max_length=20,
        description="Lista de exactamente 20 características numéricas",
    )


def make_prediction(features):
    prediction = int(model.predict([features])[0])
    probabilities = model.predict_proba([features])[0]

    probability_class_0 = round(float(probabilities[0]), 4)
    probability_class_1 = round(float(probabilities[1]), 4)

    log_prediction(
        prediction,
        probability_class_0,
        probability_class_1,
    )

    return prediction, probability_class_0, probability_class_1


@app.get("/")
def home():
    return {
        "message": "API de clasificación binaria funcionando",
        "model_path": MODEL_URI,
        "version": app.version,
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
        prediction, prob_0, prob_1 = make_prediction(data.features)
        return {
            "prediction": prediction,
            "probability_class_0": prob_0,
            "probability_class_1": prob_1,
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error al generar la predicción: {exc}",
        ) from exc


@app.post("/predict/explain")
def predict_explain(data: PredictionInput):
    try:
        prediction, prob_0, prob_1 = make_prediction(data.features)

        explanation = []
        explanation_method = "No hay un método de importancia compatible con este modelo."

        # Explicación local para modelos lineales, como LogisticRegression.
        if hasattr(model, "steps") and model.steps:
            estimator = model.steps[-1][1]

            if hasattr(estimator, "coef_"):
                transformed = model[:-1].transform([data.features])
                coefficients = estimator.coef_[0]
                contributions = transformed[0] * coefficients

                explanation = [
                    {
                        "feature": f"feature_{i + 1}",
                        "input_value": float(data.features[i]),
                        "contribution_to_class_1": round(float(value), 6),
                    }
                    for i, value in enumerate(contributions)
                ]
                explanation.sort(
                    key=lambda item: abs(item["contribution_to_class_1"]),
                    reverse=True,
                )
                explanation = explanation[:5]
                explanation_method = (
                    "Contribuciones lineales aproximadas en el espacio transformado; "
                    "no equivalen a valores SHAP."
                )

            elif hasattr(estimator, "feature_importances_"):
                importances = estimator.feature_importances_
                explanation = sorted(
                    [
                        {
                            "feature": f"feature_{i + 1}",
                            "input_value": float(data.features[i]),
                            "global_importance": round(float(value), 6),
                        }
                        for i, value in enumerate(importances)
                    ],
                    key=lambda item: item["global_importance"],
                    reverse=True,
                )[:5]
                explanation_method = (
                    "Importancia global del modelo; no representa una explicación local."
                )

        return {
            "prediction": prediction,
            "probability_class_0": prob_0,
            "probability_class_1": prob_1,
            "top_features": explanation,
            "explanation_method": explanation_method,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error al explicar la predicción: {exc}",
        ) from exc


@app.get("/metrics")
def metrics():
    try:
        total_predictions = 0
        predicted_class_0 = 0
        predicted_class_1 = 0
        probability_sum_0 = 0.0
        probability_sum_1 = 0.0

        if LOG_PATH.exists():
            with LOG_PATH.open(
                "r", newline="", encoding="utf-8-sig"
            ) as file:
                reader = csv.DictReader(file)

                for row in reader:
                    try:
                        prediction = int(row["prediction"])
                        prob_0 = float(row["probability_class_0"])
                        prob_1 = float(row["probability_class_1"])
                    except (KeyError, TypeError, ValueError):
                        continue

                    total_predictions += 1
                    predicted_class_0 += int(prediction == 0)
                    predicted_class_1 += int(prediction == 1)
                    probability_sum_0 += prob_0
                    probability_sum_1 += prob_1

        return {
            "total_predictions_logged": total_predictions,
            "predictions_class_0": predicted_class_0,
            "predictions_class_1": predicted_class_1,
            "mean_probability_class_0": (
                round(probability_sum_0 / total_predictions, 4)
                if total_predictions else None
            ),
            "mean_probability_class_1": (
                round(probability_sum_1 / total_predictions, 4)
                if total_predictions else None
            ),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error al consultar las métricas: {exc}",
        ) from exc
