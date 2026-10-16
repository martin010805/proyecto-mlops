
from pathlib import Path

import mlflow
import mlflow.sklearn
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main():
    # 1. Crear datos reproducibles con clases desbalanceadas
    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=8,
        n_redundant=4,
        weights=[0.90, 0.10],
        random_state=42,
    )

    print("DistribuciÃ³n de clases:")
    print(f"Clase 0: {(y == 0).sum()} ejemplos")
    print(f"Clase 1: {(y == 1).sum()} ejemplos")

    # 2. Separar los datos para entrenar y evaluar
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # 3. Crear el modelo
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        ),
    )

    # 4. Configurar MLflow
    project_root = Path(__file__).resolve().parent.parent
    mlflow.set_tracking_uri("sqlite:///" + (project_root / "mlflow.db").as_posix())
    mlflow.set_experiment("clasificacion-binaria")

    # 5. Entrenar, evaluar y registrar resultados
    with mlflow.start_run():
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "f1": f1_score(
                y_test, predictions, zero_division=0
            ),
            "roc_auc": roc_auc_score(y_test, probabilities),
        }

        mlflow.log_params({
            "model": "LogisticRegression",
            "class_weight": "balanced",
            "test_size": 0.20,
            "random_state": 42,
            "samples": len(y),
        })

        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path="modelo")

        print("\nMÃ©tricas del modelo:")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")

        print("\nInforme de clasificaciÃ³n:")
        print(classification_report(y_test, predictions))

        print(f"Run ID de MLflow: {mlflow.active_run().info.run_id}")

    print("\nÂ¡Entrenamiento completado!")


if __name__ == "__main__":
    main()

