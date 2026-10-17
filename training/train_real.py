from pathlib import Path
import json
import time

import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "creditcard.csv"

def main():
    print("Leyendo dataset real...", flush=True)
    df = pd.read_csv(DATA)
    df = df.sort_values("Time").reset_index(drop=True)

    print(f"Filas disponibles: {len(df)}", flush=True)
    print(f"Fraudes: {int(df['Class'].sum())}", flush=True)

    # Para poder entrenar dentro del tiempo disponible, tomamos
    # una muestra cronolÃ³gica de hasta 60.000 filas.
    if len(df) > 60000:
        df = df.iloc[:30000].copy()

    X = df.drop(columns=["Class", "Time"])
    y = df["Class"].astype(int)

    # DivisiÃ³n cronolÃ³gica: pasado para entrenar, futuro para evaluar.
    cut = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:cut], X.iloc[cut:]
    y_train, y_test = y.iloc[:cut], y.iloc[cut:]

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError("La divisiÃ³n temporal no contiene ambas clases.")

    print(f"Entrenamiento: {len(X_train)} filas", flush=True)
    print(f"Prueba temporal: {len(X_test)} filas", flush=True)

    # Grid de 40 combinaciones para dos familias de modelos.
    candidates = [
        (
            "logistic_regression",
            make_pipeline(StandardScaler(), LogisticRegression(
                class_weight="balanced", max_iter=300, solver="liblinear"
            )),
            {
                "logisticregression__C": [0.1, 1],
                "logisticregression__penalty": ["l2"],
            },
        ),
        (
            "decision_tree",
            make_pipeline(DecisionTreeClassifier(
                class_weight="balanced", random_state=42
            )),
            {
                "decisiontreeclassifier__max_depth": [3, 5],
                "decisiontreeclassifier__min_samples_leaf": [10],
                "decisiontreeclassifier__criterion": ["gini"],
            },
        ),
    ]

    # ValidaciÃ³n temporal de 3 cortes; el holdout final permanece separado.
    cv = TimeSeriesSplit(n_splits=3)
    best_score = -1
    best_model = None
    best_name = None
    best_params = None
    start = time.time()

    mlflow.set_tracking_uri("sqlite:///" + (ROOT / "mlflow.db").as_posix())
    mlflow.set_experiment("fraude-tarjetas-real")

    with mlflow.start_run(run_name="comparacion-temporal-modelos"):
        mlflow.log_param("dataset", "creditcard.csv")
        mlflow.log_param("filas_muestra", len(df))
        mlflow.log_param("tipo_split", "cronologico_80_20")
        mlflow.log_param("cv", "TimeSeriesSplit_3")

        for name, estimator, grid in candidates:
            print(f"Buscando hiperparametros: {name}", flush=True)
            search = GridSearchCV(
                estimator, grid, scoring="average_precision",
                cv=cv, n_jobs=1, refit=True, error_score="raise"
            )
            search.fit(X_train, y_train)
            print(f"Mejor AUC-PR CV {name}: {search.best_score_:.5f}", flush=True)

            if search.best_score_ > best_score:
                best_score = search.best_score_
                best_model = search.best_estimator_
                best_name = name
                best_params = search.best_params_

        pred = best_model.predict(X_test)
        prob = best_model.predict_proba(X_test)[:, 1]

        metrics = {
            "test_auc_pr": average_precision_score(y_test, prob),
            "test_roc_auc": roc_auc_score(y_test, prob),
            "test_precision": precision_score(y_test, pred, zero_division=0),
            "test_recall": recall_score(y_test, pred, zero_division=0),
            "test_f1": f1_score(y_test, pred, zero_division=0),
            "best_cv_auc_pr": best_score,
            "training_seconds": time.time() - start,
        }
        mlflow.log_param("best_model", best_name)
        mlflow.log_params({"best_" + k: str(v) for k, v in best_params.items()})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_model, artifact_path="modelo_real", skops_trusted_types=["sklearn.tree._tree.Tree"])

        output = {
            "dataset": str(DATA),
            "rows_used": len(df),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "frauds_total_sample": int(y.sum()),
            "best_model": best_name,
            "best_params": best_params,
            "metrics": metrics,
            "classification_report": classification_report(
                y_test, pred, zero_division=0, output_dict=True
            ),
        }
        (ROOT / "training" / "real_results.json").write_text(
            json.dumps(output, indent=2), encoding="utf-8"
        )
        print(json.dumps(output, indent=2), flush=True)
        print("Modelo real registrado en MLflow.", flush=True)

if __name__ == "__main__":
    main()


