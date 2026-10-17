import json
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient

root = Path.cwd()
mlflow.set_tracking_uri("sqlite:///" + (root / "mlflow.db").as_posix())
client = MlflowClient()

experiments = client.search_experiments()
exp = next((e for e in experiments if e.name == "fraude-tarjetas-real"), None)

if exp is None:
    print("No se encontro el experimento.")
    raise SystemExit(1)

runs = client.search_runs(
    experiment_ids=[exp.experiment_id],
    order_by=["attributes.start_time DESC"],
    max_results=10
)

results = []
for run in runs:
    results.append({
        "run_id": run.info.run_id,
        "status": run.info.status,
        "params": run.data.params,
        "metrics": run.data.metrics
    })

out = root / "training" / "real_results.json"
out.write_text(json.dumps(results, indent=2), encoding="utf-8")
print(json.dumps(results, indent=2))
print(f"\nResultados guardados en: {out}")
