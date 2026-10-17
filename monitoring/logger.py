
import csv
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = PROJECT_ROOT / "monitoring" / "prediction_log.csv"

CSV_COLUMNS = [
    "timestamp",
    "prediction",
    "probability_class_0",
    "probability_class_1",
]


def log_prediction(prediction, probability_class_0, probability_class_1):
    """Guarda una predicción de la API en un archivo CSV."""
    file_exists = LOG_PATH.exists()
    needs_header = not file_exists or LOG_PATH.stat().st_size == 0

    with LOG_PATH.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)

        if needs_header:
            writer.writeheader()

        writer.writerow({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prediction": prediction,
            "probability_class_0": probability_class_0,
            "probability_class_1": probability_class_1,
        })
