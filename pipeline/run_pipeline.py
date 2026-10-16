from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_SCRIPT = PROJECT_ROOT / "training" / "train.py"


def main():
    print("=== INICIO DEL PIPELINE MLOps ===")
    print("Paso 1: Ejecutando el entrenamiento del modelo...")

    subprocess.run(
        [sys.executable, str(TRAINING_SCRIPT)],
        cwd=PROJECT_ROOT,
        check=True,
    )

    print("\n=== PIPELINE COMPLETADO CORRECTAMENTE ===")


if __name__ == "__main__":
    main()
