"""Orquestador End-to-End del Pipeline de Optimización Logística.

Ejecuta secuencialmente las 4 etapas analíticas del proyecto:
- Fase 1: Ingesta, auditoría de calidad y limpieza (`src/01_data_cleaning.py`)
- Fase 2: EDA y contrastes de hipótesis (`src/02_eda_statistics.py`)
- Fase 3: Detección de anomalías y Pareto 80/20 (`src/03_anomalies_pareto.py`)
- Fase 4: Generación de visualizaciones a 300 DPI (`src/04_visualizations.py`)
"""

import importlib.util
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

PIPELINE_STEPS = [
    ("Fase 1: Ingesta y Auditoría de Calidad", SRC_DIR / "01_data_cleaning.py"),
    ("Fase 2: EDA y Pruebas de Hipótesis", SRC_DIR / "02_eda_statistics.py"),
    ("Fase 3: Anomalías y Pareto 80/20", SRC_DIR / "03_anomalies_pareto.py"),
    ("Fase 4: Visualización Ejecutiva (300 DPI)", SRC_DIR / "04_visualizations.py"),
]


def run_module_main(step_name: str, script_path: Path) -> None:
    """Carga dinámicamente un script `.py` y ejecuta su función `main()`."""
    if not script_path.exists():
        raise FileNotFoundError(f"No se encontró el módulo requerido: {script_path}")

    logger.info(">>> INICIANDO %s (%s)", step_name, script_path.name)
    spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo cargar la especificación para {script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "main"):
        raise AttributeError(f"El script {script_path.name} no define una función `main()`.")

    module.main()
    logger.info("<<< COMPLETADO %s", step_name)


def main() -> None:
    """Ejecuta el pipeline completo midiendo tiempos de procesamiento."""
    start_time = time.perf_counter()
    logger.info("=================================================================")
    logger.info("INICIANDO PIPELINE COMPLETO: SMART LOGISTICS OPTIMIZATION")
    logger.info("=================================================================")

    try:
        for step_name, script_path in PIPELINE_STEPS:
            run_module_main(step_name, script_path)

        elapsed = time.perf_counter() - start_time
        logger.info("=================================================================")
        logger.info("PIPELINE FINALIZADO EXITOSAMENTE EN %.2f SEGUNDOS.", elapsed)
        logger.info("=================================================================")

    except Exception as exc:
        logger.error("Ejecución abortada debido a un error crítico: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()