"""Módulo de ingesta, auditoría de calidad y limpieza de datos logísticos.

Este script implementa la Fase 1 del pipeline:
1. Carga inmutable desde `data/raw/`.
2. Diagnóstico estructural (tipos, nulos, duplicados).
3. Validación programática de reglas de negocio y consistencia física/financiera.
4. Limpieza, imputación semántica y enriquecimiento temporal.
5. Exportación auditada hacia `data/processed/smart_logistics_clean.csv`.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

# Configuración de logging estructurado para trazabilidad de auditoría
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Definición de rutas relativas a la raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = PROCESSED_DATA_DIR / "smart_logistics_clean.csv"

EXPECTED_COLUMNS = {
    "Timestamp",
    "Asset_ID",
    "Latitude",
    "Longitude",
    "Inventory_Level",
    "Shipment_Status",
    "Temperature",
    "Humidity",
    "Traffic_Status",
    "Waiting_Time",
    "User_Transaction_Amount",
    "User_Purchase_Frequency",
    "Logistics_Delay_Reason",
    "Asset_Utilization",
    "Demand_Forecast",
    "Logistics_Delay",
}


def resolve_raw_csv_path(raw_dir: Path) -> Path:
    """Localiza el archivo CSV crudo dentro de `data/raw/`.

    Args:
        raw_dir: Directorio donde se almacenan los datos originales.

    Returns:
        Ruta absoluta al archivo CSV encontrado.

    Raises:
        FileNotFoundError: Si no existe ningún archivo CSV en `data/raw/`.
        ValueError: Si existen múltiples archivos CSV y hay ambigüedad.
    """
    if not raw_dir.exists():
        raise FileNotFoundError(f"El directorio no existe: {raw_dir}")

    csv_files = list(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No se encontró ningún archivo .csv en {raw_dir}. "
            "Descarga el dataset de Kaggle y guárdalo en esa carpeta."
        )
    if len(csv_files) > 1:
        raise ValueError(
            f"Se encontraron múltiples archivos CSV en {raw_dir}: "
            f"{[f.name for f in csv_files]}. Deja únicamente el dataset principal."
        )
    return csv_files[0]


def load_raw_data(file_path: Path) -> pd.DataFrame:
    """Carga el dataset crudo validando esquema mínimo de columnas.

    Args:
        file_path: Ruta al archivo CSV de origen.

    Returns:
        DataFrame de pandas con los datos crudos.
    """
    logger.info("Iniciando carga inmutable desde: %s", file_path.name)
    df = pd.read_csv(file_path)

    # Limpiar espacios en blanco accidentales en encabezados
    df.columns = df.columns.str.strip()

    missing_cols = EXPECTED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"Faltan columnas esperadas en el dataset: {missing_cols}")

    logger.info(
        "Dataset cargado exitosamente: %d filas x %d columnas (Memoria: %.2f KB)",
        df.shape[0],
        df.shape[1],
        df.memory_usage(deep=True).sum() / 1024,
    )
    return df


def audit_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Ejecuta un diagnóstico integral de completitud, unicidad y tipos.

    Args:
        df: DataFrame crudo a auditar.

    Returns:
        Diccionario con métricas resumen de la auditoría.
    """
    logger.info("--- INICIO DE AUDITORÍA DE CALIDAD DE DATOS ---")

    # 1. Duplicados exactos y por clave operativa
    exact_dups = int(df.duplicated().sum())
    key_dups = int(df.duplicated(subset=["Asset_ID", "Timestamp"]).sum())
    logger.info("Duplicados exactos detectados: %d", exact_dups)
    logger.info("Duplicados por clave (Asset_ID + Timestamp): %d", key_dups)

    # 2. Reporte de valores nulos
    null_counts = df.isnull().sum()
    null_pct = (null_counts / len(df)) * 100
    null_report = pd.DataFrame({"nulos": null_counts, "porcentaje": null_pct})
    null_report = null_report[null_report["nulos"] > 0]

    if null_report.empty:
        logger.info("Completitud: 100%% (0 valores nulos detectados).")
    else:
        logger.warning("Columnas con valores nulos detectadas:")
        for col, row in null_report.iterrows():
            logger.warning(
                "  -> %-25s : %d nulos (%.2f%%)",
                col,
                int(row["nulos"]),
                row["porcentaje"],
            )

    # 3. Tipos de datos actuales
    logger.info("Tipos de datos detectados (dtypes):")
    for col, dtype in df.dtypes.items():
        logger.info("  -> %-25s : %s", col, dtype)

    return {
        "rows": len(df),
        "exact_duplicates": exact_dups,
        "key_duplicates": key_dups,
        "null_columns": null_report.to_dict(orient="index"),
    }


def validate_business_rules(df: pd.DataFrame) -> None:
    """Valida reglas físicas, geográficas y de control de gestión sobre los datos.

    Registra advertencias (`WARNING`) si existen violaciones de dominio.
    """
    logger.info("--- VALIDACIÓN DE REGLAS DE NEGOCIO Y DOMINIO ---")

    rules = {
        "Asset_Utilization fuera de [0, 100]%": ~df["Asset_Utilization"].between(0, 100),
        "Humidity fuera de [0, 100]%": ~df["Humidity"].between(0, 100),
        "Waiting_Time negativo (< 0)": df["Waiting_Time"] < 0,
        "Inventory_Level negativo (< 0)": df["Inventory_Level"] < 0,
        "User_Transaction_Amount negativo (< 0)": df["User_Transaction_Amount"] < 0,
        "Demand_Forecast negativo (< 0)": df["Demand_Forecast"] < 0,
        "Latitude fuera de [-90, 90]": ~df["Latitude"].between(-90, 90),
        "Longitude fuera de [-180, 180]": ~df["Longitude"].between(-180, 180),
        "Logistics_Delay no es binario {0, 1}": ~df["Logistics_Delay"].isin([0, 1]),
    }

    violations_found = False
    for rule_name, invalid_mask in rules.items():
        count = int(invalid_mask.sum())
        if count > 0:
            violations_found = True
            logger.warning("REGLA VIOLADA [%s]: %d registros afectados.", rule_name, count)

    # Consistencia cruzada: Retraso vs Motivo de Retraso
    delay_without_reason = df[
        (df["Logistics_Delay"] == 1) & (df["Logistics_Delay_Reason"].isna())
    ]
    if not delay_without_reason.empty:
        violations_found = True
        logger.warning(
            "INCONSISTENCIA: %d envíos marcados con retraso (1) carecen de motivo.",
            len(delay_without_reason),
        )

    if not violations_found:
        logger.info("Todas las reglas de negocio y rangos de dominio superados sin errores.")


def clean_and_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica limpieza determinística, imputación semántica y variables temporales.

    Args:
        df: DataFrame crudo validado.

    Returns:
        DataFrame limpio listo para análisis exploratorio y estadístico.
    """
    logger.info("--- INICIO DE TRANSFORMACIÓN Y LIMPIEZA ---")
    df_clean = df.copy()

    # 1. Eliminar duplicados exactos si existen
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates().reset_index(drop=True)
    removed_dups = initial_rows - len(df_clean)
    if removed_dups > 0:
        logger.info("Se eliminaron %d registros duplicados exactos.", removed_dups)

    # 2. Parseo estricto de Timestamp
    df_clean["Timestamp"] = pd.to_datetime(df_clean["Timestamp"], errors="coerce")
    invalid_dates = int(df_clean["Timestamp"].isna().sum())
    if invalid_dates > 0:
        logger.warning("Se detectaron %d fechas inválidas (NaT) tras parseo.", invalid_dates)

    # 3. Estandarizar texto en variables categóricas
    cat_cols = ["Asset_ID", "Shipment_Status", "Traffic_Status", "Logistics_Delay_Reason"]
    for col in cat_cols:
        # Preservar NaN reales antes de aplicar strip()
        mask_notna = df_clean[col].notna()
        df_clean.loc[mask_notna, col] = df_clean.loc[mask_notna, col].astype(str).str.strip()

    # 4. Imputación semántica de Nulos Estructurales en Logistics_Delay_Reason
    # Si Logistics_Delay == 0 y el motivo es nulo o 'None', asignamos 'No Delay'
    no_delay_mask = (df_clean["Logistics_Delay"] == 0) & (
        df_clean["Logistics_Delay_Reason"].isna()
        | (df_clean["Logistics_Delay_Reason"].str.lower() == "none")
    )
    imputed_no_delay = int(no_delay_mask.sum())
    if imputed_no_delay > 0:
        df_clean.loc[no_delay_mask, "Logistics_Delay_Reason"] = "No Delay"
        logger.info(
            "Imputación semántica: %d registros sin retraso etiquetados como 'No Delay'.",
            imputed_no_delay,
        )

    # Si aún quedaran nulos en Logistics_Delay_Reason (cuando Delay == 1), etiquetar 'Unknown'
    remaining_reason_nulls = int(df_clean["Logistics_Delay_Reason"].isna().sum())
    if remaining_reason_nulls > 0:
        df_clean["Logistics_Delay_Reason"] = df_clean["Logistics_Delay_Reason"].fillna("Unknown")
        logger.warning(
            "Se etiquetaron %d motivos de retraso faltantes como 'Unknown'.",
            remaining_reason_nulls,
        )

    # 5. Imputación defensiva para nulos numéricos residuales (usando mediana)
    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        nulls_in_col = int(df_clean[col].isna().sum())
        if nulls_in_col > 0:
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)
            logger.warning(
                "Imputación numérica en '%s': %d nulos reemplazados por mediana (%.2f).",
                col,
                nulls_in_col,
                median_val,
            )

    # 6. Asegurar tipo entero en variable objetivo binaria
    df_clean["Logistics_Delay"] = df_clean["Logistics_Delay"].astype(int)

    # 7. Enriquecimiento temporal para análisis de estacionalidad
    df_clean = df_clean.sort_values("Timestamp").reset_index(drop=True)
    df_clean["Month"] = df_clean["Timestamp"].dt.month
    df_clean["Day_of_Week"] = df_clean["Timestamp"].dt.day_name()
    df_clean["Hour"] = df_clean["Timestamp"].dt.hour

    logger.info(
        "Limpieza finalizada. Dataset resultante: %d filas x %d columnas.",
        df_clean.shape[0],
        df_clean.shape[1],
    )
    return df_clean


def main() -> None:
    """Orquesta la ejecución completa de la Fase 1."""
    try:
        raw_csv_path = resolve_raw_csv_path(RAW_DATA_DIR)
        df_raw = load_raw_data(raw_csv_path)
        audit_data_quality(df_raw)
        validate_business_rules(df_raw)
        df_clean = clean_and_transform(df_raw)

        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        df_clean.to_csv(OUTPUT_FILE, index=False)
        logger.info("Dataset procesado guardado exitosamente en: %s", OUTPUT_FILE)

    except Exception as exc:
        logger.error("Fallo crítico en el pipeline de limpieza: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()