"""Módulo de Detección Estadística de Anomalías y Análisis de Pareto (80/20).

Este script implementa la Fase 3 del pipeline:
1. Detección de outliers univariados mediante Rango Intercuartílico (IQR) y Z-Score.
2. Detección de anomalías operativas compuestas (Exposición financiera en riesgo crítico).
3. Análisis de Pareto (80/20) sobre causas raíz de retraso y combinaciones Activo-Tráfico.
4. Exportación de reporte cuantitativo y dataset enriquecido con flags de control.
"""

import logging
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from scipy import stats

# Configuración de logging estructurado
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Rutas del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "smart_logistics_clean.csv"
ENRICHED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "smart_logistics_enriched.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORT_OUTPUT_PATH = REPORTS_DIR / "anomalies_pareto_report.txt"

AUDIT_NUMERIC_COLS: List[str] = [
    "Inventory_Level",
    "Temperature",
    "Humidity",
    "Waiting_Time",
    "User_Transaction_Amount",
    "User_Purchase_Frequency",
    "Asset_Utilization",
    "Demand_Forecast",
]


def load_clean_data(file_path: Path) -> pd.DataFrame:
    """Carga el dataset limpio de la Fase 1."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"No se encontró {file_path}. Ejecuta primero `python src/01_data_cleaning.py`."
        )
    df = pd.read_csv(file_path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    return df


def detect_univariate_anomalies(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Evalúa presencia de outliers univariados usando IQR (Tukey) y Z-Score (|Z| > 2.5)."""
    records = []
    for col in cols:
        series = df[col].dropna()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        iqr_outliers = int(((series < lower_bound) | (series > upper_bound)).sum())
        z_scores = np.abs(stats.zscore(series))
        z_outliers_25 = int((z_scores > 2.5).sum())
        z_outliers_30 = int((z_scores > 3.0).sum())

        records.append(
            {
                "Variable": col,
                "Limite_Inf_IQR": round(lower_bound, 2),
                "Limite_Sup_IQR": round(upper_bound, 2),
                "Outliers_IQR": iqr_outliers,
                "Outliers_Z_2.5": z_outliers_25,
                "Outliers_Z_3.0": z_outliers_30,
            }
        )
    return pd.DataFrame(records)


def flag_operational_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Crea indicadores (flags) de anomalías compuestas y exposición financiera.

    Criterios de Control de Gestión:
    - High_Value_Flag: Transacción >= Percentil 75 (Q3).
    - Critical_Financial_Delay: Retraso confirmado (1) en operación de alto valor (>= Q3).
    - Severe_Bottleneck: Tráfico 'Heavy' + Tiempo de espera >= Percentil 75 + Retraso (1).
    """
    df_out = df.copy()

    q3_amount = df_out["User_Transaction_Amount"].quantile(0.75)
    q3_wait = df_out["Waiting_Time"].quantile(0.75)

    df_out["High_Value_Flag"] = (df_out["User_Transaction_Amount"] >= q3_amount).astype(int)
    df_out["Critical_Financial_Delay"] = (
        (df_out["Logistics_Delay"] == 1) & (df_out["High_Value_Flag"] == 1)
    ).astype(int)
    df_out["Severe_Bottleneck"] = (
        (df_out["Logistics_Delay"] == 1)
        & (df_out["Traffic_Status"] == "Heavy")
        & (df_out["Waiting_Time"] >= q3_wait)
    ).astype(int)

    # Índice de Riesgo Operativo (0 a 100) ponderando tráfico, espera y utilización
    traffic_risk = df_out["Traffic_Status"].map({"Heavy": 1.0, "Detour": 0.4, "Clear": 0.2})
    norm_wait = (df_out["Waiting_Time"] - df_out["Waiting_Time"].min()) / (
        df_out["Waiting_Time"].max() - df_out["Waiting_Time"].min()
    )
    norm_util = df_out["Asset_Utilization"] / 100.0

    df_out["Operational_Risk_Score"] = (
        (0.50 * traffic_risk + 0.30 * norm_wait + 0.20 * norm_util) * 100
    ).round(2)

    return df_out


def build_pareto_table(
    df: pd.DataFrame, group_cols: List[str], value_col: str, agg_func: str = "count"
) -> pd.DataFrame:
    """Construye una tabla de Pareto (80/20) ordenada por impacto descendente."""
    if agg_func == "count":
        grouped = df.groupby(group_cols, as_index=False)[value_col].count()
    elif agg_func == "sum":
        grouped = df.groupby(group_cols, as_index=False)[value_col].sum()
    else:
        raise ValueError("agg_func debe ser 'count' o 'sum'.")

    grouped = grouped.rename(columns={value_col: "Impacto"})
    grouped = grouped.sort_values("Impacto", ascending=False).reset_index(drop=True)

    total = grouped["Impacto"].sum()
    grouped["Pct_Individual"] = ((grouped["Impacto"] / total) * 100).round(2)
    grouped["Pct_Acumulado"] = grouped["Pct_Individual"].cumsum().round(2)
    grouped["Clasificacion_ABC"] = np.where(
        grouped["Pct_Acumulado"] <= 80.0, "A - Vital (<=80%)", "B/C - Trivial (>80%)"
    )
    # Asegurar que el primer elemento que cruza el umbral del 80% quede incluido en Clase A
    first_over_80 = grouped[grouped["Pct_Acumulado"] > 80.0].index.min()
    if pd.notna(first_over_80):
        grouped.loc[first_over_80, "Clasificacion_ABC"] = "A - Vital (Corte 80%)"

    return grouped


def main() -> None:
    """Orquesta la detección de anomalías y análisis de Pareto (Fase 3)."""
    try:
        df = load_clean_data(CLEAN_DATA_PATH)

        # 1. Auditoría univariada de outliers
        univariate_outliers = detect_univariate_anomalies(df, AUDIT_NUMERIC_COLS)

        # 2. Enriquecimiento con anomalías compuestas y riesgo operativo
        df_enriched = flag_operational_anomalies(df)
        delayed_df = df_enriched[df_enriched["Logistics_Delay"] == 1].copy()

        # 3. Análisis de Pareto 80/20
        # 3a. Pareto por Causa Raíz de Retraso (Frecuencia)
        pareto_reason_freq = build_pareto_table(
            delayed_df, ["Logistics_Delay_Reason"], "Timestamp", agg_func="count"
        )
        # 3b. Pareto por Causa Raíz de Retraso (Impacto Financiero / Monto en Riesgo)
        pareto_reason_fin = build_pareto_table(
            delayed_df, ["Logistics_Delay_Reason"], "User_Transaction_Amount", agg_func="sum"
        )
        # 3c. Pareto por Binomio Activo + Tráfico (Frecuencia de Retrasos)
        pareto_asset_traffic = build_pareto_table(
            delayed_df, ["Asset_ID", "Traffic_Status"], "Timestamp", agg_func="count"
        )

        # Métricas ejecutivas de impacto
        total_revenue = df_enriched["User_Transaction_Amount"].sum()
        delayed_revenue = delayed_df["User_Transaction_Amount"].sum()
        critical_delays_count = int(df_enriched["Critical_Financial_Delay"].sum())
        critical_delays_rev = df_enriched.loc[
            df_enriched["Critical_Financial_Delay"] == 1, "User_Transaction_Amount"
        ].sum()
        severe_bottlenecks = int(df_enriched["Severe_Bottleneck"].sum())

        report_lines = [
            "=" * 92,
            "REPORTE DE DETECCIÓN DE ANOMALÍAS Y ANÁLISIS DE PARETO (80/20)",
            "=" * 92,
            f"Facturación Total Analizada: ${total_revenue:,.2f}",
            f"Facturación Expuesta a Retrasos: ${delayed_revenue:,.2f} ({delayed_revenue/total_revenue*100:.2f}%)",
            f"Anomalías Críticas (Retraso + Alto Valor >= Q3): {critical_delays_count} ops (${critical_delays_rev:,.2f})",
            f"Cuellos de Botella Severos (Tráfico Heavy + Espera >= Q3): {severe_bottlenecks} ops",
            "",
            "1. AUDITORÍA DE OUTLIERS UNIVARIADOS (IQR DE TUKEY Y Z-SCORE)",
            "-" * 92,
            univariate_outliers.to_string(index=False),
            "",
            "2. PARETO 80/20 POR CAUSA RAÍZ DE RETRASO (FRECUENCIA DE INCIDENCIAS)",
            "-" * 92,
            pareto_reason_freq.to_string(index=False),
            "",
            "3. PARETO 80/20 POR CAUSA RAÍZ DE RETRASO (EXPOSICIÓN FINANCIERA $)",
            "-" * 92,
            pareto_reason_fin.to_string(index=False),
            "",
            "4. TOP 10 COMBINACIONES CRÍTICAS (ACTIVO + CONDICIÓN DE TRÁFICO)",
            "-" * 92,
            pareto_asset_traffic.head(10).to_string(index=False),
            "=" * 92,
        ]

        full_report = "\n".join(report_lines)

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_OUTPUT_PATH.write_text(full_report, encoding="utf-8")
        df_enriched.to_csv(ENRICHED_DATA_PATH, index=False)

        print(full_report)
        logger.info("Reporte de Anomalías y Pareto guardado en: %s", REPORT_OUTPUT_PATH)
        logger.info("Dataset enriquecido guardado en: %s", ENRICHED_DATA_PATH)

    except Exception as exc:
        logger.error("Error durante la Fase 3 (Anomalías y Pareto): %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()