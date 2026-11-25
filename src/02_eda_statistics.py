"""Módulo de Análisis Exploratorio de Datos (EDA) y Estadística Aplicada.

Este script implementa la Fase 2 del pipeline:
1. Estadística descriptiva univariada (paramétrica y no paramétrica) y pruebas de normalidad.
2. Análisis bivariado: correlaciones (Pearson y Spearman) y segmentación operativa.
3. Contrastes de hipótesis (Mann-Whitney U, Kruskal-Wallis H, Chi-Cuadrado) con tamaño del efecto.
4. Generación de reporte estadístico auditado en `reports/eda_statistical_report.txt`.
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
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORT_OUTPUT_PATH = REPORTS_DIR / "eda_statistical_report.txt"

NUMERIC_FEATURES: List[str] = [
    "Inventory_Level",
    "Temperature",
    "Humidity",
    "Waiting_Time",
    "User_Transaction_Amount",
    "User_Purchase_Frequency",
    "Asset_Utilization",
    "Demand_Forecast",
]

CATEGORICAL_FEATURES: List[str] = [
    "Traffic_Status",
    "Shipment_Status",
    "Logistics_Delay_Reason",
    "Day_of_Week",
]


def load_clean_data(file_path: Path) -> pd.DataFrame:
    """Carga el dataset limpio generado en la Fase 1.

    Args:
        file_path: Ruta al archivo CSV procesado.

    Returns:
        DataFrame limpio con tipos verificados.
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo procesado en {file_path}. "
            "Ejecuta primero `python src/01_data_cleaning.py`."
        )
    df = pd.read_csv(file_path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    logger.info("Dataset limpio cargado: %d registros x %d columnas.", df.shape[0], df.shape[1])
    return df


def compute_univariate_analysis(df: pd.DataFrame, num_cols: List[str]) -> pd.DataFrame:
    """Calcula métricas paramétricas, no paramétricas y test de normalidad Shapiro-Wilk."""
    records = []
    for col in num_cols:
        series = df[col].dropna()
        q1 = series.quantile(0.25)
        median = series.median()
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        shapiro_stat, shapiro_p = stats.shapiro(series)

        records.append(
            {
                "Variable": col,
                "Media": round(series.mean(), 2),
                "Std": round(series.std(), 2),
                "Mediana": round(median, 2),
                "IQR": round(iqr, 2),
                "Min": round(series.min(), 2),
                "Max": round(series.max(), 2),
                "Asimetria": round(series.skew(), 3),
                "Shapiro_W": round(shapiro_stat, 4),
                "Shapiro_p": round(shapiro_p, 6),
                "Distribucion": "Normal" if shapiro_p >= 0.05 else "No Normal",
            }
        )
    return pd.DataFrame(records)


def run_numeric_hypothesis_tests(
    df: pd.DataFrame, num_cols: List[str], target: str = "Logistics_Delay"
) -> pd.DataFrame:
    """Compara variables numéricas entre grupos de `Logistics_Delay` (0 vs 1).

    Utiliza la prueba no paramétrica U de Mann-Whitney y calcula la correlación
    biserial por rangos como medida de tamaño del efecto.
    """
    group_0 = df[df[target] == 0]
    group_1 = df[df[target] == 1]
    n0, n1 = len(group_0), len(group_1)

    results = []
    for col in num_cols:
        s0 = group_0[col].dropna()
        s1 = group_1[col].dropna()

        u_stat, p_val = stats.mannwhitneyu(s0, s1, alternative="two-sided")
        # Tamaño del efecto: Rank-Biserial Correlation (r_rb)
        rank_biserial = abs(1.0 - (2.0 * u_stat) / (n0 * n1))

        results.append(
            {
                "Variable": col,
                "Mediana_Sin_Retraso(0)": round(s0.median(), 2),
                "Mediana_Con_Retraso(1)": round(s1.median(), 2),
                "Diff_Medianas": round(s1.median() - s0.median(), 2),
                "MannWhitney_U": round(u_stat, 1),
                "p_valor": round(p_val, 6),
                "Effect_Size_r": round(rank_biserial, 4),
                "Significativo_5pct": "SI" if p_val < 0.05 else "NO",
            }
        )
    return pd.DataFrame(results).sort_values("p_valor")


def run_categorical_hypothesis_tests(
    df: pd.DataFrame, cat_cols: List[str], target: str = "Logistics_Delay"
) -> pd.DataFrame:
    """Evalúa independencia entre variables categóricas y `Logistics_Delay`.

    Aplica Chi-Cuadrado de Pearson y calcula la V de Cramér para medir intensidad.
    """
    results = []
    n_total = len(df)

    for col in cat_cols:
        if col == "Logistics_Delay_Reason":
            # Excluir motivo de retraso porque está determinísticamente ligado al target
            continue

        contingency = pd.crosstab(df[col], df[target])
        chi2, p_val, dof, _ = stats.chi2_contingency(contingency)

        # V de Cramér
        min_dim = min(contingency.shape[0] - 1, contingency.shape[1] - 1)
        cramers_v = np.sqrt(chi2 / (n_total * min_dim)) if min_dim > 0 else 0.0

        results.append(
            {
                "Variable": col,
                "Chi2": round(chi2, 4),
                "Grados_Libertad": dof,
                "p_valor": round(p_val, 6),
                "Cramers_V": round(cramers_v, 4),
                "Significativo_5pct": "SI" if p_val < 0.05 else "NO",
            }
        )
    return pd.DataFrame(results).sort_values("p_valor")


def run_kruskal_by_traffic(df: pd.DataFrame, num_cols: List[str]) -> pd.DataFrame:
    """Evalúa diferencias entre estados de tráfico usando Kruskal-Wallis H."""
    traffic_levels = df["Traffic_Status"].unique()
    results = []

    for col in num_cols:
        groups = [df[df["Traffic_Status"] == lvl][col].dropna() for lvl in traffic_levels]
        h_stat, p_val = stats.kruskal(*groups)
        results.append(
            {
                "Variable": col,
                "Kruskal_H": round(h_stat, 4),
                "p_valor": round(p_val, 6),
                "Significativo_5pct": "SI" if p_val < 0.05 else "NO",
            }
        )
    return pd.DataFrame(results).sort_values("p_valor")


def generate_operational_segments(df: pd.DataFrame) -> dict:
    """Genera tablas de segmentación operativa por tráfico, activos y KPIs globales."""
    delay_rate = df["Logistics_Delay"].mean() * 100

    by_traffic = (
        df.groupby("Traffic_Status", as_index=False)
        .agg(
            Operaciones=("Asset_ID", "count"),
            Tasa_Retraso_Pct=("Logistics_Delay", lambda x: round(x.mean() * 100, 2)),
            Espera_Mediana_Min=("Waiting_Time", "median"),
            Utilizacion_Mediana_Pct=("Asset_Utilization", "median"),
            Monto_Transaccion_Mediana=("User_Transaction_Amount", "median"),
        )
        .sort_values("Tasa_Retraso_Pct", ascending=False)
    )

    by_asset = (
        df.groupby("Asset_ID", as_index=False)
        .agg(
            Operaciones=("Timestamp", "count"),
            Retrasos_Totales=("Logistics_Delay", "sum"),
            Tasa_Retraso_Pct=("Logistics_Delay", lambda x: round(x.mean() * 100, 2)),
            Espera_Promedio_Min=("Waiting_Time", lambda x: round(x.mean(), 2)),
            Utilizacion_Promedio_Pct=("Asset_Utilization", lambda x: round(x.mean(), 2)),
            Facturacion_Total=("User_Transaction_Amount", lambda x: round(x.sum(), 2)),
        )
        .sort_values("Tasa_Retraso_Pct", ascending=False)
    )

    return {
        "delay_rate": round(delay_rate, 2),
        "by_traffic": by_traffic,
        "by_asset": by_asset,
    }


def main() -> None:
    """Orquesta el análisis estadístico completo de la Fase 2."""
    try:
        df = load_clean_data(CLEAN_DATA_PATH)

        univariate_df = compute_univariate_analysis(df, NUMERIC_FEATURES)
        mw_df = run_numeric_hypothesis_tests(df, NUMERIC_FEATURES)
        chi2_df = run_categorical_hypothesis_tests(df, CATEGORICAL_FEATURES)
        kruskal_df = run_kruskal_by_traffic(df, NUMERIC_FEATURES)
        segments = generate_operational_segments(df)

        # Correlaciones de Spearman y Pearson
        spearman_corr = df[NUMERIC_FEATURES + ["Logistics_Delay"]].corr(method="spearman").round(3)
        pearson_corr = df[NUMERIC_FEATURES + ["Logistics_Delay"]].corr(method="pearson").round(3)

        # Construcción del reporte textual estructurado
        report_lines = [
            "=" * 88,
            "REPORTE DE EDA Y ESTADÍSTICA APLICADA - SMART LOGISTICS OPTIMIZATION",
            "=" * 88,
            f"Total de operaciones analizadas: {len(df)}",
            f"Tasa global de retrasos (Logistics_Delay = 1): {segments['delay_rate']}%",
            "",
            "1. ESTADÍSTICA DESCRIPTIVA UNIVARIADA Y TEST DE NORMALIDAD (SHAPIRO-WILK)",
            "-" * 88,
            univariate_df.to_string(index=False),
            "",
            "2. CONTRASTE DE HIPÓTESIS NUMÉRICAS VS. RETRASO (MANN-WHITNEY U)",
            "-" * 88,
            mw_df.to_string(index=False),
            "",
            "3. CONTRASTE DE INDEPENDENCIA CATEGÓRICA VS. RETRASO (CHI-CUADRADO)",
            "-" * 88,
            chi2_df.to_string(index=False),
            "",
            "4. ANÁLISIS DE VARIANZA POR CONDICIÓN DE TRÁFICO (KRUSKAL-WALLIS H)",
            "-" * 88,
            kruskal_df.to_string(index=False),
            "",
            "5. SEGMENTACIÓN OPERATIVA POR ESTADO DE TRÁFICO",
            "-" * 88,
            segments["by_traffic"].to_string(index=False),
            "",
            "6. DESEMPEÑO OPERATIVO POR ACTIVO (ASSET_ID)",
            "-" * 88,
            segments["by_asset"].to_string(index=False),
            "",
            "7. MATRIZ DE CORRELACIÓN DE SPEARMAN (MONÓTONA)",
            "-" * 88,
            spearman_corr.to_string(),
            "",
            "8. MATRIZ DE CORRELACIÓN DE PEARSON (LINEAL)",
            "-" * 88,
            pearson_corr.to_string(),
            "=" * 88,
        ]

        full_report = "\n".join(report_lines)

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_OUTPUT_PATH.write_text(full_report, encoding="utf-8")

        print(full_report)
        logger.info("Reporte estadístico exportado exitosamente en: %s", REPORT_OUTPUT_PATH)

    except Exception as exc:
        logger.error("Error durante el análisis estadístico (Fase 2): %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()