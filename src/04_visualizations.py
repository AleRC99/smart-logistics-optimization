"""Módulo de Visualización Ejecutiva (300 DPI) para Optimización Logística.

Este script implementa la Fase 4 del pipeline:
1. Configura una identidad visual corporativa limpia (sin ruido gráfico).
2. Genera 4 visualizaciones clave basadas en los hallazgos estadísticos de Fases 2 y 3:
   - 01_traffic_bottleneck_analysis.png: Impacto determinístico del tráfico en retrasos.
   - 02_pareto_financial_exposure.png: Pareto 80/20 de exposición financiera por causa raíz.
   - 03_asset_risk_heatmap.png: Mapa de calor de tasa de retrasos por Activo y Tráfico.
   - 04_financial_anomaly_quadrants.png: Cuadrantes de riesgo financiero y anomalías >= Q3.
3. Exporta todas las figuras a 300 DPI en `reports/figures/`.
"""

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Rutas del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENRICHED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "smart_logistics_enriched.csv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

# Paleta corporativa semántica
PALETTE = {
    "navy": "#1B365D",
    "slate": "#5C768D",
    "light_gray": "#D9E1E8",
    "crimson": "#C0392B",
    "amber": "#D97706",
    "teal": "#0D9488",
    "text_dark": "#1E293B",
    "text_muted": "#475569",
}


def configure_style() -> None:
    """Establece estilo tipográfico y minimalista global para Matplotlib y Seaborn."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "axes.edgecolor": "#CBD5E1",
            "axes.linewidth": 0.8,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.titlecolor": PALETTE["text_dark"],
            "axes.labelsize": 10.5,
            "axes.labelcolor": PALETTE["text_dark"],
            "xtick.color": PALETTE["text_muted"],
            "ytick.color": PALETTE["text_muted"],
            "grid.color": "#E2E8F0",
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def load_enriched_data(file_path: Path) -> pd.DataFrame:
    """Carga el dataset enriquecido generado en la Fase 3."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"No se encontró {file_path}. Ejecuta primero `python src/03_anomalies_pareto.py`."
        )
    df = pd.read_csv(file_path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    return df


def plot_traffic_bottleneck(df: pd.DataFrame, output_dir: Path) -> None:
    """Gráfico 1: Tasa de retraso y volumen operativo por condición de tráfico."""
    summary = (
        df.groupby("Traffic_Status", as_index=False)
        .agg(
            Total_Ops=("Asset_ID", "count"),
            Delayed_Ops=("Logistics_Delay", "sum"),
            Delay_Rate=("Logistics_Delay", lambda x: x.mean() * 100),
        )
        .sort_values("Delay_Rate", ascending=False)
    )

    global_rate = df["Logistics_Delay"].mean() * 100
    colors = [
        PALETTE["crimson"] if status == "Heavy" else PALETTE["slate"]
        for status in summary["Traffic_Status"]
    ]

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    bars = ax.bar(
        summary["Traffic_Status"],
        summary["Delay_Rate"],
        color=colors,
        width=0.52,
        edgecolor="none",
        zorder=3,
    )

    # Línea de referencia del promedio global
    ax.axhline(
        global_rate,
        color=PALETTE["navy"],
        linestyle="--",
        linewidth=1.4,
        zorder=4,
        label=f"Tasa Global de Retraso ({global_rate:.1f}%)",
    )

    # Anotaciones directas sobre las barras
    for bar, (_, row) in zip(bars, summary.iterrows()):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 2.5,
            f"{height:.1f}%\n({int(row['Delayed_Ops'])}/{int(row['Total_Ops'])} ops)",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=PALETTE["crimson"] if row["Traffic_Status"] == "Heavy" else PALETTE["text_dark"],
        )

    ax.set_ylim(0, 118)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_ylabel("Tasa de Retraso (%)")
    ax.set_xlabel("Condición de Tráfico Reportada")
    ax.set_title(
        "Cuello de Botella Crítico: El Tráfico 'Heavy' Genera 100% de Retrasos",
        loc="left",
        pad=22,
    )
    plt.suptitle(
        "Prueba Chi-Cuadrado: χ² = 372.62, p < 0.0001 | V de Cramér = 0.610 (Asociación Fuerte)",
        x=0.125,
        y=0.91,
        ha="left",
        fontsize=9.5,
        color=PALETTE["text_muted"],
    )

    sns.despine(top=True, right=True)
    ax.legend(frameon=True, facecolor="white", edgecolor="#CBD5E1", loc="upper right")
    plt.tight_layout(rect=[0, 0, 1, 0.89])

    out_path = output_dir / "01_traffic_bottleneck_analysis.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Gráfico guardado: %s", out_path.name)


def plot_pareto_financial_exposure(df: pd.DataFrame, output_dir: Path) -> None:
    """Gráfico 2: Diagrama de Pareto (80/20) de exposición financiera por causa de retraso."""
    delayed = df[df["Logistics_Delay"] == 1]
    pareto = (
        delayed.groupby("Logistics_Delay_Reason", as_index=False)["User_Transaction_Amount"]
        .sum()
        .rename(columns={"User_Transaction_Amount": "Exposure_USD"})
        .sort_values("Exposure_USD", ascending=False)
        .reset_index(drop=True)
    )

    total_exp = pareto["Exposure_USD"].sum()
    pareto["Pct"] = (pareto["Exposure_USD"] / total_exp) * 100
    pareto["Cum_Pct"] = pareto["Pct"].cumsum()

    bar_colors = [
        PALETTE["amber"] if reason == "Unknown" else PALETTE["navy"]
        for reason in pareto["Logistics_Delay_Reason"]
    ]

    fig, ax1 = plt.subplots(figsize=(10, 5.8))
    bars = ax1.bar(
        pareto["Logistics_Delay_Reason"],
        pareto["Exposure_USD"],
        color=bar_colors,
        width=0.52,
        zorder=3,
    )

    ax1.set_ylabel("Exposición Financiera en Riesgo (USD)", color=PALETTE["navy"])
    ax1.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
    ax1.set_ylim(0, pareto["Exposure_USD"].max() * 1.25)

    for bar, (_, row) in zip(bars, pareto.iterrows()):
        h = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 900,
            f"${h:,.0f}\n({row['Pct']:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color=PALETTE["amber"] if row["Logistics_Delay_Reason"] == "Unknown" else PALETTE["text_dark"],
        )

    # Eje secundario para curva acumulada de Pareto
    ax2 = ax1.twinx()
    ax2.plot(
        pareto["Logistics_Delay_Reason"],
        pareto["Cum_Pct"],
        color=PALETTE["crimson"],
        marker="o",
        linewidth=2.3,
        markersize=7,
        zorder=5,
        label="% Acumulado",
    )
    ax2.axhline(80, color=PALETTE["crimson"], linestyle="--", linewidth=1.1, alpha=0.8)
    ax2.text(
        len(pareto) - 0.55,
        81.8,
        "Umbral Pareto 80%",
        color=PALETTE["crimson"],
        fontsize=8.5,
        fontweight="bold",
        ha="right",
    )

    for idx, row in pareto.iterrows():
        ax2.annotate(
            f"{row['Cum_Pct']:.1f}%",
            (idx, row["Cum_Pct"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=9,
            fontweight="bold",
            color=PALETTE["crimson"],
        )

    ax2.set_ylim(0, 112)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax2.set_ylabel("Porcentaje Acumulado (%)", color=PALETTE["crimson"])
    ax2.grid(False)

    ax1.set_title(
        f"Pareto de Exposición Financiera por Causa Raíz (Total en Riesgo: ${total_exp:,.0f})",
        loc="left",
        pad=22,
    )
    plt.suptitle(
        "Hallazgo de Gobernanza: 'Unknown' concentra el 25.3% ($42,983) sin trazabilidad operativa",
        x=0.125,
        y=0.91,
        ha="left",
        fontsize=9.5,
        color=PALETTE["text_muted"],
    )

    sns.despine(top=True, right=False)
    plt.tight_layout(rect=[0, 0, 1, 0.89])

    out_path = output_dir / "02_pareto_financial_exposure.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Gráfico guardado: %s", out_path.name)


def plot_asset_traffic_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    """Gráfico 3: Heatmap de tasa de retraso (%) por Activo (Truck_1..10) y Tráfico."""
    pivot_delay = (
        df.pivot_table(
            index="Asset_ID",
            columns="Traffic_Status",
            values="Logistics_Delay",
            aggfunc=lambda x: x.mean() * 100,
        )
        .reindex(columns=["Heavy", "Detour", "Clear"])
    )
    # Ordenar activos por su tasa promedio global de retraso
    asset_order = (
        df.groupby("Asset_ID")["Logistics_Delay"].mean().sort_values(ascending=False).index
    )
    pivot_delay = pivot_delay.loc[asset_order]

    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    sns.heatmap(
        pivot_delay,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        cbar_kws={"label": "Tasa de Retraso (%)", "format": "%.0f%%"},
        linewidths=1.2,
        linecolor="white",
        vmin=20,
        vmax=100,
        ax=ax,
        annot_kws={"size": 10, "weight": "bold"},
    )

    ax.set_title(
        "Matriz de Vulnerabilidad Operativa: Tasa de Retraso (%) por Activo y Tráfico",
        loc="left",
        pad=22,
    )
    plt.suptitle(
        "Activos ordenados por tasa global de fallo | Saturación uniforme al 100% bajo tráfico Heavy",
        x=0.125,
        y=0.91,
        ha="left",
        fontsize=9.5,
        color=PALETTE["text_muted"],
    )
    ax.set_xlabel("Condición de Tráfico")
    ax.set_ylabel("Identificador de Activo (Flota)")
    ax.tick_params(axis="y", rotation=0)

    plt.tight_layout(rect=[0, 0, 1, 0.89])
    out_path = output_dir / "03_asset_risk_heatmap.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Gráfico guardado: %s", out_path.name)


def plot_financial_anomaly_quadrants(df: pd.DataFrame, output_dir: Path) -> None:
    """Gráfico 4: Dispersión de Monto Transaccionado vs. Tiempo de Espera y Cuadrantes Q3."""
    q3_amount = df["User_Transaction_Amount"].quantile(0.75)
    q3_wait = df["Waiting_Time"].quantile(0.75)

    # Clasificar cada operación en 3 grupos visuales claros
    conditions = [
        df["Critical_Financial_Delay"] == 1,
        df["Logistics_Delay"] == 1,
    ]
    choices = [
        "Anomalía Crítica: Retraso + Alto Valor ≥ Q3 (145 ops | $65,442)",
        "Retraso Estándar: Valor < Q3 (421 ops)",
    ]
    df_plot = df.copy()
    df_plot["Segmento_Riesgo"] = np.select(
        conditions, choices, default="Operación a Tiempo (434 ops)"
    )

    color_map = {
        "Anomalía Crítica: Retraso + Alto Valor ≥ Q3 (145 ops | $65,442)": PALETTE["crimson"],
        "Retraso Estándar: Valor < Q3 (421 ops)": PALETTE["slate"],
        "Operación a Tiempo (434 ops)": PALETTE["light_gray"],
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    for segment in [
        "Operación a Tiempo (434 ops)",
        "Retraso Estándar: Valor < Q3 (421 ops)",
        "Anomalía Crítica: Retraso + Alto Valor ≥ Q3 (145 ops | $65,442)",
    ]:
        subset = df_plot[df_plot["Segmento_Riesgo"] == segment]
        ax.scatter(
            subset["Waiting_Time"],
            subset["User_Transaction_Amount"],
            c=color_map[segment],
            label=segment,
            alpha=0.88 if "Anomalía" in segment else 0.65,
            s=48 if "Anomalía" in segment else 32,
            edgecolors="white" if "Anomalía" in segment else "none",
            linewidth=0.5,
            zorder=4 if "Anomalía" in segment else 2,
        )

    # Líneas de corte Q3 (Percentil 75)
    ax.axhline(q3_amount, color=PALETTE["navy"], linestyle="--", linewidth=1.2, zorder=5)
    ax.axvline(q3_wait, color=PALETTE["amber"], linestyle="--", linewidth=1.2, zorder=5)

    # Sombrear cuadrante superior (Alto Valor >= Q3)
    ax.axhspan(q3_amount, 520, color=PALETTE["crimson"], alpha=0.05, zorder=1)

    ax.text(
        10.5,
        q3_amount + 6,
        f"Umbral Alto Valor Q3 = ${q3_amount:.2f}",
        color=PALETTE["navy"],
        fontsize=9,
        fontweight="bold",
    )
    ax.text(
        q3_wait + 0.6,
        108,
        f"Umbral Espera Q3 = {q3_wait:.0f} min",
        color=PALETTE["amber"],
        fontsize=9,
        fontweight="bold",
    )

    ax.set_ylim(90, 520)
    ax.set_xlim(8, 62)
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
    ax.set_xlabel("Tiempo de Espera Operativo (Minutos)")
    ax.set_ylabel("Monto de Transacción del Cliente (USD)")
    ax.set_title(
        "Segmentación de Anomalías Compuestas: Exposición Financiera vs. Espera",
        loc="left",
        pad=22,
    )
    plt.suptitle(
        "145 operaciones críticas superan el percentil 75 de facturación ($397) y sufren retraso confirmado",
        x=0.125,
        y=0.91,
        ha="left",
        fontsize=9.5,
        color=PALETTE["text_muted"],
    )

    sns.despine(top=True, right=True)
    ax.legend(
        frameon=True,
        facecolor="white",
        edgecolor="#CBD5E1",
        loc="lower left",
        fontsize=8.8,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.89])

    out_path = output_dir / "04_financial_anomaly_quadrants.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Gráfico guardado: %s", out_path.name)


def main() -> None:
    """Orquesta la generación de todas las figuras ejecutivas (Fase 4)."""
    try:
        configure_style()
        df = load_enriched_data(ENRICHED_DATA_PATH)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

        plot_traffic_bottleneck(df, FIGURES_DIR)
        plot_pareto_financial_exposure(df, FIGURES_DIR)
        plot_asset_traffic_heatmap(df, FIGURES_DIR)
        plot_financial_anomaly_quadrants(df, FIGURES_DIR)

        logger.info("Fase 4 completada: 4 visualizaciones ejecutivas exportadas a 300 DPI.")

    except Exception as exc:
        logger.error("Error durante la generación de gráficos (Fase 4): %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()