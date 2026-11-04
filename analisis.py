"""Convierte resultados/tareas.csv en las 5 métricas pedidas y genera las gráficas."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RES = Path(__file__).parent / "resultados"


def cargar() -> pd.DataFrame:
    df = pd.read_csv(RES / "tareas.csv")
    df["overhead_s"] = df["latencia_s"] - df["seg_llm"].fillna(0)  # tiempo fuera del LLM (código, framework, SQL)
    return df


def metricas(df: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for (impl, cfg), g in df.groupby(["implementacion", "config"], sort=False):
        n, r = g[g.dataset == "normal"], g[g.dataset == "ruido"]
        rutas = n.dropna(subset=["ruta"])
        filas.append({
            "implementación": impl, "configuración": cfg,
            # calidad
            "exactitud %": 100 * n.correcta.mean(),
            "exactitud simples %": 100 * n[n.nivel == "simple"].correcta.mean(),
            "exactitud complejas %": 100 * n[n.nivel == "compleja"].correcta.mean(),
            # costo
            "tokens/tarea": n.tokens_total.mean(),
            "llamadas LLM/tarea": n.llamadas_llm.mean(),
            "USD por millón de tareas*": 1e6 * n.costo_usd.mean(),
            # latencia
            "latencia media s": n.latencia_s.mean(),
            "latencia p95 s": n.latencia_s.quantile(0.95),
            "overhead no-LLM ms": 1000 * n.overhead_s.mean(),
            # robustez
            "exactitud con ruido %": 100 * r.correcta.mean() if len(r) else float("nan"),
            "SQL inválido %": 100 * (~g.sql_valido.astype(bool)).mean(),
            "caídas del sistema": int(g.fallo_sistema.sum()),
            # coordinación
            "% tokens coordinación": 100 * n.tokens_coordinacion.sum() / n.tokens_total.sum(),
            "% tiempo coordinación": 100 * n.seg_coordinacion.sum() / n.seg_llm.sum(),
            "correcciones/tarea": n.correcciones.mean(),
            "acierto del router %": 100 * (rutas.ruta == rutas.nivel).mean() if len(rutas) else float("nan"),
        })
    return pd.DataFrame(filas).round(2)


def graficas(df: pd.DataFrame, m: pd.DataFrame) -> list[Path]:
    rutas = []
    colores = {"sin framework": "#2f6db3", "LangGraph": "#d1772c"}
    for eje_x, nombre, etiqueta in [("latencia media s", "pareto_latencia.png", "Latencia media por tarea (s)"),
                                    ("USD por millón de tareas*", "pareto_costo.png", "Costo equivalente: USD por millón de tareas (precios de referencia)")]:
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        for impl, g in m.groupby("implementación"):
            ax.scatter(g[eje_x], g["exactitud %"], s=70, color=colores.get(impl), label=impl)
            if impl != "sin framework":
                continue  # una etiqueta por configuración: los dos puntos quedan casi encima
            for _, f in g.iterrows():
                ax.annotate(f["configuración"].split(" · ")[0], (f[eje_x], f["exactitud %"]),
                            textcoords="offset points", xytext=(6, 6), fontsize=10, weight="bold")
        ax.margins(x=0.25, y=0.12)
        ax.set_xlabel(etiqueta)
        ax.set_ylabel("Exactitud por ejecución (%)")
        ax.set_title("Calidad vs " + ("latencia" if "lat" in eje_x else "costo") + " (arriba-izquierda es mejor)")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right")
        ax.text(0.01, 0.99, "\n".join(m.configuración.unique()), transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="#ccc"))
        fig.tight_layout()
        fig.savefig(RES / nombre, dpi=130)
        plt.close(fig)
        rutas.append(RES / nombre)

    # Calidad: normal vs ruido, por configuración (solo sin framework: la lógica es idéntica)
    sf = m[m["implementación"] == "sin framework"].set_index("configuración")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    sf[["exactitud simples %", "exactitud complejas %", "exactitud con ruido %"]].plot.bar(ax=ax, rot=15)
    ax.set_ylabel("%")
    ax.set_title("Calidad por tipo de pregunta y con ruido (sin framework)")
    ax.set_ylim(0, 115)
    ax.legend(loc="upper center", ncol=3, fontsize=8.5, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RES / "calidad_detalle.png", dpi=130)
    plt.close(fig)
    rutas.append(RES / "calidad_detalle.png")
    return rutas


if __name__ == "__main__":
    d = cargar()
    m = metricas(d)
    m.to_csv(RES / "metricas.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(m.T)
    print(graficas(d, m))
