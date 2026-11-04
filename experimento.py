"""Corre todas las configuraciones en las dos implementaciones y guarda una fila por tarea.

Uso:  uv run python experimento.py            (todo)
      uv run python experimento.py --rapido   (solo 4 preguntas, para probar)
Se puede interrumpir y relanzar: retoma desde las tareas que faltan.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import ollama
import pandas as pd
from dotenv import load_dotenv

load_dotenv()  # activa LangSmith si .env tiene LANGSMITH_TRACING=true y LANGSMITH_API_KEY

import con_langgraph
import sin_framework
from comun import (MODELO_FUERTE, MODELO_RAPIDO, cargar_preguntas, opciones, cargar_preguntas_ruido, ejecutar_sql,
                   es_correcta, resumen_tarea)

SALIDA = Path(__file__).parent / "resultados" / "tareas.csv"
IMPLEMENTACIONES = {"sin framework": sin_framework.resolver, "LangGraph": con_langgraph.resolver}


def calentar():
    """Carga ambos modelos en memoria para que la primera tarea no pague el tiempo de carga."""
    for m in (MODELO_RAPIDO, MODELO_FUERTE):
        ollama.chat(model=m, messages=[{"role": "user", "content": "SELECT 1"}], options=opciones(m, num_predict=1),
                    keep_alive="30m")


def correr_tarea(impl: str, config: str, p: dict, dataset: str) -> dict:
    fila = {"implementacion": impl, "config": config, "dataset": dataset, "id": p["id"], "nivel": p["nivel"]}
    t0 = time.perf_counter()
    try:
        r = IMPLEMENTACIONES[impl](p["pregunta"], **sin_framework.CONFIGURACIONES[config])
        latencia = time.perf_counter() - t0
        _, error = ejecutar_sql(r["sql"])
        fila.update(sql=r["sql"], ruta=r["ruta"], correcciones=r["correcciones"], latencia_s=latencia,
                    sql_valido=error is None, correcta=es_correcta(r["sql"], p["sql"]), fallo_sistema=False,
                    **resumen_tarea(r["traza"]))
    except Exception as e:  # el sistema se cayó: cuenta como fallo de robustez
        fila.update(sql="", latencia_s=time.perf_counter() - t0, sql_valido=False, correcta=False,
                    fallo_sistema=True, error_sistema=repr(e)[:200])
    return fila


def main(rapido: bool = False):
    preguntas = {"normal": cargar_preguntas(), "ruido": cargar_preguntas_ruido()}
    if rapido:
        preguntas = {"normal": preguntas["normal"][:2] + preguntas["normal"][12:14], "ruido": []}
    hechas = set()
    filas = []
    if SALIDA.exists() and not rapido:
        previas = pd.read_csv(SALIDA)
        filas = previas.to_dict("records")
        hechas = set(zip(previas.implementacion, previas.config, previas.dataset, previas.id))
    SALIDA.parent.mkdir(exist_ok=True)
    calentar()
    total = sum(len(v) for v in preguntas.values()) * len(IMPLEMENTACIONES) * len(sin_framework.CONFIGURACIONES)
    n = len(hechas)
    for impl in IMPLEMENTACIONES:
        for config in sin_framework.CONFIGURACIONES:
            for dataset, lista in preguntas.items():
                for p in lista:
                    if (impl, config, dataset, p["id"]) in hechas:
                        continue
                    fila = correr_tarea(impl, config, p, dataset)
                    filas.append(fila)
                    n += 1
                    print(f"[{n}/{total}] {impl:13} {config:26} {dataset:6} {p['id']} "
                          f"{'OK ' if fila['correcta'] else 'mal'} {fila['latencia_s']:.1f}s", flush=True)
                    if not rapido:
                        pd.DataFrame(filas).to_csv(SALIDA, index=False)
    return pd.DataFrame(filas)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true")
    df = main(ap.parse_args().rapido)
    print(df.groupby(["implementacion", "config"])[["correcta", "latencia_s", "tokens_total"]].mean().round(2))
