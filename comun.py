"""Piezas compartidas por las dos implementaciones (sin framework y LangGraph).

Aquí NO hay lógica de agentes: solo la base de datos, el dataset, la evaluación y las trazas,
para que la comparación entre implementaciones sea justa.
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

RAIZ = Path(__file__).parent
RUTA_BD = RAIZ / "datos" / "tienda.db"

# --- Modelos por rol -------------------------------------------------------------------
MODELO_RAPIDO = "qwen2.5-coder:1.5b"   # router y preguntas simples: barato y rápido
MODELO_FUERTE = "qwen2.5-coder:3b"     # preguntas complejas y corrector
OPCIONES_LLM = {"temperature": 0, "num_ctx": 2048, "seed": 7}  # 2048 basta: los prompts usan < 700 tokens

# Capas en GPU por modelo, ajustadas a una GTX 1650 Ti (4 GB) para que los DOS modelos queden
# cargados a la vez. Si Ollama tuviera que descargar uno para cargar el otro, cada cambio de
# modelo costaría ~10 s y el router sería más lento que el agente único.
CAPAS_GPU = {MODELO_RAPIDO: 99, MODELO_FUERTE: 32}


def opciones(modelo: str, **extra) -> dict:
    return {**OPCIONES_LLM, "num_gpu": CAPAS_GPU[modelo], **extra}

# Precios de referencia (USD por millón de tokens) para traducir tokens a dinero.
# Con Ollama el costo real es $0; estos valores son un SUPUESTO (proporcionales al tamaño)
# para comparar con lo que costarían modelos equivalentes en una API.
PRECIOS = {
    MODELO_RAPIDO: {"entrada": 0.03, "salida": 0.06},
    MODELO_FUERTE: {"entrada": 0.06, "salida": 0.12},
}

# --- Esquema que ven los agentes -------------------------------------------------------
ESQUEMA = """\
Tablas SQLite:
categorias(id, nombre)
productos(id, nombre, categoria_id -> categorias.id, precio, stock)
clientes(id, nombre, ciudad, segmento, fecha_registro)
pedidos(id, cliente_id -> clientes.id, fecha, estado)
detalle_pedido(pedido_id -> pedidos.id, producto_id -> productos.id, cantidad, precio_unitario)

Valores: estado IN ('entregado','pendiente','cancelado'); segmento IN ('normal','premium');
ciudad con tilde (ej. 'Bogotá', 'Medellín'); categorias.nombre con tilde (ej. 'Electrónica');
fechas texto 'YYYY-MM-DD'; precios en pesos colombianos (enteros)."""

ESQUEMA_CORTO = ("categorias(id,nombre); productos(id,nombre,categoria_id,precio,stock); "
                 "clientes(id,nombre,ciudad,segmento,fecha_registro); pedidos(id,cliente_id,fecha,estado); "
                 "detalle_pedido(pedido_id,producto_id,cantidad,precio_unitario)")


# --- Dataset ---------------------------------------------------------------------------
def cargar_preguntas() -> list[dict]:
    return json.loads((RAIZ / "datos" / "preguntas.json").read_text(encoding="utf-8"))


def cargar_preguntas_ruido() -> list[dict]:
    """Variantes con errores de ortografía y lenguaje informal; usan el SQL de su pregunta base."""
    base = {p["id"]: p for p in cargar_preguntas()}
    ruido = json.loads((RAIZ / "datos" / "preguntas_ruido.json").read_text(encoding="utf-8"))
    return [{**r, "sql": base[r["base"]]["sql"]} for r in ruido]


# --- Herramienta: ejecutar SQL ---------------------------------------------------------
def ejecutar_sql(sql: str, limite_filas: int = 200) -> tuple[list[tuple] | None, str | None]:
    """Ejecuta en modo solo lectura. Devuelve (filas, error)."""
    if not sql or not sql.strip():
        return None, "No se generó ninguna consulta SQL."
    con = sqlite3.connect(f"file:{RUTA_BD}?mode=ro", uri=True)
    try:
        filas = con.execute(sql).fetchmany(limite_filas)
        return filas, None
    except Exception as e:  # noqa: BLE001 - el error se devuelve al agente como retroalimentación
        return None, f"{type(e).__name__}: {e}"
    finally:
        con.close()


def extraer_sql(texto: str) -> str:
    """Saca la consulta de la respuesta del modelo (bloque ```sql``` o texto plano)."""
    bloque = re.search(r"```(?:sql|sqlite)?\s*(.*?)```", texto, re.S | re.I)
    sql = bloque.group(1) if bloque else texto
    inicio = re.search(r"\b(SELECT|WITH)\b", sql, re.I)
    if not inicio:
        return ""
    sql = sql[inicio.start():]
    return sql.split(";")[0].strip()


# --- Evaluación: exactitud por ejecución -----------------------------------------------
def _norm(v):
    if isinstance(v, (int, float)):
        return round(float(v), 2)
    if isinstance(v, str):
        s = v.strip()
        try:
            return round(float(s), 2)
        except ValueError:
            return s.lower()
    return v


def resultados_iguales(pred: list[tuple] | None, oro: list[tuple]) -> bool:
    """Correcta si tiene las mismas filas y cada columna esperada aparece en la predicción.

    Es tolerante a columnas extra (ej. devolver nombre y total cuando solo se pidió el nombre)
    y al orden de filas y columnas.
    """
    if pred is None or len(pred) != len(oro):
        return False
    if not oro:
        return True
    cols_pred = [sorted(map(_norm, col), key=repr) for col in zip(*pred)] if pred else []
    for col in zip(*oro):
        if sorted(map(_norm, col), key=repr) not in cols_pred:
            return False
    return True


def es_correcta(sql_pred: str, sql_oro: str) -> bool:
    filas_oro, _ = ejecutar_sql(sql_oro)
    filas_pred, err = ejecutar_sql(sql_pred)
    return err is None and resultados_iguales(filas_pred, filas_oro)


# --- Trazas ----------------------------------------------------------------------------
@dataclass
class Llamada:
    agente: str          # rol: generador, router, corrector...
    modelo: str
    tokens_entrada: int
    tokens_salida: int
    segundos: float
    inicio: float = 0.0  # relativo al inicio de la tarea (para el Gantt)


@dataclass
class Traza:
    llamadas: list[Llamada] = field(default_factory=list)
    t0: float = field(default_factory=time.perf_counter)

    def registrar(self, agente, modelo, tok_in, tok_out, inicio_abs, fin_abs):
        self.llamadas.append(Llamada(agente, modelo, tok_in, tok_out, fin_abs - inicio_abs, inicio_abs - self.t0))


def costo_usd(modelo: str, tok_in: int, tok_out: int) -> float:
    p = PRECIOS.get(modelo, {"entrada": 0, "salida": 0})
    return (tok_in * p["entrada"] + tok_out * p["salida"]) / 1e6


ROLES_COORDINACION = {"router", "evaluador", "corrector"}


def resumen_tarea(traza: Traza) -> dict:
    """Totales de una tarea. La coordinación es todo lo que NO es la primera generación de SQL."""
    tok = sum(l.tokens_entrada + l.tokens_salida for l in traza.llamadas)
    tok_coord = sum(l.tokens_entrada + l.tokens_salida for l in traza.llamadas if l.agente in ROLES_COORDINACION)
    seg_coord = sum(l.segundos for l in traza.llamadas if l.agente in ROLES_COORDINACION)
    return {
        "llamadas_llm": len(traza.llamadas),
        "tokens_entrada": sum(l.tokens_entrada for l in traza.llamadas),
        "tokens_salida": sum(l.tokens_salida for l in traza.llamadas),
        "tokens_total": tok,
        "tokens_coordinacion": tok_coord,
        "seg_llm": sum(l.segundos for l in traza.llamadas),
        "seg_coordinacion": seg_coord,
        "costo_usd": sum(costo_usd(l.modelo, l.tokens_entrada, l.tokens_salida) for l in traza.llamadas),
    }
