"""Text-to-SQL multiagente SIN framework: Python puro + cliente de Ollama.

Dos modelos locales asignados por rol: qwen2.5-coder:1.5b (rápido) y qwen2.5-coder:3b (fuerte).

Agentes:
  - router:       modelo rápido; nombra las tablas necesarias → 1 tabla = SIMPLE, varias = COMPLEJA.
  - generador:    escribe el SQL. Con router hay dos especialistas:
                    · simple   → modelo rápido + prompt corto (esquema compacto): menos tokens y tiempo.
                    · compleja → modelo fuerte + esquema completo + ejemplos de JOIN/GROUP BY/subconsultas.
  - evaluador:    ejecuta el SQL contra la base (herramienta) y revisa reglas. No usa LLM: cuesta 0 tokens.
  - corrector:    modelo fuerte; si el evaluador encuentra un problema, reescribe el SQL con el error como pista.

Patrones: router y evaluador-optimizador. Con los dos apagados queda el agente único (baseline).
"""
from __future__ import annotations

import re
import time

import ollama
from langsmith import traceable  # solo envía trazas si LANGSMITH_TRACING=true y hay API key

from comun import ESQUEMA, ESQUEMA_CORTO, MODELO_FUERTE, MODELO_RAPIDO, Traza, ejecutar_sql, extraer_sql, opciones

FORMATO = "Responde solo con la consulta dentro de un bloque ```sql```."

# Agente único: un prompt general para cualquier pregunta.
PROMPT_GENERADOR = f"""Eres un experto en SQLite. Escribe UNA sola consulta SQL que responda la pregunta.
Usa solo tablas y columnas del esquema y devuelve solo las columnas necesarias. {FORMATO}

{ESQUEMA}"""

# Especialista en preguntas simples: esquema compacto (menos tokens de entrada).
PROMPT_SIMPLE = f"""Eres un experto en SQLite. Escribe UNA consulta SQL corta. {FORMATO}
Tablas: {ESQUEMA_CORTO}
Valores: estado IN ('entregado','pendiente','cancelado'); segmento IN ('normal','premium');
ciudades con tilde ('Bogotá','Medellín'); fechas 'YYYY-MM-DD'."""

# Especialista en preguntas complejas: esquema completo + ejemplos de los patrones difíciles.
PROMPT_COMPLEJA = f"""Eres un experto en SQLite. Escribe UNA sola consulta SQL que responda la pregunta.
Une las tablas con JOIN por sus llaves, usa GROUP BY para totales por grupo y NOT IN para "nunca".
El dinero vendido es SUM(cantidad * precio_unitario) de detalle_pedido. {FORMATO}

{ESQUEMA}

Ejemplos:
P: Unidades vendidas por ciudad del cliente.
```sql
SELECT c.ciudad, SUM(d.cantidad) FROM detalle_pedido d JOIN pedidos p ON p.id = d.pedido_id
JOIN clientes c ON c.id = p.cliente_id GROUP BY c.ciudad
```
P: ¿Qué categorías no tienen ningún producto?
```sql
SELECT nombre FROM categorias WHERE id NOT IN (SELECT categoria_id FROM productos)
```
P: Nombre del producto con más dinero vendido en pedidos pendientes.
```sql
SELECT pr.nombre FROM detalle_pedido d JOIN productos pr ON pr.id = d.producto_id
JOIN pedidos p ON p.id = d.pedido_id WHERE p.estado = 'pendiente'
GROUP BY pr.id ORDER BY SUM(d.cantidad * d.precio_unitario) DESC LIMIT 1
```"""

# El router no clasifica directamente (un modelo de 1.5b casi siempre respondía "COMPLEJA"):
# nombra las tablas necesarias y una regla decide: 1 tabla → simple, 2 o más → compleja.
PROMPT_ROUTER = f"""Tablas: {ESQUEMA_CORTO}
Di qué tablas se necesitan para responder la pregunta. Responde solo los nombres de las tablas separados por coma.
Ejemplos:
¿Cuántos productos tienen stock 0? -> productos
¿Cuántos pedidos están pendientes? -> pedidos
¿Cuántos pedidos hizo cada ciudad? -> pedidos, clientes
Productos vendidos en pedidos cancelados -> productos, detalle_pedido, pedidos
¿Qué categorías no tienen ningún producto? -> categorias, productos"""
TABLAS = ("categorias", "productos", "clientes", "pedidos", "detalle_pedido")
TOKENS_ROUTER = 20

PROMPT_CORRECTOR = f"""Eres un experto en SQLite que corrige consultas. Te doy la pregunta, la consulta anterior
y el problema que encontró el evaluador. Escribe la consulta corregida. {FORMATO}

{ESQUEMA}"""

PROMPTS_GENERADOR = {None: PROMPT_GENERADOR, "simple": PROMPT_SIMPLE, "compleja": PROMPT_COMPLEJA}
MODELO_POR_RUTA = {"simple": MODELO_RAPIDO, "compleja": MODELO_FUERTE}
MAX_CORRECCIONES = 2
TOKENS_SQL = 160  # tope de salida: las consultas caben en <100 tokens; evita respuestas largas que disparan la latencia


# --- Cliente LLM: punto único para medir tokens y tiempo -------------------------------
def llamar_llm(traza: Traza, agente: str, modelo: str, system: str, user: str, **extra) -> str:
    t_ini = time.perf_counter()
    r = ollama.chat(
        model=modelo,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        options=opciones(modelo, **extra),
        keep_alive="30m",
    )
    traza.registrar(agente, modelo, r.prompt_eval_count or 0, r.eval_count or 0, t_ini, time.perf_counter())
    return r.message.content


# --- Agentes ---------------------------------------------------------------------------
def leer_ruta(salida: str) -> str:
    tablas = {t for t in TABLAS if re.search(rf"\b{t}\b", salida.lower())}
    return "simple" if len(tablas) <= 1 else "compleja"


@traceable(name="router")
def router(traza: Traza, pregunta: str) -> str:
    return leer_ruta(llamar_llm(traza, "router", MODELO_RAPIDO, PROMPT_ROUTER, pregunta, num_predict=TOKENS_ROUTER))


@traceable(name="generador")
def generador(traza: Traza, pregunta: str, ruta: str | None, modelo: str) -> str:
    return extraer_sql(llamar_llm(traza, "generador", modelo, PROMPTS_GENERADOR[ruta], pregunta, num_predict=TOKENS_SQL))


def evaluador(sql: str) -> str | None:
    """Ejecuta el SQL y devuelve el problema encontrado, o None si se acepta."""
    filas, error = ejecutar_sql(sql)
    if error:
        return f"La consulta falló al ejecutarse: {error}"
    if not filas:
        return "La consulta devolvió 0 filas. Revisa los filtros (tildes, mayúsculas, valores exactos de las columnas)."
    if all(v is None for v in filas[0]):
        return "La consulta devolvió solo NULL. Revisa los JOIN y los filtros."
    return None


def mensaje_corrector(pregunta: str, sql: str, problema: str) -> str:
    return f"Pregunta: {pregunta}\nConsulta anterior:\n```sql\n{sql}\n```\nProblema: {problema}"


@traceable(name="corrector")
def corrector(traza: Traza, pregunta: str, sql: str, problema: str) -> str:
    user = mensaje_corrector(pregunta, sql, problema)
    return extraer_sql(llamar_llm(traza, "corrector", MODELO_FUERTE, PROMPT_CORRECTOR, user, num_predict=TOKENS_SQL))


# --- Arquitectura ----------------------------------------------------------------------
@traceable(name="tarea_sql")
def resolver(pregunta: str, usar_router: bool, usar_evaluador: bool, modelo: str = MODELO_FUERTE) -> dict:
    traza = Traza()
    ruta = None
    if usar_router:
        ruta = router(traza, pregunta)
        modelo = MODELO_POR_RUTA[ruta]
    sql = generador(traza, pregunta, ruta, modelo)
    correcciones = 0
    if usar_evaluador:
        problema = evaluador(sql)
        while problema and correcciones < MAX_CORRECCIONES:
            sql = corrector(traza, pregunta, sql, problema)
            correcciones += 1
            problema = evaluador(sql)
    return {"sql": sql, "ruta": ruta, "correcciones": correcciones, "traza": traza}


CONFIGURACIONES = {
    "A · único 1.5b": dict(usar_router=False, usar_evaluador=False, modelo=MODELO_RAPIDO),
    "B · único 3b": dict(usar_router=False, usar_evaluador=False, modelo=MODELO_FUERTE),
    "C · router": dict(usar_router=True, usar_evaluador=False),
    # Cascada: genera el modelo barato; el evaluador solo escala al 3b (corrector) si hay un problema.
    "D · evaluador-optimizador": dict(usar_router=False, usar_evaluador=True, modelo=MODELO_RAPIDO),
    "E · router + evaluador": dict(usar_router=True, usar_evaluador=True),
}

if __name__ == "__main__":
    r = resolver("¿Qué ciudad tiene el mayor número de pedidos?", usar_router=True, usar_evaluador=True)
    print(r["ruta"], r["correcciones"], r["sql"])
    for l in r["traza"].llamadas:
        print(l)
