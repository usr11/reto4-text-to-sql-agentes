"""Text-to-SQL multiagente CON framework: LangGraph + ChatOllama.

Mismos agentes, modelos, prompts y patrones que `sin_framework.py`, pero el flujo se declara como un grafo:

    START ─▶ [router] ─┬─▶ generador ─▶ evaluador ─┬─▶ END
                       │                   ▲       │
                       └───(sin router)    └─ corrector ◀┘ (si hay problema y quedan intentos)

  - Router            → arista condicional desde START (o nodo `router` si está activo).
  - Evaluador-optimizador → ciclo evaluador ⇄ corrector con contador de intentos en el estado.
Las trazas se toman con un callback de LangChain (lo estándar en el ecosistema; LangSmith las recoge solo).
"""
from __future__ import annotations

import time
from typing import Literal, TypedDict
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from comun import CAPAS_GPU, MODELO_FUERTE, MODELO_RAPIDO, OPCIONES_LLM, Traza, extraer_sql
from sin_framework import (MAX_CORRECCIONES, MODELO_POR_RUTA, PROMPT_CORRECTOR, PROMPT_ROUTER, PROMPTS_GENERADOR,
                           TOKENS_ROUTER, TOKENS_SQL,
                           evaluador, leer_ruta, mensaje_corrector)


# --- Modelos por rol -------------------------------------------------------------------
def modelo(nombre: str, agente: str, num_predict: int):
    llm = ChatOllama(model=nombre, temperature=OPCIONES_LLM["temperature"], num_ctx=OPCIONES_LLM["num_ctx"],
                     seed=OPCIONES_LLM["seed"], num_gpu=CAPAS_GPU[nombre], num_predict=num_predict,
                     keep_alive="30m")
    return llm.with_config(metadata={"agente": agente, "modelo": nombre})


LLM_ROUTER = modelo(MODELO_RAPIDO, "router", TOKENS_ROUTER)
LLM_GEN = {MODELO_RAPIDO: modelo(MODELO_RAPIDO, "generador", TOKENS_SQL), MODELO_FUERTE: modelo(MODELO_FUERTE, "generador", TOKENS_SQL)}
LLM_CORRECTOR = modelo(MODELO_FUERTE, "corrector", TOKENS_SQL)


# --- Trazas con callback ---------------------------------------------------------------
class Trazador(BaseCallbackHandler):
    """Guarda una fila por llamada al modelo en la `Traza` de la tarea."""

    def __init__(self, traza: Traza):
        self.traza, self._abiertas = traza, {}

    def on_chat_model_start(self, serialized, messages, *, run_id: UUID, metadata=None, **kw):
        self._abiertas[run_id] = (time.perf_counter(), metadata or {})

    def on_llm_end(self, response, *, run_id: UUID, **kw):
        t_ini, meta = self._abiertas.pop(run_id, (time.perf_counter(), {}))
        uso = getattr(response.generations[0][0].message, "usage_metadata", None) or {}
        self.traza.registrar(meta.get("agente", "?"), meta.get("modelo", "?"),
                             uso.get("input_tokens", 0), uso.get("output_tokens", 0), t_ini, time.perf_counter())


# --- Estado compartido -----------------------------------------------------------------
class Estado(TypedDict, total=False):
    pregunta: str
    modelo: str
    ruta: str | None
    sql: str
    problema: str | None
    correcciones: int


# --- Nodos -----------------------------------------------------------------------------
def nodo_router(estado: Estado) -> Estado:
    ruta = leer_ruta(LLM_ROUTER.invoke([SystemMessage(PROMPT_ROUTER), HumanMessage(estado["pregunta"])]).content)
    return {"ruta": ruta, "modelo": MODELO_POR_RUTA[ruta]}


def nodo_generador(estado: Estado) -> Estado:
    prompt = PROMPTS_GENERADOR[estado.get("ruta")]
    r = LLM_GEN[estado["modelo"]].invoke([SystemMessage(prompt), HumanMessage(estado["pregunta"])])
    return {"sql": extraer_sql(r.content), "correcciones": 0}


def nodo_evaluador(estado: Estado) -> Estado:
    return {"problema": evaluador(estado["sql"])}


def nodo_corrector(estado: Estado) -> Estado:
    user = mensaje_corrector(estado["pregunta"], estado["sql"], estado["problema"])
    r = LLM_CORRECTOR.invoke([SystemMessage(PROMPT_CORRECTOR), HumanMessage(user)])
    return {"sql": extraer_sql(r.content), "correcciones": estado["correcciones"] + 1}


def decidir_tras_evaluar(estado: Estado) -> Literal["corrector", "__end__"]:
    if estado.get("problema") and estado["correcciones"] < MAX_CORRECCIONES:
        return "corrector"
    return END


# --- Construcción del grafo ------------------------------------------------------------
def construir_grafo(usar_router: bool, usar_evaluador: bool):
    g = StateGraph(Estado)
    g.add_node("generador", nodo_generador)
    if usar_router:
        g.add_node("router", nodo_router)
        g.add_edge(START, "router")
        g.add_edge("router", "generador")
    else:
        g.add_edge(START, "generador")
    if usar_evaluador:
        g.add_node("evaluador", nodo_evaluador)
        g.add_node("corrector", nodo_corrector)
        g.add_edge("generador", "evaluador")
        g.add_conditional_edges("evaluador", decidir_tras_evaluar)
        g.add_edge("corrector", "evaluador")
    else:
        g.add_edge("generador", END)
    return g.compile()


_GRAFOS: dict = {}


def resolver(pregunta: str, usar_router: bool, usar_evaluador: bool, modelo: str = MODELO_FUERTE) -> dict:
    clave = (usar_router, usar_evaluador)
    if clave not in _GRAFOS:
        _GRAFOS[clave] = construir_grafo(*clave)
    traza = Traza()
    final = _GRAFOS[clave].invoke(
        {"pregunta": pregunta, "modelo": modelo, "ruta": None},
        config={"callbacks": [Trazador(traza)], "run_name": "tarea_sql"},
    )
    return {"sql": final["sql"], "ruta": final.get("ruta"), "correcciones": final.get("correcciones", 0), "traza": traza}


if __name__ == "__main__":
    print(construir_grafo(True, True).get_graph().draw_mermaid())
    r = resolver("¿Qué ciudad tiene el mayor número de pedidos?", usar_router=True, usar_evaluador=True)
    print(r["ruta"], r["correcciones"], r["sql"])
    for l in r["traza"].llamadas:
        print(l)
