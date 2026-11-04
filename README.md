# Reto 4 · Arquitectura agéntica: Text-to-SQL con modelos locales

Un sistema que responde preguntas en español sobre las ventas de una tienda ficticia generando SQL y ejecutándolo.
Compara un **agente único** contra **multiagente** con dos patrones: **router** y **evaluador-optimizador**.
Cada arquitectura está implementada **sin framework** y **con LangGraph**, y se miden calidad, costo, latencia,
robustez y coordinación.

## Resultado principal

| | A · único 1.5b | B · único 3b | C · router | **D · cascada** | **E · router + evaluador** |
|---|---|---|---|---|---|
| Calidad (exactitud por ejecución) | 79.2 % | 87.5 % | 83.3 % | **87.5 %** | **91.7 %** |
| Costo (USD / millón de tareas\*) | 9.7 | 18.3 | 24.3 | **14.8** | 27.0 |
| Latencia media / p95 (s) | 0.52 / 0.96 | 1.12 / 2.17 | 1.00 / 2.36 | **1.00 / 1.71** | 1.15 / 2.57 |
| Robustez (exactitud con ruido) | 58 % | 100 % | 75 % | 75 % | 83 % |
| Coordinación (% tokens / % tiempo) | 0 / 0 | 0 / 0 | 38 / 10 | 20 / 46 | 43 / 23 |

\* Con Ollama el costo real es $0; los precios en USD son de referencia (supuestos) para comparar.

- **Recomendada por costo y latencia: D (cascada).** El 1.5b genera, un evaluador ejecuta el SQL y solo si falla
  escala al 3b. Tiene la misma calidad que el 3b solo, es 19 % más barata y su p95 es 21 % más rápido.
- **Recomendada por calidad: E (router + evaluador)**, con +4 puntos de calidad y casi la misma latencia.
- Con framework y sin framework los números son prácticamente iguales: LangGraph agrega solo 2–4 ms por tarea.

El análisis completo, con gráficas, está en [informe.ipynb](informe.ipynb). El documento con la explicación y los resultados está [aquí](https://claude.ai/artifact/Q2aEvC9g4F3Hh6PnsU22gz).

## Modelos

| Modelo | Rol |
|---|---|
| `qwen2.5-coder:1.5b` | router y generador de preguntas simples (rápido) |
| `qwen2.5-coder:3b` | generador de preguntas complejas y corrector |

Están ajustados para que los dos quepan a la vez en una GPU de 4 GB (`CAPAS_GPU` en [comun.py](comun.py)).
Con otra GPU, conviene revisar ese valor con `ollama ps`.

## Archivos

| Archivo | Qué contiene |
|---|---|
| [comun.py](comun.py) | esquema, ejecución de SQL, evaluación por ejecución, trazas y precios (compartido) |
| [sin_framework.py](sin_framework.py) | agentes y arquitecturas en Python puro + cliente `ollama` |
| [con_langgraph.py](con_langgraph.py) | las mismas arquitecturas como `StateGraph` de LangGraph |
| [experimento.py](experimento.py) | corre 5 configuraciones × 2 implementaciones × 36 preguntas |
| [analisis.py](analisis.py) | calcula las 5 métricas y genera las gráficas |
| [informe.ipynb](informe.ipynb) | informe con tablas, gráficas y conclusiones |
| [demo.py](demo.py) | página local (Streamlit) para probar las arquitecturas en vivo y ver las métricas |
| `datos/` | base SQLite, script que la crea, 24 preguntas con SQL de referencia y 12 con ruido |
| `resultados/` | `tareas.csv` (una fila por tarea), `metricas.csv` y gráficas. `tareas_router_v1.csv` y `tareas_sin_tope.csv` son corridas anteriores (router original y sin tope de tokens) |

## Cómo correrlo

```bash
uv sync
```

```bash
ollama pull qwen2.5-coder:1.5b && ollama pull qwen2.5-coder:3b
```

```bash
uv run python experimento.py
```

```bash
uv run python analisis.py
```

Demo en vivo para mostrar el sistema (abre http://localhost:8501):

```bash
uv run streamlit run demo.py
```

El experimento tarda unos 7 minutos en una GTX 1650 Ti. Para trazas en LangSmith, copiar `.env.example` a `.env`
y poner la API key.
