"""Demo local para mostrar el sistema en vivo.

Correr:  uv run streamlit run demo.py
"""
from __future__ import annotations

import sqlite3
import time

import pandas as pd
import streamlit as st

import con_langgraph
import sin_framework
from comun import RUTA_BD, cargar_preguntas, cargar_preguntas_ruido, costo_usd, ejecutar_sql, es_correcta

st.set_page_config(page_title="Text-to-SQL con agentes locales", page_icon="🗄️", layout="wide")

IMPLEMENTACIONES = {"Sin framework": sin_framework.resolver, "LangGraph": con_langgraph.resolver}
DESCRIPCION = {
    "A · único 1.5b": "Un solo agente con el modelo pequeño (1.5B) y un prompt general.",
    "B · único 3b": "Un solo agente con el modelo grande (3B) y un prompt general.",
    "C · router": "El router (1.5B) nombra las tablas: 1 tabla → 1.5B con prompt corto; 2+ → 3B con ejemplos.",
    "D · evaluador-optimizador": "Cascada: genera el 1.5B, el evaluador ejecuta el SQL y solo si falla corrige el 3B.",
    "E · router + evaluador": "Router + evaluador + corrector: el flujo completo.",
}
EJEMPLOS = {p["pregunta"]: p for p in cargar_preguntas() + cargar_preguntas_ruido()}

st.title("🗄️ Text-to-SQL con agentes locales")
st.caption("Tienda Andina (ficticia) · modelos qwen2.5-coder 1.5B y 3B en Ollama · patrones: router y evaluador-optimizador")

tab_demo, tab_metricas, tab_datos = st.tabs(["Probar en vivo", "Métricas del experimento", "Base de datos"])

# --- Probar en vivo --------------------------------------------------------------------
with tab_demo:
    col_izq, col_der = st.columns([1, 2], gap="large")
    with col_izq:
        config = st.radio("Arquitectura", list(sin_framework.CONFIGURACIONES), index=3)
        st.info(DESCRIPCION[config])
        impl = st.radio("Implementación", list(IMPLEMENTACIONES), horizontal=True)
        ejemplo = st.selectbox("Pregunta de ejemplo (o escribe la tuya abajo)", ["—"] + list(EJEMPLOS))
        pregunta = st.text_area("Pregunta", value="" if ejemplo == "—" else ejemplo, height=80,
                                placeholder="¿Qué ciudad tiene el mayor número de pedidos?")
        correr = st.button("Preguntar", type="primary", use_container_width=True, disabled=not pregunta.strip())

    with col_der:
        if correr:
            with st.spinner("Los agentes están trabajando…"):
                t0 = time.perf_counter()
                try:
                    r = IMPLEMENTACIONES[impl](pregunta.strip(), **sin_framework.CONFIGURACIONES[config])
                except Exception as e:  # noqa: BLE001
                    st.error(f"El sistema falló: {e}. ¿Está corriendo Ollama?")
                    st.stop()
                latencia = time.perf_counter() - t0
            filas, error = ejecutar_sql(r["sql"])
            llamadas = r["traza"].llamadas
            tokens = sum(l.tokens_entrada + l.tokens_salida for l in llamadas)
            costo = sum(costo_usd(l.modelo, l.tokens_entrada, l.tokens_salida) for l in llamadas)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Latencia", f"{latencia:.2f} s")
            m2.metric("Llamadas al LLM", len(llamadas))
            m3.metric("Tokens", f"{tokens:,}")
            m4.metric("Costo equivalente", f"${costo * 1e6:.1f} / millón")

            pasos = []
            if r["ruta"]:
                pasos.append(f"**Router:** pregunta **{r['ruta']}**")
            pasos.append(f"**Correcciones del evaluador:** {r['correcciones']}")
            st.markdown(" · ".join(pasos))

            ref = EJEMPLOS.get(pregunta.strip())
            if ref:
                if es_correcta(r["sql"], ref["sql"]):
                    st.success("✅ Correcta: el resultado coincide con la consulta de referencia.")
                else:
                    st.error("❌ Incorrecta: el resultado no coincide con la consulta de referencia.")
                    with st.expander("Ver SQL de referencia"):
                        st.code(ref["sql"], language="sql")

            st.subheader("SQL generado")
            st.code(r["sql"] or "(no generó SQL)", language="sql")
            st.subheader("Resultado")
            if error:
                st.error(error)
            else:
                st.dataframe(pd.DataFrame(filas), use_container_width=True, hide_index=True)

            st.subheader("Traza: qué hizo cada agente")
            st.dataframe(pd.DataFrame([{
                "agente": l.agente, "modelo": l.modelo, "tokens entrada": l.tokens_entrada,
                "tokens salida": l.tokens_salida, "inicio (s)": round(l.inicio, 2), "duración (s)": round(l.segundos, 2),
            } for l in llamadas]), use_container_width=True, hide_index=True)
        else:
            st.markdown("#### Cómo funciona")
            st.markdown(
                "1. **Router** (1.5B): nombra las tablas que necesita la pregunta → simple o compleja.\n"
                "2. **Generador**: el 1.5B escribe el SQL de las simples; el 3B, el de las complejas.\n"
                "3. **Evaluador** (sin LLM, 0 tokens): ejecuta el SQL. ¿Error, 0 filas o solo NULL?\n"
                "4. **Corrector** (3B): solo si el evaluador encontró un problema; reescribe con el error (máx. 2 veces)."
            )
            st.markdown("Escoge una arquitectura y una pregunta, y presiona **Preguntar**.")

# --- Métricas del experimento ----------------------------------------------------------
with tab_metricas:
    try:
        m = pd.read_csv("resultados/metricas.csv")
    except FileNotFoundError:
        st.warning("Aún no hay resultados. Corre `uv run python experimento.py` y `uv run python analisis.py`.")
    else:
        st.markdown("Datos de 360 tareas: 36 preguntas × 5 arquitecturas × 2 implementaciones.")
        sf = m[m["implementación"] == "sin framework"].set_index("configuración")
        c1, c2, c3 = st.columns(3)
        mejor_d = sf.loc["D · evaluador-optimizador"]
        base = sf.loc["B · único 3b"]
        c1.metric("Cascada D: calidad", f"{mejor_d['exactitud %']:.1f} %",
                  f"{mejor_d['exactitud %'] - base['exactitud %']:+.1f} pts vs 3B solo")
        c2.metric("Cascada D: costo", f"${mejor_d['USD por millón de tareas*']:.1f} / millón",
                  f"{100 * (mejor_d['USD por millón de tareas*'] / base['USD por millón de tareas*'] - 1):+.0f} % vs 3B solo",
                  delta_color="inverse")
        c3.metric("Cascada D: latencia p95", f"{mejor_d['latencia p95 s']:.2f} s",
                  f"{100 * (mejor_d['latencia p95 s'] / base['latencia p95 s'] - 1):+.0f} % vs 3B solo",
                  delta_color="inverse")
        grupos = {
            "Calidad": ["exactitud %", "exactitud simples %", "exactitud complejas %"],
            "Costo": ["USD por millón de tareas*", "tokens/tarea", "llamadas LLM/tarea"],
            "Latencia": ["latencia media s", "latencia p95 s", "overhead no-LLM ms"],
            "Robustez": ["exactitud con ruido %", "SQL inválido %", "caídas del sistema"],
            "Coordinación": ["% tokens coordinación", "% tiempo coordinación", "correcciones/tarea",
                             "acierto del router %"],
        }
        impl_sel = st.radio("Implementación", ["sin framework", "LangGraph"], horizontal=True, key="impl_m")
        tabla = m[m["implementación"] == impl_sel].set_index("configuración")
        for titulo, cols in grupos.items():
            st.markdown(f"**{titulo}**")
            st.dataframe(tabla[cols], use_container_width=True)
        st.caption("* Con Ollama el costo real es $0; el USD usa precios de referencia supuestos.")
        g1, g2 = st.columns(2)
        g1.image("resultados/pareto_latencia.png")
        g2.image("resultados/pareto_costo.png")
        st.image("resultados/calidad_detalle.png")

# --- Base de datos ---------------------------------------------------------------------
with tab_datos:
    con = sqlite3.connect(RUTA_BD)
    for t in ["categorias", "productos", "clientes", "pedidos", "detalle_pedido"]:
        n = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        with st.expander(f"{t} · {n} filas"):
            st.dataframe(pd.read_sql(f"SELECT * FROM {t} LIMIT 50", con), use_container_width=True, hide_index=True)
    con.close()
