"""
Dashboard de segmentación de usuarios de streaming.

Consume la API (api/main.py) vía HTTP. Organizado en 3 vistas según
audiencia (ejecutiva, técnica, operativa), con un filtro global de
segmentos en la barra lateral que afecta a las tres pestañas.
"""
import time
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Segmentación de Usuarios — Streaming", layout="wide")

# --- Interpretación de negocio por cluster ---
INTERPRETACIONES = {
    0: "Usuarios habituales exploradores: sesiones frecuentes pero cortas, consumo moderado.",
    1: "Usuarios nuevos sensibles a precio: baja antigüedad, alto uso de promociones, bajo consumo.",
    2: "Usuarios premium leales: alto gasto, alta finalización, baja sensibilidad a promociones.",
}


@st.cache_data
def load_data():
    """Obtiene usuarios, centroides y métricas desde la API, esperando si no está lista."""
    url = "http://api:8000/dashboard-data"
    max_intentos = 15
    
    for intento in range(max_intentos):
        try:
            respuesta = requests.get(url)
            if respuesta.status_code == 200:
                payload = respuesta.json()
                usuarios = pd.DataFrame(payload["usuarios"])
                centroides = pd.DataFrame(payload["centroides"])
                evaluacion_k = pd.DataFrame(payload["evaluacion_k"])
                metricas = payload["metricas"]
                return usuarios, centroides, evaluacion_k, metricas
        except requests.exceptions.ConnectionError:
            st.sidebar.info(f"Esperando que la API se inicialice... (Intento {intento + 1}/{max_intentos})")
            time.sleep(3)
            
    st.error("No se pudo conectar con la API de segmentación. Verifica si el contenedor 'api' inició correctamente.")
    st.stop()
@st.cache_data
def load_metricas_supervisado():
    """Obtiene las métricas de los modelos de regresión y clasificación.
    Devuelve None si aún no se han entrenado (endpoint responde 503)."""
    try:
        respuesta = requests.get("http://api:8000/metricas-supervisado")
        if respuesta.status_code == 200:
            return respuesta.json()
    except requests.exceptions.ConnectionError:
        pass
    return None

# LLAMADA A LA FUNCIÓN Y ASIGNACIÓN DE VARIABLES 
data, centroides, evaluacion_k, metricas = load_data()
metricas_supervisado = load_metricas_supervisado()

st.title("Segmentación de Usuarios — Streaming")

# --- Filtro global (sidebar): afecta a las 3 vistas ---
clusters_disponibles = sorted(data["cluster"].unique())
clusters_seleccionados = st.sidebar.multiselect(
    "Filtrar por segmento",
    options=clusters_disponibles,
    default=clusters_disponibles,
    format_func=lambda c: f"Cluster {c}",
)
df = data[data["cluster"].isin(clusters_seleccionados)]

if df.empty:
    st.warning("Selecciona al menos un segmento en la barra lateral.")
    st.stop()

tab_ejecutiva, tab_tecnica, tab_operativa, tab_predictiva = st.tabs(
    ["Vista Ejecutiva", "Vista Técnica", "Vista Operativa", "Modelos Predictivos"]
)

# ============================================================
# VISTA EJECUTIVA: tamaño de segmentos + interpretación de negocio
# ============================================================
with tab_ejecutiva:
    st.subheader("¿Cuántos usuarios hay en cada segmento?")

    col1, col2 = st.columns(2)

    df_grafico = df.copy()
    df_grafico["cluster_label"] = "Cluster " + df_grafico["cluster"].astype(str)
    conteo = df_grafico["cluster_label"].value_counts().sort_index()

    with col1:
        fig_barras = px.bar(
            conteo,
            labels={"index": "Segmento", "value": "Usuarios"},
            title="Cantidad de usuarios por segmento",
        )
        fig_barras.update_layout(showlegend=False)
        st.plotly_chart(fig_barras, use_container_width=True)

    with col2:
        fig_torta = px.pie(
            df_grafico, names="cluster_label", title="Distribución porcentual", hole=0.4,
        )
        st.plotly_chart(fig_torta, use_container_width=True)

    st.subheader("Perfil de cada segmento")

    perfil_ejecutivo = df.groupby("cluster")[
        ["horas_consumo_mensual", "gasto_mensual", "cantidad_contenidos_vistos",
         "antiguedad_cliente_meses", "porcentaje_uso_promociones", "dispositivos_registrados"]
    ].mean()

    # porcentaje_uso_promociones viene como fracción (0-1) en los datos crudos; se muestra como %
    perfil_ejecutivo["porcentaje_uso_promociones"] *= 100
    perfil_ejecutivo = perfil_ejecutivo.round(1)

    perfil_ejecutivo = perfil_ejecutivo.rename(columns={
        "horas_consumo_mensual": "Horas de consumo (prom./mes)",
        "gasto_mensual": "Gasto mensual (prom.)",
        "cantidad_contenidos_vistos": "Contenidos vistos (prom.)",
        "antiguedad_cliente_meses": "Antigüedad (meses, prom.)",
        "porcentaje_uso_promociones": "Uso de promociones (%)",
        "dispositivos_registrados": "Dispositivos (prom.)",
    })
    perfil_ejecutivo.index = [f"Cluster {c}" for c in perfil_ejecutivo.index]
    perfil_ejecutivo.index.name = "Segmento"

    st.dataframe(perfil_ejecutivo, use_container_width=True)

    st.subheader("Interpretación de negocio")
    for cluster in sorted(df["cluster"].unique()):
        n_usuarios = (df["cluster"] == cluster).sum()
        pct = 100 * n_usuarios / len(df)
        descripcion = INTERPRETACIONES.get(cluster, "Pendiente de interpretar.")
        st.markdown(f"**Cluster {cluster}** ({n_usuarios} usuarios, {pct:.1f}%): {descripcion}")

# ============================================================
# VISTA TÉCNICA
# ============================================================
with tab_tecnica:
    st.subheader("Selección del número de clusters (k)")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Silhouette Score", f"{metricas['silhouette_score']:.3f}")
    with col2:
        st.metric("K óptimo", metricas["k_optimo"])
    with col3:
        st.metric("Varianza explicada (PCA)", f"{metricas['varianza_pca'] * 100:.1f}%")

    st.caption(
        "El Silhouette se calcula sobre las 15 variables escaladas, no sobre los "
        "componentes de PCA. PCA se usa únicamente para visualización."
    )
    
    fig_codo = go.Figure()
    fig_codo.add_trace(go.Scatter(
        x=evaluacion_k["k"], y=evaluacion_k["inertia"],
        name="Inercia", mode="lines+markers",
    ))
    fig_codo.add_trace(go.Scatter(
        x=evaluacion_k["k"], y=evaluacion_k["silhouette"],
        name="Silhouette", mode="lines+markers", yaxis="y2",
    ))
    fig_codo.add_vline(
        x=metricas["k_optimo"], line_dash="dash", line_color="red",
        annotation_text=f"k óptimo = {metricas['k_optimo']}",
    )
    fig_codo.update_layout(
        title="Método del codo (inercia) + coeficiente de Silhouette",
        xaxis_title="Número de clusters (k)",
        yaxis=dict(title="Inercia"),
        yaxis2=dict(title="Silhouette", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_codo, use_container_width=True)
    st.caption(
        "La línea roja marca el k elegido por KneeLocator (método del codo). "
        "Se confirma cruzándolo con el punto de mayor Silhouette."
    )

    st.subheader("Proyección PCA de los segmentos")

    fig_pca = px.scatter(
        df,
        x="pc1",
        y="pc2",
        color=df["cluster"].astype(str),
        hover_data=["id_cliente"],
        labels={"color": "Cluster", "pc1": "PC1", "pc2": "PC2"},
        title="Segmentos proyectados en 2 componentes principales",
    )

    # Marcar los centroides como puntos grandes con X
    centroides_pca = df.groupby("cluster")[["pc1", "pc2"]].mean()
    fig_pca.add_trace(go.Scatter(
        x=centroides_pca["pc1"],
        y=centroides_pca["pc2"],
        mode="markers",
        marker=dict(size=18, symbol="x", color="red", line=dict(width=2)),
        name="Centroides",
        showlegend=True,
    ))

    st.plotly_chart(fig_pca, use_container_width=True)

# ============================================================
# VISTA OPERATIVA
# ============================================================
with tab_operativa:
    st.subheader("Buscar usuario individual")
    st.caption(
        "Esta búsqueda no usa el filtro de la barra lateral: buscar un cliente "
        "puntual es una operación distinta a explorar los segmentos seleccionados."
    )

    id_seleccionado = st.selectbox(
        "id_cliente",
        options=sorted(data["id_cliente"].unique()),
    )

    usuario = data[data["id_cliente"] == id_seleccionado].iloc[0]
    cluster_usuario = int(usuario["cluster"])
    centroide_cluster = centroides.iloc[cluster_usuario]

    st.markdown(
        f"**Cluster {cluster_usuario}** — {INTERPRETACIONES.get(cluster_usuario, '')}"
    )

    variables_lookup = [
        "horas_consumo_mensual", "gasto_mensual", "cantidad_contenidos_vistos",
        "antiguedad_cliente_meses", "porcentaje_uso_promociones", "dispositivos_registrados",
    ]

    comparacion_usuario = pd.DataFrame({
        "Variable": variables_lookup,
        "Usuario": [round(usuario[v], 2) for v in variables_lookup],
        "Promedio del cluster": [round(centroide_cluster[v], 2) for v in variables_lookup],
    })
    comparacion_usuario["Diferencia vs. cluster"] = (
        (comparacion_usuario["Usuario"] - comparacion_usuario["Promedio del cluster"])
        / comparacion_usuario["Promedio del cluster"] * 100
    ).round(1).astype(str) + "%"

    st.dataframe(comparacion_usuario, use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("Tabla de usuarios")
    st.dataframe(df, use_container_width=True, height=300)

    variables_comparables = [
        "horas_consumo_mensual", "gasto_mensual", "cantidad_contenidos_vistos",
        "sesiones_semana", "porcentaje_finalizacion", "tiempo_promedio_sesion_min",
        "cantidad_generos_consumidos", "porcentaje_uso_promociones",
        "antiguedad_cliente_meses", "edad", "dispositivos_registrados",
        "porcentaje_uso_app_movil", "cantidad_perfiles_creados",
        "interacciones_mensuales_soporte", "distancia_promedio_red_km",
    ]

    perfil_promedio = df.groupby("cluster")[variables_comparables].mean()
    perfil_normalizado = (perfil_promedio - perfil_promedio.min()) / (
        perfil_promedio.max() - perfil_promedio.min()
    )

    st.subheader("Mapa de calor: variables por segmento")
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=perfil_normalizado.values,
        x=perfil_normalizado.columns,
        y=[f"Cluster {c}" for c in perfil_normalizado.index],
        text=perfil_promedio.round(1).values,
        texttemplate="%{text}",
        textfont=dict(size=10),
        colorscale="Blues",
        colorbar=dict(title="Nivel<br>relativo"),
    ))
    fig_heatmap.update_layout(xaxis_title="Variable", yaxis_title="Cluster")
    st.plotly_chart(fig_heatmap, use_container_width=True)
    st.caption(
        "El color indica la posición relativa de cada segmento en esa variable "
        "(más oscuro = más alto entre los 3 clusters). El número en cada celda "
        "es el valor real (escala original), no el normalizado."
    )

    st.subheader("Comparación de segmentos en variables clave")

    variables_destacadas = [
        "gasto_mensual", "porcentaje_finalizacion", "tiempo_promedio_sesion_min",
        "porcentaje_uso_promociones", "antiguedad_cliente_meses",
        "sesiones_semana", "cantidad_generos_consumidos",
    ]

    datos_comparacion = perfil_normalizado[variables_destacadas].reset_index()
    datos_comparacion = datos_comparacion.melt(
        id_vars="cluster", var_name="variable", value_name="nivel_relativo"
    )
    datos_comparacion["cluster"] = "Cluster " + datos_comparacion["cluster"].astype(str)

    fig_comparacion = px.bar(
        datos_comparacion,
        x="variable",
        y="nivel_relativo",
        color="cluster",
        barmode="group",
        labels={
            "variable": "Variable",
            "nivel_relativo": "Nivel relativo (0-1)",
            "cluster": "Segmento",
        },
    )
    fig_comparacion.update_layout(xaxis_tickangle=-30)
    st.plotly_chart(fig_comparacion, use_container_width=True)
    st.caption(
        "Mismas variables y misma normalización que el mapa de calor, en formato "
        "de barras agrupadas — más preciso de leer que un gráfico radial."
    )
# ============================================================
# VISTA MODELOS PREDICTIVOS: regresión (gasto) y clasificación (riesgo)
# ============================================================
with tab_predictiva:
    if metricas_supervisado is None:
        st.warning(
            "Los modelos supervisados (regresión y clasificación) todavía no están "
            "entrenados. Corre `model/train_supervisado.py` y recarga la página."
        )
        st.stop()

    st.subheader("Comparación de modelos entrenados")
    st.caption(
        "Se probó más de un algoritmo por tarea; el modelo ganador se eligió "
        "por la métrica más relevante para el negocio (R² para regresión, F1 "
        "para clasificación, dado el desbalance de clases)."
    )

    col_reg, col_clf = st.columns(2)

    # --- Regresión: predicción de gasto mensual ---
    with col_reg:
        st.markdown("##### Regresión — predicción de gasto mensual")
        reg = metricas_supervisado["regresion"]
        mejor_reg = reg["mejor_modelo"]

        df_reg = pd.DataFrame({k: v for k, v in reg.items() if isinstance(v, dict)}).T
        df_reg = df_reg.rename(columns={"r2": "R²", "mae": "MAE", "rmse": "RMSE"})
        st.dataframe(df_reg.round(3), use_container_width=True)
        st.success(f"Modelo elegido: **{mejor_reg}**")

    # --- Clasificación: predicción de riesgo ---
    with col_clf:
        st.markdown("##### Clasificación — riesgo de bajo compromiso")
        clf = metricas_supervisado["clasificacion"]
        mejor_clf = clf["mejor_modelo"]

        df_clf = pd.DataFrame({
            k: v for k, v in clf.items() if isinstance(v, dict)
        }).T
        df_clf = df_clf.rename(columns={
            "accuracy": "Accuracy", "precision": "Precision",
            "recall": "Recall", "f1": "F1", "roc_auc": "ROC-AUC",
        })
        st.dataframe(df_clf.round(3), use_container_width=True)
        st.success(f"Modelo elegido: **{mejor_clf}**")

    st.caption(
        "Nota: porcentaje_finalizacion y sesiones_semana no se usan como "
        "variables en el modelo de clasificación porque son las que definen "
        "la etiqueta de riesgo (evita fuga de datos)."
    )

    st.divider()

    st.subheader("Simulador: predicción para un usuario nuevo")
    st.caption(
        "Ingresa las variables de un usuario (real o hipotético) para ver el "
        "gasto mensual estimado y su riesgo de bajo compromiso."
    )

    with st.form("form_prediccion"):
        c1, c2, c3 = st.columns(3)
        with c1:
            horas_consumo = st.number_input("Horas de consumo mensual", 0, 200, 40)
            contenidos = st.number_input("Cantidad de contenidos vistos", 0, 200, 20)
            sesiones = st.number_input("Sesiones por semana", 0, 30, 6)
            finalizacion = st.number_input("Porcentaje de finalización", 0, 100, 60)
        with c2:
            tiempo_sesion = st.number_input("Tiempo promedio de sesión (min)", 0, 400, 100)
            generos = st.number_input("Cantidad de géneros consumidos", 0, 20, 5)
            promociones = st.slider("Uso de promociones (%)", 0.0, 1.0, 0.3)
            antiguedad = st.number_input("Antigüedad del cliente (meses)", 0, 150, 30)
        with c3:
            edad = st.number_input("Edad", 15, 100, 35)
            dispositivos = st.number_input("Dispositivos registrados", 1, 10, 2)
            uso_app = st.slider("Uso de app móvil (%)", 0.0, 1.0, 0.5)
            perfiles = st.number_input("Cantidad de perfiles creados", 1, 10, 3)

        col4, col5 = st.columns(2)
        with col4:
            interacciones_soporte = st.number_input("Interacciones mensuales con soporte", 0, 30, 2)
        with col5:
            distancia_red = st.number_input("Distancia promedio a red (km)", 0.0, 200.0, 20.0)

        enviado = st.form_submit_button("Predecir")

    if enviado:
        payload = {
            "horas_consumo_mensual": horas_consumo,
            "cantidad_contenidos_vistos": contenidos,
            "sesiones_semana": sesiones,
            "porcentaje_finalizacion": finalizacion,
            "tiempo_promedio_sesion_min": tiempo_sesion,
            "cantidad_generos_consumidos": generos,
            "porcentaje_uso_promociones": promociones,
            "antiguedad_cliente_meses": antiguedad,
            "edad": edad,
            "dispositivos_registrados": dispositivos,
            "porcentaje_uso_app_movil": uso_app,
            "cantidad_perfiles_creados": perfiles,
            "interacciones_mensuales_soporte": interacciones_soporte,
            "distancia_promedio_red_km": distancia_red,
        }

        col_res1, col_res2 = st.columns(2)

        try:
            r_gasto = requests.post("http://api:8000/predict-gasto", json=payload)
            r_riesgo = requests.post("http://api:8000/predict-riesgo", json=payload)

            with col_res1:
                if r_gasto.status_code == 200:
                    gasto = r_gasto.json()["gasto_mensual_predicho"]
                    st.metric("Gasto mensual estimado", f"${gasto:,.0f}")
                else:
                    st.error("No se pudo obtener la predicción de gasto.")

            with col_res2:
                if r_riesgo.status_code == 200:
                    resultado = r_riesgo.json()
                    riesgo = resultado["riesgo_bajo_compromiso"]
                    prob = resultado["probabilidad"]
                    if riesgo == 1:
                        st.metric("Riesgo de bajo compromiso", "Sí", delta=f"{prob:.0%} prob.", delta_color="inverse")
                    else:
                        st.metric("Riesgo de bajo compromiso", "No", delta=f"{prob:.0%} prob.", delta_color="normal")
                else:
                    st.error("No se pudo obtener la predicción de riesgo.")
        except requests.exceptions.ConnectionError:
            st.error("No se pudo conectar con la API.")