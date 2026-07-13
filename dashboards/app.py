"""
Dashboard de segmentación de usuarios de streaming.

Consume la API (api/main.py) vía HTTP. Organizado en vistas ejecutiva,
técnica, operativa, predictiva y de pipeline/calidad.
"""
import time

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st


API_URL = "http://api:8000"

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
    url = f"{API_URL}/dashboard-data"
    max_intentos = 15

    for intento in range(max_intentos):
        try:
            respuesta = requests.get(url, timeout=10)
            if respuesta.status_code == 200:
                payload = respuesta.json()
                usuarios = pd.DataFrame(payload["usuarios"])
                centroides = pd.DataFrame(payload["centroides"])
                evaluacion_k = pd.DataFrame(payload["evaluacion_k"])
                metricas = payload["metricas"]
                return usuarios, centroides, evaluacion_k, metricas
        except requests.exceptions.RequestException:
            st.sidebar.info(
                f"Esperando que la API se inicialice... "
                f"(Intento {intento + 1}/{max_intentos})"
            )
            time.sleep(3)

    st.error(
        "No se pudo conectar con la API de segmentación. "
        "Verifica si el contenedor 'api' inició correctamente."
    )
    st.stop()


@st.cache_data
def load_metricas_supervisado():
    """
    Obtiene las métricas de los modelos de regresión y clasificación.
    Devuelve None si aún no se han entrenado o el endpoint no responde.
    """
    try:
        respuesta = requests.get(f"{API_URL}/metricas-supervisado", timeout=10)
        if respuesta.status_code == 200:
            return respuesta.json()
    except requests.exceptions.RequestException:
        pass
    return None


@st.cache_data
def load_pipeline_y_calidad():
    """Obtiene estado de API, KPIs y reporte de calidad desde la API."""
    health = None
    kpis = None
    reporte = None

    try:
        respuesta_health = requests.get(f"{API_URL}/health", timeout=10)
        if respuesta_health.status_code == 200:
            health = respuesta_health.json()
    except requests.exceptions.RequestException:
        health = None

    try:
        respuesta_kpis = requests.get(f"{API_URL}/kpis-negocio", timeout=10)
        if respuesta_kpis.status_code == 200:
            kpis = pd.DataFrame(respuesta_kpis.json().get("kpis", []))
    except requests.exceptions.RequestException:
        kpis = None

    try:
        respuesta_reporte = requests.get(f"{API_URL}/reporte-calidad", timeout=10)
        if respuesta_reporte.status_code == 200:
            reporte = respuesta_reporte.json()
    except requests.exceptions.RequestException:
        reporte = None

    return health, kpis, reporte


def normalizar_perfil(perfil_promedio: pd.DataFrame) -> pd.DataFrame:
    """Normaliza variables por columna evitando división por cero."""
    denominador = perfil_promedio.max() - perfil_promedio.min()
    denominador = denominador.replace(0, 1)
    return (perfil_promedio - perfil_promedio.min()) / denominador


def extraer_estado_artefactos(health_api: dict | None) -> dict:
    """Extrae estado de artefactos desde health endpoint con tolerancia a esquemas."""
    if not health_api:
        return {}

    if isinstance(health_api.get("artefactos"), dict):
        return health_api["artefactos"]

    if isinstance(health_api.get("artifacts"), dict):
        return health_api["artifacts"]

    return health_api


# ============================================================
# CARGA DE DATOS
# ============================================================
data, centroides, evaluacion_k, metricas = load_data()
metricas_supervisado = load_metricas_supervisado()
health_api, kpis_negocio, reporte_calidad = load_pipeline_y_calidad()

st.title("Segmentación de Usuarios — Streaming")

# --- Filtro global (sidebar) ---
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

tab_ejecutiva, tab_tecnica, tab_operativa, tab_predictiva, tab_pipeline = st.tabs(
    [
        "Vista Ejecutiva",
        "Vista Técnica",
        "Vista Operativa",
        "Modelos Predictivos",
        "Pipeline y Calidad",
    ]
)

# ============================================================
# VISTA EJECUTIVA
# ============================================================
with tab_ejecutiva:
    st.subheader("Resumen ejecutivo de segmentos")

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)

    with col_m1:
        st.metric("Usuarios analizados", f"{len(df):,}")
    with col_m2:
        st.metric("Segmentos activos", df["cluster"].nunique())
    with col_m3:
        st.metric("Gasto promedio", f"${df['gasto_mensual'].mean():,.0f}")
    with col_m4:
        st.metric("Consumo promedio", f"{df['horas_consumo_mensual'].mean():.1f} h/mes")

    st.divider()

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
            df_grafico,
            names="cluster_label",
            title="Distribución porcentual",
            hole=0.4,
        )
        st.plotly_chart(fig_torta, use_container_width=True)

    st.subheader("Perfil de cada segmento")

    perfil_ejecutivo = df.groupby("cluster")[
        [
            "horas_consumo_mensual",
            "gasto_mensual",
            "cantidad_contenidos_vistos",
            "antiguedad_cliente_meses",
            "porcentaje_uso_promociones",
            "dispositivos_registrados",
        ]
    ].mean()

    perfil_ejecutivo["porcentaje_uso_promociones"] *= 100
    perfil_ejecutivo = perfil_ejecutivo.round(1)

    perfil_ejecutivo = perfil_ejecutivo.rename(
        columns={
            "horas_consumo_mensual": "Horas de consumo (prom./mes)",
            "gasto_mensual": "Gasto mensual (prom.)",
            "cantidad_contenidos_vistos": "Contenidos vistos (prom.)",
            "antiguedad_cliente_meses": "Antigüedad (meses, prom.)",
            "porcentaje_uso_promociones": "Uso de promociones (%)",
            "dispositivos_registrados": "Dispositivos (prom.)",
        }
    )
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
        "El Silhouette se calcula sobre las variables escaladas. "
        "PCA se usa únicamente para visualización."
    )

    fig_codo = go.Figure()
    fig_codo.add_trace(
        go.Scatter(
            x=evaluacion_k["k"],
            y=evaluacion_k["inertia"],
            name="Inercia",
            mode="lines+markers",
        )
    )
    fig_codo.add_trace(
        go.Scatter(
            x=evaluacion_k["k"],
            y=evaluacion_k["silhouette"],
            name="Silhouette",
            mode="lines+markers",
            yaxis="y2",
        )
    )
    fig_codo.add_vline(
        x=metricas["k_optimo"],
        line_dash="dash",
        line_color="red",
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
        "La línea roja marca el k elegido por KneeLocator. "
        "Se contrasta con el comportamiento de Silhouette."
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

    centroides_pca = df.groupby("cluster")[["pc1", "pc2"]].mean()
    fig_pca.add_trace(
        go.Scatter(
            x=centroides_pca["pc1"],
            y=centroides_pca["pc2"],
            mode="markers",
            marker=dict(size=18, symbol="x", color="red", line=dict(width=2)),
            name="Centroides",
            showlegend=True,
        )
    )

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

    st.markdown(f"**Cluster {cluster_usuario}** — {INTERPRETACIONES.get(cluster_usuario, '')}")

    variables_lookup = [
        "horas_consumo_mensual",
        "gasto_mensual",
        "cantidad_contenidos_vistos",
        "antiguedad_cliente_meses",
        "porcentaje_uso_promociones",
        "dispositivos_registrados",
    ]

    comparacion_usuario = pd.DataFrame(
        {
            "Variable": variables_lookup,
            "Usuario": [round(usuario[v], 2) for v in variables_lookup],
            "Promedio del cluster": [round(centroide_cluster[v], 2) for v in variables_lookup],
        }
    )

    comparacion_usuario["Diferencia vs. cluster"] = (
        (comparacion_usuario["Usuario"] - comparacion_usuario["Promedio del cluster"])
        / comparacion_usuario["Promedio del cluster"].replace(0, 1)
        * 100
    ).round(1).astype(str) + "%"

    st.dataframe(comparacion_usuario, use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("Tabla de usuarios")
    st.dataframe(df, use_container_width=True, height=300)

    variables_comparables = [
        "horas_consumo_mensual",
        "gasto_mensual",
        "cantidad_contenidos_vistos",
        "sesiones_semana",
        "porcentaje_finalizacion",
        "tiempo_promedio_sesion_min",
        "cantidad_generos_consumidos",
        "porcentaje_uso_promociones",
        "antiguedad_cliente_meses",
        "edad",
        "dispositivos_registrados",
        "porcentaje_uso_app_movil",
        "cantidad_perfiles_creados",
        "interacciones_mensuales_soporte",
        "distancia_promedio_red_km",
    ]

    perfil_promedio = df.groupby("cluster")[variables_comparables].mean()
    perfil_normalizado = normalizar_perfil(perfil_promedio)

    st.subheader("Mapa de calor: variables por segmento")
    fig_heatmap = go.Figure(
        data=go.Heatmap(
            z=perfil_normalizado.values,
            x=perfil_normalizado.columns,
            y=[f"Cluster {c}" for c in perfil_normalizado.index],
            text=perfil_promedio.round(1).values,
            texttemplate="%{text}",
            textfont=dict(size=10),
            colorscale="Blues",
            colorbar=dict(title="Nivel<br>relativo"),
        )
    )
    fig_heatmap.update_layout(xaxis_title="Variable", yaxis_title="Cluster")
    st.plotly_chart(fig_heatmap, use_container_width=True)
    st.caption(
        "El color indica la posición relativa de cada segmento en esa variable "
        "(más oscuro = más alto entre los segmentos). El número es el valor real."
    )

    st.subheader("Comparación de segmentos en variables clave")

    variables_destacadas = [
        "gasto_mensual",
        "porcentaje_finalizacion",
        "tiempo_promedio_sesion_min",
        "porcentaje_uso_promociones",
        "antiguedad_cliente_meses",
        "sesiones_semana",
        "cantidad_generos_consumidos",
    ]

    datos_comparacion = perfil_normalizado[variables_destacadas].reset_index()
    datos_comparacion = datos_comparacion.melt(
        id_vars="cluster",
        var_name="variable",
        value_name="nivel_relativo",
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
        "Mismas variables y misma normalización que el mapa de calor, "
        "en formato de barras agrupadas."
    )


# ============================================================
# VISTA MODELOS PREDICTIVOS
# ============================================================
with tab_predictiva:
    if metricas_supervisado is None:
        st.warning(
            "Los modelos supervisados todavía no están entrenados. "
            "Ejecuta el pipeline integrado y recarga la página."
        )
        st.stop()

    st.subheader("Comparación de modelos entrenados")
    st.caption(
        "Se probó más de un algoritmo por tarea; el modelo ganador se eligió "
        "por la métrica más relevante para el negocio."
    )

    col_reg, col_clf = st.columns(2)

    with col_reg:
        st.markdown("##### Regresión — predicción de gasto mensual")
        reg = metricas_supervisado["regresion"]
        mejor_reg = reg["mejor_modelo"]

        df_reg = pd.DataFrame({k: v for k, v in reg.items() if isinstance(v, dict)}).T
        df_reg = df_reg.rename(columns={"r2": "R²", "mae": "MAE", "rmse": "RMSE"})
        st.dataframe(df_reg.round(3), use_container_width=True)
        st.success(f"Modelo elegido: **{mejor_reg}**")

    with col_clf:
        st.markdown("##### Clasificación — riesgo de bajo compromiso")
        clf = metricas_supervisado["clasificacion"]
        mejor_clf = clf["mejor_modelo"]

        df_clf = pd.DataFrame({k: v for k, v in clf.items() if isinstance(v, dict)}).T
        df_clf = df_clf.rename(
            columns={
                "accuracy": "Accuracy",
                "precision": "Precision",
                "recall": "Recall",
                "f1": "F1",
                "roc_auc": "ROC-AUC",
            }
        )
        st.dataframe(df_clf.round(3), use_container_width=True)
        st.success(f"Modelo elegido: **{mejor_clf}**")

    st.caption(
        "Nota: porcentaje_finalizacion y sesiones_semana no se usan como variables "
        "en el modelo de clasificación porque son las que definen la etiqueta proxy "
        "de riesgo. Esto evita fuga de datos."
    )

    st.divider()

    st.subheader("Simulador: predicción para un usuario nuevo")
    st.caption(
        "Ingresa las variables de un usuario real o hipotético para estimar gasto mensual "
        "y riesgo de bajo compromiso."
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
            interacciones_soporte = st.number_input(
                "Interacciones mensuales con soporte", 0, 30, 2
            )
        with col5:
            distancia_red = st.number_input(
                "Distancia promedio a red (km)", 0.0, 200.0, 20.0
            )

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
            r_gasto = requests.post(f"{API_URL}/predict-gasto", json=payload, timeout=10)
            r_riesgo = requests.post(f"{API_URL}/predict-riesgo", json=payload, timeout=10)

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
                        st.metric(
                            "Riesgo de bajo compromiso",
                            "Sí",
                            delta=f"{prob:.0%} prob.",
                            delta_color="inverse",
                        )
                    else:
                        st.metric(
                            "Riesgo de bajo compromiso",
                            "No",
                            delta=f"{prob:.0%} prob.",
                            delta_color="normal",
                        )
                else:
                    st.error("No se pudo obtener la predicción de riesgo.")
        except requests.exceptions.RequestException:
            st.error("No se pudo conectar con la API.")


# ============================================================
# VISTA PIPELINE Y CALIDAD
# ============================================================
with tab_pipeline:
    st.subheader("Pipeline integrado y calidad de datos")

    st.caption(
        "Esta vista resume el flujo end-to-end del proyecto, los artefactos generados, "
        "los KPIs de negocio y la evidencia de calidad de datos antes del modelado."
    )

    st.markdown("### Estado de la API y artefactos")

    estado = extraer_estado_artefactos(health_api)

    if not estado:
        st.warning("No se pudo obtener el estado de la API.")
    else:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("API", "OK" if health_api else "No disponible")
        with col2:
            st.metric("KMeans", "OK" if estado.get("kmeans", True) else "No disponible")
        with col3:
            st.metric("Regresión", "OK" if estado.get("regresion", True) else "No disponible")
        with col4:
            st.metric(
                "Clasificación",
                "OK" if estado.get("clasificacion", True) else "No disponible",
            )

    st.divider()

    st.markdown("### Flujo general del sistema")

    st.code(
        """
Fuentes de datos
↓
Extracción e integración
↓
Validación de datos
↓
Preparación del dataset
↓
Dataset base limpio + dataset analítico
↓
KMeans + modelos supervisados
↓
API + dashboard
        """,
        language="text",
    )

    st.markdown("### Artefactos principales")

    artefactos = pd.DataFrame(
        [
            {
                "Artefacto": "data/data_consolidada.csv",
                "Uso": "Dataset integrado desde las fuentes originales",
            },
            {
                "Artefacto": "data/dataset_base_limpio.csv",
                "Uso": "Entrada oficial para KMeans, regresión y clasificación",
            },
            {
                "Artefacto": "data/dataset_analitico.csv",
                "Uso": "Variables derivadas, análisis y KPIs",
            },
            {
                "Artefacto": "data/kpis_negocio.csv",
                "Uso": "Indicadores de negocio por engagement y valor de cliente",
            },
            {
                "Artefacto": "data/reporte_calidad.json",
                "Uso": "Evidencia de calidad, nulos, rangos, outliers y variables derivadas",
            },
            {
                "Artefacto": "models/pipeline_manifest.json",
                "Uso": "Evidencia de ejecución del pipeline integrado",
            },
        ]
    )

    st.dataframe(artefactos, use_container_width=True, hide_index=True)

    st.divider()

    st.markdown("### Reporte de calidad de datos")

    if reporte_calidad is None:
        st.warning("No se pudo cargar el reporte de calidad desde la API.")
    else:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Filas originales", reporte_calidad.get("filas_originales", "N/D"))
        with col2:
            st.metric("Filas finales", reporte_calidad.get("filas_finales", "N/D"))
        with col3:
            st.metric("Columnas finales", reporte_calidad.get("columnas_finales", "N/D"))
        with col4:
            st.metric("Duplicados", reporte_calidad.get("duplicados_eliminados", "N/D"))

        st.markdown("#### Variables derivadas")

        variables_derivadas = reporte_calidad.get("variables_derivadas", [])
        if variables_derivadas:
            st.write(", ".join(variables_derivadas))
        else:
            st.info("No se registraron variables derivadas en el reporte.")

        st.markdown("#### Diagnóstico de outliers")

        outliers = reporte_calidad.get("outliers_iqr", {})
        if outliers:
            df_outliers = pd.DataFrame(outliers).T.reset_index()
            df_outliers = df_outliers.rename(columns={"index": "variable"})
            st.dataframe(df_outliers, use_container_width=True)
        else:
            st.info("No se registró diagnóstico de outliers.")

        with st.expander("Ver reporte de calidad completo"):
            st.json(reporte_calidad)

    st.divider()

    st.markdown("### KPIs de negocio")

    if kpis_negocio is None or kpis_negocio.empty:
        st.warning("No se pudieron cargar los KPIs de negocio desde la API.")
    else:
        st.dataframe(kpis_negocio, use_container_width=True)

        if "nivel_engagement" in kpis_negocio.columns and "usuarios" in kpis_negocio.columns:
            fig_kpis = px.bar(
                kpis_negocio,
                x="nivel_engagement",
                y="usuarios",
                color="valor_cliente" if "valor_cliente" in kpis_negocio.columns else None,
                barmode="group",
                title="Usuarios por nivel de engagement y valor de cliente",
                labels={
                    "nivel_engagement": "Nivel de engagement",
                    "usuarios": "Cantidad de usuarios",
                    "valor_cliente": "Valor de cliente",
                },
            )
            st.plotly_chart(fig_kpis, use_container_width=True)

        if "gasto_promedio" in kpis_negocio.columns:
            fig_gasto = px.bar(
                kpis_negocio,
                x="nivel_engagement",
                y="gasto_promedio",
                color="valor_cliente" if "valor_cliente" in kpis_negocio.columns else None,
                barmode="group",
                title="Gasto promedio por nivel de engagement",
                labels={
                    "nivel_engagement": "Nivel de engagement",
                    "gasto_promedio": "Gasto promedio",
                    "valor_cliente": "Valor de cliente",
                },
            )
            st.plotly_chart(fig_gasto, use_container_width=True)

    st.info(
        "Esta vista conecta ETL, preparación de datos, calidad, KPIs y modelos. "
        "Permite demostrar que la solución no solo entrena modelos, sino que también "
        "controla la calidad de los datos antes del modelado."
    )