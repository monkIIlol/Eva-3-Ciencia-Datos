"""
Dashboard de segmentación de usuarios de streaming.

Consume la API (api/main.py) vía HTTP. Organizado en 3 vistas según
audiencia (ejecutiva, técnica, operativa), con un filtro global de
segmentos en la barra lateral que afecta a las tres pestañas.
"""
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Segmentación de Usuarios — Streaming", layout="wide")

# --- Interpretación de negocio por cluster ---
# Completada a partir del análisis de los centroides reales.
# Ver docs/decisiones_diseno.md para la justificación de cada etiqueta.
INTERPRETACIONES = {
    0: "Usuarios habituales exploradores: sesiones frecuentes pero cortas, consumo moderado.",
    1: "Usuarios nuevos sensibles a precio: baja antigüedad, alto uso de promociones, bajo consumo.",
    2: "Usuarios premium leales: alto gasto, alta finalización, baja sensibilidad a promociones.",
}


@st.cache_data
def load_data():
    """Obtiene usuarios, centroides y métricas desde la API."""
    respuesta = requests.get("http://api:8000/dashboard-data")
    payload = respuesta.json()
    usuarios = pd.DataFrame(payload["usuarios"])
    centroides = pd.DataFrame(payload["centroides"])
    evaluacion_k = pd.DataFrame(payload["evaluacion_k"])
    metricas = payload["metricas"]
    return usuarios, centroides, evaluacion_k, metricas


data, centroides, evaluacion_k, metricas = load_data()

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

tab_ejecutiva, tab_tecnica, tab_operativa = st.tabs(
    ["Vista Ejecutiva", "Vista Técnica", "Vista Operativa"]
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

    st.subheader("Interpretación de negocio")
    for cluster in sorted(df["cluster"].unique()):
        n_usuarios = (df["cluster"] == cluster).sum()
        pct = 100 * n_usuarios / len(df)
        descripcion = INTERPRETACIONES.get(cluster, "Pendiente de interpretar.")
        st.markdown(f"**Cluster {cluster}** ({n_usuarios} usuarios, {pct:.1f}%): {descripcion}")

# ============================================================
# VISTA TÉCNICA Y OPERATIVA — las construimos en el siguiente paso
# ============================================================
with tab_tecnica:
    st.subheader("Selección del número de clusters (k)")

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

with tab_operativa:
    st.info("🚧 En construcción — la añadimos en el siguiente paso.")