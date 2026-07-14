"""Dashboard ejecutivo, técnico y operativo de analítica de streaming.

La interfaz consume exclusivamente la API REST. Presenta las cinco capas
principales del proyecto: valor de negocio, calidad/pipeline, segmentación,
modelos supervisados y demo end-to-end.
"""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st


st.set_page_config(
    page_title="Analítica de Usuarios - Streaming",
    page_icon="📊",
    layout="wide",
)

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT = 15

NOMBRES_VARIABLES = {
    "horas_consumo_mensual": "Horas de consumo mensual",
    "gasto_mensual": "Gasto mensual",
    "cantidad_contenidos_vistos": "Contenidos vistos",
    "sesiones_semana": "Sesiones por semana",
    "porcentaje_finalizacion": "Finalización (%)",
    "tiempo_promedio_sesion_min": "Duración promedio de sesión",
    "cantidad_generos_consumidos": "Géneros consumidos",
    "porcentaje_uso_promociones": "Uso de promociones",
    "antiguedad_cliente_meses": "Antigüedad del cliente",
    "edad": "Edad",
    "dispositivos_registrados": "Dispositivos registrados",
    "porcentaje_uso_app_movil": "Uso de app móvil",
    "cantidad_perfiles_creados": "Perfiles creados",
    "interacciones_mensuales_soporte": "Interacciones con soporte",
    "distancia_promedio_red_km": "Distancia promedio a red",
}


# ---------------------------------------------------------------------------
# Acceso a API
# ---------------------------------------------------------------------------


def _detalle_error(respuesta: requests.Response) -> str:
    try:
        detalle = respuesta.json().get("detail")
        if detalle:
            return str(detalle)
    except ValueError:
        pass
    return f"HTTP {respuesta.status_code}"


def _get_json(endpoint: str, obligatorio: bool = False) -> dict[str, Any] | None:
    try:
        respuesta = requests.get(
            f"{API_BASE_URL}{endpoint}",
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as exc:
        if obligatorio:
            st.error(f"No se pudo conectar con la API: {exc}")
            st.stop()
        return None

    if respuesta.status_code == 200:
        return respuesta.json()

    if obligatorio:
        st.error(f"La API no pudo entregar {endpoint}: {_detalle_error(respuesta)}")
        st.stop()
    return None


@st.cache_data(ttl=60, show_spinner=False)
def cargar_dashboard() -> dict[str, Any]:
    """Carga todos los productos publicados requeridos por las cinco vistas."""
    principal = _get_json("/dashboard-data", obligatorio=True)
    return {
        "principal": principal,
        "health": _get_json("/health"),
        "supervisado": _get_json("/metricas-supervisado"),
        "kpis": _get_json("/business-kpis"),
        "calidad": _get_json("/data-quality"),
        "pipeline": _get_json("/pipeline-status"),
        "evidencia": _get_json("/model-evidence"),
    }


def _post_json(endpoint: str, payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    try:
        respuesta = requests.post(
            f"{API_BASE_URL}{endpoint}",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as exc:
        return None, f"No se pudo conectar con la API: {exc}"

    if respuesta.status_code == 200:
        return respuesta.json(), None
    return None, _detalle_error(respuesta)


# ---------------------------------------------------------------------------
# Preparación visual
# ---------------------------------------------------------------------------


def _normalizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    minimo = df.min()
    rango = (df.max() - minimo).replace(0, np.nan)
    return ((df - minimo) / rango).fillna(0.5)


def _nombres_segmentos(centroides: pd.DataFrame) -> dict[int, dict[str, str]]:
    """Asigna etiquetas descriptivas según el perfil relativo de centroides."""
    c = centroides.copy()
    if "cluster" not in c.columns:
        c.insert(0, "cluster", range(len(c)))
    c = c.set_index("cluster")

    ids = [int(valor) for valor in c.index]
    resultado = {
        cluster: {
            "nombre": f"Segmento {cluster}",
            "descripcion": "Perfil conductual identificado mediante KMeans.",
            "accion": "Analizar el perfil y diseñar una estrategia específica.",
        }
        for cluster in ids
    }

    if len(ids) != 3:
        return resultado

    ranking = c.rank(pct=True)
    premium_score = (
        ranking["gasto_mensual"]
        + ranking["porcentaje_finalizacion"]
        + ranking["antiguedad_cliente_meses"]
        - ranking["porcentaje_uso_promociones"]
    )
    premium = int(premium_score.idxmax())

    candidatos = [cluster for cluster in ids if cluster != premium]
    precio_score = (
        ranking.loc[candidatos, "porcentaje_uso_promociones"]
        - ranking.loc[candidatos, "gasto_mensual"]
        - ranking.loc[candidatos, "antiguedad_cliente_meses"]
    )
    precio = int(precio_score.idxmax())
    habitual = next(cluster for cluster in ids if cluster not in {premium, precio})

    resultado[premium] = {
        "nombre": "Premium leales",
        "descripcion": "Alto gasto, alta finalización y elevada antigüedad relativa.",
        "accion": "Fidelización, beneficios exclusivos y estrenos premium.",
    }
    resultado[precio] = {
        "nombre": "Nuevos sensibles a precio",
        "descripcion": "Menor antigüedad, gasto bajo y mayor uso relativo de promociones.",
        "accion": "Onboarding, promociones de continuidad y retención temprana.",
    }
    resultado[habitual] = {
        "nombre": "Habituales exploradores",
        "descripcion": "Uso frecuente y consumo intermedio con comportamiento exploratorio.",
        "accion": "Recomendaciones personalizadas y continuidad de contenidos.",
    }
    return resultado


def _etiqueta_cluster(cluster: int, segmentos: dict[int, dict[str, str]]) -> str:
    return f"Cluster {cluster} - {segmentos[int(cluster)]['nombre']}"


def _nombre_feature(nombre: str) -> str:
    return NOMBRES_VARIABLES.get(nombre, nombre.replace("_", " ").title())


bundle = cargar_dashboard()
principal = bundle["principal"]
data = pd.DataFrame(principal["usuarios"])
centroides = pd.DataFrame(principal["centroides"])
evaluacion_k = pd.DataFrame(principal["evaluacion_k"])
metricas_kmeans = principal["metricas"]
segmentos = _nombres_segmentos(centroides)

if "cluster" not in centroides.columns:
    centroides.insert(0, "cluster", range(len(centroides)))
centroides_indexados = centroides.set_index("cluster")

data["segmento"] = data["cluster"].astype(int).map(
    lambda cluster: _etiqueta_cluster(cluster, segmentos)
)

st.title("Analítica de Usuarios - Plataforma de Streaming")
st.caption(
    "Solución end-to-end: integración CSV + PostgreSQL, calidad de datos, "
    "segmentación, modelos supervisados, API y despliegue con Docker."
)

with st.sidebar:
    st.header("Control del dashboard")
    if st.button("Actualizar datos", width="stretch"):
        cargar_dashboard.clear()
        st.rerun()

    health = bundle.get("health") or {}
    estado_api = health.get("status", "sin conexión")
    if estado_api == "ok":
        st.success("API y artefactos disponibles")
    elif estado_api == "degraded":
        st.warning("API en modo degradado")
    else:
        st.error("Estado de API no disponible")

    clusters_disponibles = sorted(data["cluster"].astype(int).unique())
    clusters_seleccionados = st.multiselect(
        "Filtrar segmentos",
        options=clusters_disponibles,
        default=clusters_disponibles,
        format_func=lambda c: _etiqueta_cluster(c, segmentos),
    )

if not clusters_seleccionados:
    st.warning("Selecciona al menos un segmento.")
    st.stop()

df = data[data["cluster"].isin(clusters_seleccionados)].copy()

(
    tab_resumen,
    tab_pipeline,
    tab_segmentacion,
    tab_modelos,
    tab_operacion,
) = st.tabs(
    [
        "Resumen ejecutivo",
        "Datos y pipeline",
        "Segmentación",
        "Modelos predictivos",
        "Operación y demo",
    ]
)


# ---------------------------------------------------------------------------
# 1. Resumen ejecutivo
# ---------------------------------------------------------------------------

with tab_resumen:
    kpis_payload = bundle.get("kpis") or {"kpis": [], "resumen": {}}
    kpis = pd.DataFrame(kpis_payload.get("kpis", []))
    resumen_kpis = kpis_payload.get("resumen", {})

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Usuarios analizados", f"{len(data):,}")
    m2.metric("Segmentos", data["cluster"].nunique())
    m3.metric("Gasto promedio", f"${data['gasto_mensual'].mean():,.0f}")
    m4.metric("Usuarios de alto valor", resumen_kpis.get("usuarios_alto_valor", "N/D"))
    m5.metric(
        "Valor en riesgo",
        resumen_kpis.get("usuarios_valor_en_riesgo", "N/D"),
        help="Categoría de negocio; no equivale a churn observado.",
    )

    col_a, col_b = st.columns([1, 1])
    conteo = (
        df.groupby(["cluster", "segmento"])
        .size()
        .reset_index(name="usuarios")
    )
    conteo["porcentaje_total"] = conteo["usuarios"] / len(data) * 100

    with col_a:
        fig_segmentos = px.bar(
            conteo,
            x="segmento",
            y="usuarios",
            text="usuarios",
            title="Usuarios por segmento",
            labels={"segmento": "Segmento", "usuarios": "Usuarios"},
        )
        fig_segmentos.update_layout(showlegend=False, xaxis_tickangle=-15)
        st.plotly_chart(fig_segmentos, width="stretch")
        st.caption(
            "Los porcentajes se calculan sobre los 300 usuarios, incluso cuando "
            "se aplica un filtro en la barra lateral."
        )

    with col_b:
        perfil_gasto = (
            df.groupby(["cluster", "segmento"], as_index=False)["gasto_mensual"]
            .mean()
        )
        fig_gasto = px.bar(
            perfil_gasto,
            x="segmento",
            y="gasto_mensual",
            text_auto=".0f",
            title="Gasto mensual promedio por segmento",
            labels={"segmento": "Segmento", "gasto_mensual": "Gasto promedio"},
        )
        fig_gasto.update_layout(showlegend=False, xaxis_tickangle=-15)
        st.plotly_chart(fig_gasto, width="stretch")

    if not kpis.empty:
        st.subheader("Matriz de engagement y valor del cliente")
        matriz = kpis.pivot(
            index="nivel_engagement",
            columns="valor_cliente",
            values="usuarios",
        ).fillna(0)
        orden_filas = [valor for valor in ["alto", "medio", "bajo"] if valor in matriz.index]
        orden_columnas = [
            valor
            for valor in ["alto_valor", "valor_medio", "valor_en_riesgo"]
            if valor in matriz.columns
        ]
        matriz = matriz.reindex(index=orden_filas, columns=orden_columnas)
        fig_matriz = px.imshow(
            matriz,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Blues",
            labels={"x": "Valor del cliente", "y": "Engagement", "color": "Usuarios"},
        )
        st.plotly_chart(fig_matriz, width="stretch")

    st.subheader("Perfiles y acciones sugeridas")
    columnas = st.columns(len(clusters_seleccionados))
    for columna, cluster in zip(columnas, sorted(clusters_seleccionados)):
        perfil = segmentos[int(cluster)]
        cantidad = int((data["cluster"] == cluster).sum())
        porcentaje = cantidad / len(data) * 100
        with columna:
            st.markdown(f"#### {perfil['nombre']}")
            st.metric("Usuarios", cantidad, f"{porcentaje:.1f}% del total")
            st.write(perfil["descripcion"])
            st.info(f"Acción: {perfil['accion']}")

    st.warning(
        "Las acciones propuestas son recomendaciones analíticas. El proyecto no "
        "mide todavía el efecto causal ni el retorno económico de una campaña."
    )


# ---------------------------------------------------------------------------
# 2. Datos, calidad y pipeline
# ---------------------------------------------------------------------------

with tab_pipeline:
    pipeline_payload = bundle.get("pipeline") or {}
    manifest = pipeline_payload.get("manifest", {})
    calidad = bundle.get("calidad") or {}

    if not manifest:
        st.warning("No se encontró el manifiesto de la última ejecución.")
    else:
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Estado", manifest.get("status", "N/D"))
        p2.metric("Run ID", manifest.get("run_id", "N/D"))
        p3.metric("Duración", f"{manifest.get('duration_seconds', 0):.2f} s")
        p4.metric("Errores de validación", manifest.get("validation", {}).get("errors", "N/D"))

        st.subheader("Recorrido end-to-end")
        etapas = [
            ("1", "CSV + PostgreSQL", "Fuentes"),
            ("2", "Validación estructural", "0 errores"),
            ("3", "Join one-to-one", "300 IDs compartidos"),
            ("4", "Limpieza y features", "16 → 26 columnas"),
            ("5", "KMeans + PCA", "3 segmentos"),
            ("6", "Regresión + clasificación", "Selección por CV"),
            ("7", "API + dashboard", "Publicación completada"),
        ]
        columnas_etapas = st.columns(len(etapas))
        for columna, (numero, titulo, detalle) in zip(columnas_etapas, etapas):
            with columna:
                st.success(f"{numero}. {titulo}")
                st.caption(detalle)

        integracion = manifest.get("integration", {})
        st.subheader("Integración de fuentes")
        i1, i2, i3, i4, i5 = st.columns(5)
        i1.metric("Filas streaming", integracion.get("filas_streaming", "N/D"))
        i2.metric("Filas PostgreSQL", integracion.get("filas_perfil", "N/D"))
        i3.metric("IDs compartidos", integracion.get("ids_compartidos", "N/D"))
        i4.metric("Solo en una fuente", integracion.get("ids_solo_streaming", 0) + integracion.get("ids_solo_perfil", 0))
        i5.metric("Cardinalidad", integracion.get("cardinalidad", "N/D"))

    if calidad:
        st.subheader("Calidad antes y después")
        limpieza = calidad.get("limpieza", {})
        nulos_finales = sum(calidad.get("dataset_analitico", {}).get("nulos", {}).values())
        fuera_rango = sum(limpieza.get("valores_fuera_rango", {}).values())
        outliers = limpieza.get("outliers_iqr_detectados", {})
        total_outliers = sum(outliers.values())

        tabla_calidad = pd.DataFrame(
            [
                {
                    "Indicador": "Filas",
                    "Antes": calidad.get("filas_originales"),
                    "Después": calidad.get("filas_finales"),
                },
                {
                    "Indicador": "Columnas",
                    "Antes": calidad.get("columnas_originales"),
                    "Después": calidad.get("columnas_finales"),
                },
                {
                    "Indicador": "Nulos",
                    "Antes": sum(limpieza.get("nulos_antes", {}).values()),
                    "Después": nulos_finales,
                },
                {
                    "Indicador": "Valores fuera de rango",
                    "Antes": fuera_rango,
                    "Después": 0,
                },
                {
                    "Indicador": "Duplicados exactos",
                    "Antes": limpieza.get("duplicados_exactos_eliminados", 0),
                    "Después": 0,
                },
            ]
        )
        st.dataframe(tabla_calidad, width="stretch", hide_index=True)

        c_memoria, c_outliers = st.columns(2)
        with c_memoria:
            memoria = calidad.get("memoria_mb", {})
            original = float(memoria.get("original_16_columnas", 0))
            optimizada = float(memoria.get("base_limpia_16_columnas", 0))
            reduccion = (1 - optimizada / original) * 100 if original else 0
            st.metric("Reducción de memoria", f"{reduccion:.1f}%")
            st.write(f"Base original: **{original:.4f} MB**")
            st.write(f"Base optimizada: **{optimizada:.4f} MB**")

        with c_outliers:
            outliers_df = pd.DataFrame(
                [
                    {"Variable": _nombre_feature(variable), "Outliers": cantidad}
                    for variable, cantidad in outliers.items()
                    if cantidad > 0
                ]
            )
            if outliers_df.empty:
                st.success("No se detectaron posibles outliers mediante IQR.")
            else:
                fig_outliers = px.bar(
                    outliers_df,
                    x="Variable",
                    y="Outliers",
                    text="Outliers",
                    title=f"Posibles outliers detectados: {total_outliers}",
                )
                st.plotly_chart(fig_outliers, width="stretch")

        st.info(
            "Los outliers se detectan y reportan, pero no se eliminan de forma "
            "automática: pueden representar clientes reales de alto consumo o gasto."
        )
        with st.expander("Variables derivadas generadas"):
            st.write([_nombre_feature(nombre) for nombre in calidad.get("variables_derivadas", [])])

    st.subheader("Garantías de ejecución")
    st.markdown(
        "- La integración exige una relación **one-to-one** por `id_cliente`.\n"
        "- Los errores estructurales detienen el pipeline; las anomalías corregibles se reportan.\n"
        "- Los modelos solo reciben el dataset base después de la validación posterior.\n"
        "- Los artefactos se publican después de aprobar todas las etapas.\n"
        "- `pipeline_manifest.json` funciona como marca de ejecución completa."
    )


# ---------------------------------------------------------------------------
# 3. Segmentación
# ---------------------------------------------------------------------------

with tab_segmentacion:
    s1, s2, s3 = st.columns(3)
    s1.metric("K óptimo", metricas_kmeans["k_optimo"])
    s2.metric("Silhouette", f"{metricas_kmeans['silhouette_score']:.3f}")
    s3.metric("Varianza PCA (2 componentes)", f"{metricas_kmeans['varianza_pca']:.1%}")

    st.warning(
        "El Silhouette de 0,231 indica separación moderada-baja: los perfiles son "
        "interpretables, pero existe solapamiento y no representan grupos naturales perfectos."
    )
    st.info(
        "PCA se usa solo para visualización. Dos componentes conservan cerca del 44,9% "
        "de la varianza; la vista 2D omite más de la mitad de la información original."
    )

    codo, pca_col = st.columns(2)
    with codo:
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
            x=metricas_kmeans["k_optimo"],
            line_dash="dash",
            annotation_text=f"k={metricas_kmeans['k_optimo']}",
        )
        fig_codo.update_layout(
            title="Selección de k: codo y Silhouette",
            xaxis_title="Número de clusters",
            yaxis_title="Inercia",
            yaxis2=dict(title="Silhouette", overlaying="y", side="right"),
            legend=dict(orientation="h"),
        )
        st.plotly_chart(fig_codo, width="stretch")

    with pca_col:
        fig_pca = px.scatter(
            df,
            x="pc1",
            y="pc2",
            color="segmento",
            hover_data=["id_cliente", "gasto_mensual", "porcentaje_finalizacion"],
            title="Proyección PCA de usuarios",
            labels={"pc1": "PC1", "pc2": "PC2", "segmento": "Segmento"},
        )
        st.plotly_chart(fig_pca, width="stretch")

    variables = [
        "horas_consumo_mensual",
        "gasto_mensual",
        "cantidad_contenidos_vistos",
        "sesiones_semana",
        "porcentaje_finalizacion",
        "tiempo_promedio_sesion_min",
        "cantidad_generos_consumidos",
        "porcentaje_uso_promociones",
        "antiguedad_cliente_meses",
    ]
    perfil = df.groupby("cluster")[variables].mean()
    perfil_normalizado = _normalizar_columnas(perfil)

    st.subheader("Perfil relativo de los segmentos")
    if len(perfil) < 2:
        st.info("Selecciona al menos dos segmentos para comparar sus perfiles relativos.")
    else:
        fig_heatmap = go.Figure(
            go.Heatmap(
                z=perfil_normalizado.values,
                x=[_nombre_feature(columna) for columna in perfil_normalizado.columns],
                y=[_etiqueta_cluster(int(cluster), segmentos) for cluster in perfil_normalizado.index],
                text=perfil.round(1).values,
                texttemplate="%{text}",
                colorscale="Blues",
                colorbar=dict(title="Nivel relativo"),
            )
        )
        fig_heatmap.update_layout(xaxis_tickangle=-25)
        st.plotly_chart(fig_heatmap, width="stretch")

    perfil_tabla = perfil.copy().round(2)
    perfil_tabla.index = [
        _etiqueta_cluster(int(cluster), segmentos)
        for cluster in perfil_tabla.index
    ]
    perfil_tabla.columns = [_nombre_feature(columna) for columna in perfil_tabla.columns]
    st.dataframe(perfil_tabla, width="stretch")


# ---------------------------------------------------------------------------
# 4. Modelos supervisados
# ---------------------------------------------------------------------------

with tab_modelos:
    metricas_supervisado = bundle.get("supervisado")
    evidencia = bundle.get("evidencia") or {}

    if not metricas_supervisado:
        st.warning("No están disponibles los artefactos de modelos supervisados.")
    else:
        regresion = metricas_supervisado["regresion"]
        clasificacion = metricas_supervisado["clasificacion"]
        mejor_reg = regresion["mejor_modelo"]
        mejor_clf = clasificacion["mejor_modelo"]
        reg_ganador = regresion[mejor_reg]
        clf_ganador = clasificacion[mejor_clf]

        st.subheader("Regresión - gasto mensual")
        r1, r2, r3, r4, r5 = st.columns(5)
        r1.metric("Modelo ganador", mejor_reg.replace("_", " ").title())
        r2.metric("CV R²", f"{reg_ganador['cv_r2_mean']:.3f}")
        r3.metric("Test R²", f"{reg_ganador['r2']:.3f}")
        r4.metric("MAE", f"{reg_ganador['mae']:.2f}")
        r5.metric("RMSE", f"{reg_ganador['rmse']:.2f}")

        modelos_reg = pd.DataFrame(
            [
                {
                    "Modelo": nombre.replace("_", " ").title(),
                    "CV R²": valores["cv_r2_mean"],
                    "Test R²": valores["r2"],
                    "MAE": valores["mae"],
                    "RMSE": valores["rmse"],
                }
                for nombre, valores in regresion.items()
                if isinstance(valores, dict)
            ]
        )
        fig_reg_comp = px.bar(
            modelos_reg.melt(
                id_vars="Modelo",
                value_vars=["CV R²", "Test R²"],
                var_name="Evaluación",
                value_name="R²",
            ),
            x="Modelo",
            y="R²",
            color="Evaluación",
            barmode="group",
            title="Comparación de modelos de regresión",
        )
        st.plotly_chart(fig_reg_comp, width="stretch")

        pred_reg = pd.DataFrame(evidencia.get("regression_predictions", []))
        if not pred_reg.empty:
            c_real, c_residuo = st.columns(2)
            with c_real:
                fig_real = px.scatter(
                    pred_reg,
                    x="valor_real",
                    y="valor_predicho",
                    hover_data=["id_cliente"],
                    title="Gasto real vs. predicho (holdout)",
                    labels={"valor_real": "Valor real", "valor_predicho": "Valor predicho"},
                )
                minimo = min(pred_reg["valor_real"].min(), pred_reg["valor_predicho"].min())
                maximo = max(pred_reg["valor_real"].max(), pred_reg["valor_predicho"].max())
                fig_real.add_shape(
                    type="line",
                    x0=minimo,
                    y0=minimo,
                    x1=maximo,
                    y1=maximo,
                    line=dict(dash="dash"),
                )
                st.plotly_chart(fig_real, width="stretch")
            with c_residuo:
                fig_residuo = px.histogram(
                    pred_reg,
                    x="residuo",
                    nbins=15,
                    title="Distribución de residuos",
                    labels={"residuo": "Real - predicho"},
                )
                st.plotly_chart(fig_residuo, width="stretch")

        importancia_reg = pd.DataFrame(evidencia.get("regression_feature_importance", []))
        if not importancia_reg.empty:
            importancia_reg["Variable"] = importancia_reg["feature"].map(_nombre_feature)
            fig_imp_reg = px.bar(
                importancia_reg.head(10).sort_values("importance"),
                x="importance",
                y="Variable",
                orientation="h",
                title="Variables más influyentes en el gasto",
                labels={"importance": "Importancia"},
            )
            st.plotly_chart(fig_imp_reg, width="stretch")

        st.caption(
            "R²=0,759 significa que el modelo explica aproximadamente 75,9% de la "
            "variabilidad del gasto en el holdout; no equivale a 75,9% de precisión. "
            "MAE expresa el error medio y RMSE penaliza más los errores grandes."
        )

        st.divider()
        st.subheader("Clasificación - proxy de bajo compromiso")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Modelo ganador", mejor_clf.replace("_", " ").title())
        c2.metric("CV F1", f"{clf_ganador['cv_f1_mean']:.3f}")
        c3.metric("Test F1", f"{clf_ganador['f1']:.3f}")
        c4.metric("Recall", f"{clf_ganador['recall']:.3f}")
        c5.metric("ROC-AUC", f"{clf_ganador['roc_auc']:.3f}")

        modelos_clf = pd.DataFrame(
            [
                {
                    "Modelo": nombre.replace("_", " ").title(),
                    "Precision": valores["precision"],
                    "Recall": valores["recall"],
                    "F1": valores["f1"],
                    "ROC-AUC": valores["roc_auc"],
                }
                for nombre, valores in clasificacion.items()
                if isinstance(valores, dict)
            ]
        )
        fig_clf_comp = px.bar(
            modelos_clf.melt(
                id_vars="Modelo",
                value_vars=["Precision", "Recall", "F1", "ROC-AUC"],
                var_name="Métrica",
                value_name="Valor",
            ),
            x="Modelo",
            y="Valor",
            color="Métrica",
            barmode="group",
            title="Trade-offs de clasificación",
        )
        st.plotly_chart(fig_clf_comp, width="stretch")

        cm = np.asarray(clf_ganador["confusion_matrix"])
        pred_clf = pd.DataFrame(evidencia.get("classification_predictions", []))
        cm_col, prob_col = st.columns(2)
        with cm_col:
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                x=["Predicho: sin riesgo", "Predicho: riesgo"],
                y=["Real: sin riesgo", "Real: riesgo"],
                color_continuous_scale="Blues",
                title="Matriz de confusión",
            )
            st.plotly_chart(fig_cm, width="stretch")
        with prob_col:
            if not pred_clf.empty:
                pred_clf["clase_real"] = pred_clf["valor_real"].map(
                    {0: "Sin riesgo", 1: "Riesgo"}
                )
                fig_prob = px.histogram(
                    pred_clf,
                    x="probabilidad_riesgo",
                    color="clase_real",
                    nbins=15,
                    barmode="overlay",
                    title="Probabilidad de riesgo por clase real",
                    labels={"probabilidad_riesgo": "Probabilidad de riesgo"},
                )
                st.plotly_chart(fig_prob, width="stretch")

        importancia_clf = pd.DataFrame(evidencia.get("classification_feature_importance", []))
        if not importancia_clf.empty:
            importancia_clf["Variable"] = importancia_clf["feature"].map(_nombre_feature)
            fig_imp_clf = px.bar(
                importancia_clf.head(10).sort_values("importance"),
                x="importance",
                y="Variable",
                orientation="h",
                title="Variables más influyentes en bajo compromiso",
                labels={"importance": "Importancia"},
            )
            st.plotly_chart(fig_imp_clf, width="stretch")

        umbral_finalizacion = clasificacion.get("umbral_finalizacion")
        umbral_sesiones = clasificacion.get("umbral_sesiones")
        st.info(
            f"La etiqueta proxy se construyó con finalización < {umbral_finalizacion:.0f}% "
            f"y sesiones < {umbral_sesiones:.0f} por semana. Los umbrales se calcularon "
            "solo con train y esas variables se excluyeron del clasificador para evitar fuga."
        )
        st.warning(
            "Bajo compromiso es una etiqueta proxy construida. No representa churn, "
            "cancelación observada ni causalidad."
        )


# ---------------------------------------------------------------------------
# 5. Operación y demo
# ---------------------------------------------------------------------------

with tab_operacion:
    st.subheader("Consulta de usuario existente")
    id_seleccionado = st.selectbox(
        "ID del cliente",
        options=sorted(data["id_cliente"].unique()),
    )
    usuario = data[data["id_cliente"] == id_seleccionado].iloc[0]
    cluster_usuario = int(usuario["cluster"])
    centroide = centroides_indexados.loc[cluster_usuario]
    perfil_segmento = segmentos[cluster_usuario]

    u1, u2, u3 = st.columns(3)
    u1.metric("Segmento", perfil_segmento["nombre"])
    u2.metric("Gasto mensual", f"${usuario['gasto_mensual']:,.0f}")
    u3.metric("Finalización", f"{usuario['porcentaje_finalizacion']:.1f}%")
    st.info(f"Acción sugerida: {perfil_segmento['accion']}")

    variables_usuario = [
        "horas_consumo_mensual",
        "gasto_mensual",
        "cantidad_contenidos_vistos",
        "sesiones_semana",
        "porcentaje_finalizacion",
        "antiguedad_cliente_meses",
        "porcentaje_uso_promociones",
    ]
    comparacion = []
    for variable in variables_usuario:
        promedio = float(centroide[variable])
        valor = float(usuario[variable])
        diferencia = ((valor - promedio) / promedio * 100) if promedio else np.nan
        comparacion.append(
            {
                "Variable": _nombre_feature(variable),
                "Usuario": round(valor, 2),
                "Promedio del segmento": round(promedio, 2),
                "Diferencia (%)": round(diferencia, 1) if np.isfinite(diferencia) else None,
            }
        )
    st.dataframe(pd.DataFrame(comparacion), width="stretch", hide_index=True)

    with st.expander("Explorar tabla de usuarios filtrados"):
        st.dataframe(df, width="stretch", height=320)

    st.divider()
    st.subheader("Demo end-to-end para un usuario nuevo")
    st.caption(
        "El formulario llama tres endpoints: estima gasto, asigna segmento usando ese "
        "gasto estimado y calcula el proxy de bajo compromiso."
    )

    with st.form("simulador_completo"):
        f1, f2, f3 = st.columns(3)
        with f1:
            horas = st.number_input("Horas de consumo mensual", 0.0, 200.0, 40.0)
            contenidos = st.number_input("Contenidos vistos", 0, 200, 20)
            sesiones = st.number_input("Sesiones por semana", 0, 30, 6)
            finalizacion = st.number_input("Finalización (%)", 0.0, 100.0, 60.0)
            soporte = st.number_input("Interacciones con soporte", 0, 30, 2)
        with f2:
            tiempo = st.number_input("Duración promedio de sesión (min)", 0.0, 400.0, 100.0)
            generos = st.number_input("Géneros consumidos", 0, 20, 5)
            promociones_pct = st.slider("Uso de promociones (%)", 0, 100, 30)
            antiguedad = st.number_input("Antigüedad (meses)", 0, 150, 30)
            distancia = st.number_input("Distancia promedio a red (km)", 0.0, 200.0, 20.0)
        with f3:
            edad = st.number_input("Edad", 13, 100, 35)
            dispositivos = st.number_input("Dispositivos registrados", 0, 10, 2)
            app_pct = st.slider("Uso de app móvil (%)", 0, 100, 50)
            perfiles = st.number_input("Perfiles creados", 0, 10, 3)

        enviado = st.form_submit_button("Ejecutar análisis completo", width="stretch")

    if enviado:
        base_payload = {
            "horas_consumo_mensual": horas,
            "cantidad_contenidos_vistos": contenidos,
            "sesiones_semana": sesiones,
            "porcentaje_finalizacion": finalizacion,
            "tiempo_promedio_sesion_min": tiempo,
            "cantidad_generos_consumidos": generos,
            "porcentaje_uso_promociones": promociones_pct / 100,
            "antiguedad_cliente_meses": antiguedad,
            "edad": edad,
            "dispositivos_registrados": dispositivos,
            "porcentaje_uso_app_movil": app_pct / 100,
            "cantidad_perfiles_creados": perfiles,
            "interacciones_mensuales_soporte": soporte,
            "distancia_promedio_red_km": distancia,
        }

        gasto_resultado, error_gasto = _post_json("/predict-gasto", base_payload)
        riesgo_resultado, error_riesgo = _post_json("/predict-riesgo", base_payload)

        cluster_resultado = None
        error_cluster = None
        if gasto_resultado:
            payload_cluster = {
                **base_payload,
                "gasto_mensual": gasto_resultado["gasto_mensual_predicho"],
            }
            cluster_resultado, error_cluster = _post_json("/predict", payload_cluster)

        if error_gasto or error_riesgo or error_cluster:
            for nombre, error in [
                ("gasto", error_gasto),
                ("riesgo", error_riesgo),
                ("segmentación", error_cluster),
            ]:
                if error:
                    st.error(f"Error en {nombre}: {error}")
        else:
            cluster = int(cluster_resultado["cluster"])
            gasto = float(gasto_resultado["gasto_mensual_predicho"])
            riesgo = int(riesgo_resultado["riesgo_bajo_compromiso"])
            probabilidad = float(riesgo_resultado["probabilidad"])
            perfil_nuevo = segmentos[cluster]

            d1, d2, d3 = st.columns(3)
            d1.metric("Segmento estimado", perfil_nuevo["nombre"])
            d2.metric("Gasto mensual estimado", f"${gasto:,.0f}")
            d3.metric(
                "Bajo compromiso",
                "Sí" if riesgo else "No",
                f"{probabilidad:.1%} de probabilidad",
                delta_color="inverse" if riesgo else "normal",
            )
            st.success(f"Acción sugerida: {perfil_nuevo['accion']}")
            st.caption(
                "El segmento se estima usando el gasto predicho porque el gasto mensual "
                "forma parte de las 15 variables originales del KMeans."
            )
