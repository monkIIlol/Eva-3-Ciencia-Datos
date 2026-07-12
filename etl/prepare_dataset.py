"""
Preparación del dataset analítico para modelos de Machine Learning.

Este módulo toma el dataset consolidado generado por el ETL, aplica limpieza,
transformaciones avanzadas con Pandas, feature engineering y genera archivos
listos para entrenamiento, análisis de negocio y documentación.

Entradas:
- data/data_consolidada.csv

Salidas:
- data/dataset_modelo.csv
- data/kpis_negocio.csv
- data/reporte_calidad.json
"""

from pathlib import Path
import json
import logging
import os

import numpy as np
import pandas as pd


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


RUTA_ENTRADA = Path("data/data_consolidada.csv")
RUTA_DATASET_MODELO = Path("data/dataset_modelo.csv")
RUTA_KPIS = Path("data/kpis_negocio.csv")
RUTA_REPORTE = Path("data/reporte_calidad.json")


COLUMNAS_NUMERICAS_ESPERADAS = [
    "id_cliente",
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


COLUMNAS_IMPUTABLES = [
    columna for columna in COLUMNAS_NUMERICAS_ESPERADAS
    if columna != "id_cliente"
]


COLUMNAS_OUTLIERS_IQR = [
    "horas_consumo_mensual",
    "gasto_mensual",
    "cantidad_contenidos_vistos",
    "sesiones_semana",
    "tiempo_promedio_sesion_min",
    "antiguedad_cliente_meses",
    "distancia_promedio_red_km",
]


def cargar_dataset(ruta: Path = RUTA_ENTRADA) -> pd.DataFrame:
    """Carga el dataset consolidado generado por el proceso ETL."""
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró {ruta}. Primero se debe ejecutar etl/extract.py."
        )

    df = pd.read_csv(ruta)
    logger.info(
        "Dataset consolidado cargado: %s filas, %s columnas",
        df.shape[0],
        df.shape[1],
    )
    return df


def validar_columnas_base(df: pd.DataFrame) -> None:
    """Valida que el dataset tenga las columnas mínimas necesarias."""
    faltantes = [col for col in COLUMNAS_NUMERICAS_ESPERADAS if col not in df.columns]

    if faltantes:
        raise ValueError(f"Faltan columnas requeridas para modelado: {faltantes}")


def tratar_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trata valores atípicos mediante el método IQR.

    En vez de eliminar registros, aplica winsorización con clip para conservar
    la cantidad de filas y reducir el impacto de valores extremos.
    """
    df = df.copy()
    reporte_outliers = {}

    for columna in COLUMNAS_OUTLIERS_IQR:
        q1 = df[columna].quantile(0.25)
        q3 = df[columna].quantile(0.75)
        iqr = q3 - q1

        if pd.isna(iqr) or iqr == 0:
            reporte_outliers[columna] = {
                "outliers_detectados": 0,
                "limite_inferior": None,
                "limite_superior": None,
            }
            continue

        limite_inferior = max(0, q1 - 1.5 * iqr)
        limite_superior = q3 + 1.5 * iqr

        outliers_detectados = int(
            ((df[columna] < limite_inferior) | (df[columna] > limite_superior)).sum()
        )

        df[columna] = df[columna].clip(
            lower=limite_inferior,
            upper=limite_superior,
        )

        reporte_outliers[columna] = {
            "outliers_detectados": outliers_detectados,
            "limite_inferior": round(float(limite_inferior), 4),
            "limite_superior": round(float(limite_superior), 4),
        }

    df.attrs["outliers_iqr"] = reporte_outliers
    return df


def limpiar_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia el dataset base.

    Reglas principales:
    - id_cliente no se imputa, porque es un identificador.
    - Se eliminan registros sin id_cliente.
    - Se eliminan duplicados por id_cliente.
    - Se imputan variables numéricas con mediana.
    - Se corrigen rangos de negocio.
    - Se tratan outliers mediante IQR.
    """
    df = df.copy()

    filas_iniciales = len(df)

    # El ID se trata como identificador, no como variable imputable.
    df["id_cliente"] = pd.to_numeric(df["id_cliente"], errors="coerce")
    ids_nulos_eliminados = int(df["id_cliente"].isna().sum())
    df = df.dropna(subset=["id_cliente"])
    df["id_cliente"] = df["id_cliente"].astype(int)

    filas_despues_ids = len(df)
    df = df.drop_duplicates(subset=["id_cliente"], keep="first")
    duplicados_eliminados = filas_despues_ids - len(df)

    for columna in COLUMNAS_IMPUTABLES:
        df[columna] = pd.to_numeric(df[columna], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)

    nulos_antes = df[COLUMNAS_NUMERICAS_ESPERADAS].isna().sum().astype(int).to_dict()

    for columna in COLUMNAS_IMPUTABLES:
        if df[columna].isna().any():
            mediana = df[columna].median()

            if pd.isna(mediana):
                mediana = 0

            df[columna] = df[columna].fillna(mediana)

    # Reglas de consistencia y rangos de negocio.
    df["sesiones_semana"] = df["sesiones_semana"].clip(lower=1)
    df["horas_consumo_mensual"] = df["horas_consumo_mensual"].clip(lower=0.1)
    df["gasto_mensual"] = df["gasto_mensual"].clip(lower=0)
    df["cantidad_contenidos_vistos"] = df["cantidad_contenidos_vistos"].clip(lower=0)
    df["tiempo_promedio_sesion_min"] = df["tiempo_promedio_sesion_min"].clip(lower=0)
    df["cantidad_generos_consumidos"] = df["cantidad_generos_consumidos"].clip(lower=0)
    df["antiguedad_cliente_meses"] = df["antiguedad_cliente_meses"].clip(lower=0)
    df["edad"] = df["edad"].clip(lower=13, upper=100)
    df["dispositivos_registrados"] = df["dispositivos_registrados"].clip(lower=1)
    df["cantidad_perfiles_creados"] = df["cantidad_perfiles_creados"].clip(lower=1)
    df["interacciones_mensuales_soporte"] = df["interacciones_mensuales_soporte"].clip(lower=0)
    df["distancia_promedio_red_km"] = df["distancia_promedio_red_km"].clip(lower=0)

    df["porcentaje_finalizacion"] = df["porcentaje_finalizacion"].clip(lower=0, upper=100)
    df["porcentaje_uso_promociones"] = df["porcentaje_uso_promociones"].clip(lower=0, upper=1)
    df["porcentaje_uso_app_movil"] = df["porcentaje_uso_app_movil"].clip(lower=0, upper=1)

    df = tratar_outliers_iqr(df)

    if df["id_cliente"].isna().any():
        raise ValueError("id_cliente contiene valores nulos después de la limpieza.")

    if df["id_cliente"].duplicated().any():
        raise ValueError("id_cliente contiene duplicados después de la limpieza.")

    df.attrs["filas_iniciales"] = int(filas_iniciales)
    df.attrs["ids_nulos_eliminados"] = int(ids_nulos_eliminados)
    df.attrs["duplicados_eliminados"] = int(duplicados_eliminados)
    df.attrs["nulos_antes"] = nulos_antes

    logger.info(
        "Limpieza completada. IDs nulos eliminados: %s | Duplicados eliminados: %s",
        ids_nulos_eliminados,
        duplicados_eliminados,
    )

    return df


def crear_variables_derivadas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea variables derivadas para enriquecer el análisis y los modelos.

    Estas variables resumen comportamiento de consumo, valor económico,
    uso de soporte y nivel de compromiso del usuario.
    """
    df = df.copy()

    # Se estima el total mensual de sesiones para no mezclar una variable mensual
    # con una variable semanal.
    df["sesiones_mes_estimadas"] = df["sesiones_semana"] * 4

    df["contenidos_por_sesion"] = (
        df["cantidad_contenidos_vistos"] / df["sesiones_mes_estimadas"]
    ).round(3)

    df["gasto_por_hora"] = (
        df["gasto_mensual"] / df["horas_consumo_mensual"]
    ).round(3)

    df["minutos_totales_estimados"] = (
        df["horas_consumo_mensual"] * 60
    ).round(2)

    df["soporte_por_dispositivo"] = (
        df["interacciones_mensuales_soporte"] / df["dispositivos_registrados"]
    ).round(3)

    df["generos_por_contenido"] = np.where(
        df["cantidad_contenidos_vistos"] > 0,
        df["cantidad_generos_consumidos"] / df["cantidad_contenidos_vistos"],
        0,
    ).round(3)

    # Indicador descriptivo relativo.
    # Usa rankings percentiles para combinar variables con distintas escalas.
    df["engagement_score"] = (
        df["horas_consumo_mensual"].rank(pct=True) * 0.30
        + df["sesiones_semana"].rank(pct=True) * 0.25
        + df["porcentaje_finalizacion"].rank(pct=True) * 0.25
        + df["antiguedad_cliente_meses"].rank(pct=True) * 0.20
    ).round(4)

    df["nivel_engagement"] = pd.cut(
        df["engagement_score"],
        bins=[-0.01, 0.33, 0.66, 1.0],
        labels=["bajo", "medio", "alto"],
    ).astype(str)

    df["cliente_antiguo"] = np.where(df["antiguedad_cliente_meses"] >= 36, 1, 0)
    df["uso_promociones_alto"] = np.where(df["porcentaje_uso_promociones"] >= 0.50, 1, 0)

    df["valor_cliente"] = np.select(
        [
            (df["gasto_mensual"] >= df["gasto_mensual"].quantile(0.75))
            & (df["engagement_score"] >= df["engagement_score"].quantile(0.60)),
            (df["gasto_mensual"] <= df["gasto_mensual"].quantile(0.25))
            | (df["engagement_score"] <= df["engagement_score"].quantile(0.33)),
        ],
        ["alto_valor", "valor_en_riesgo"],
        default="valor_medio",
    )

    logger.info("Variables derivadas creadas correctamente.")
    return df


def optimizar_tipos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimiza tipos de datos para reducir memoria.

    Esto es útil como práctica de escalabilidad para datasets de mayor tamaño.
    """
    df = df.copy()

    columnas_enteras = [
        "id_cliente",
        "cantidad_contenidos_vistos",
        "sesiones_semana",
        "tiempo_promedio_sesion_min",
        "cantidad_generos_consumidos",
        "antiguedad_cliente_meses",
        "edad",
        "dispositivos_registrados",
        "cantidad_perfiles_creados",
        "interacciones_mensuales_soporte",
        "cliente_antiguo",
        "uso_promociones_alto",
        "sesiones_mes_estimadas",
    ]

    for columna in columnas_enteras:
        if columna in df.columns:
            df[columna] = pd.to_numeric(df[columna], downcast="integer")

    columnas_float = df.select_dtypes(include=["float64"]).columns
    for columna in columnas_float:
        df[columna] = pd.to_numeric(df[columna], downcast="float")

    return df


def generar_kpis_negocio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera KPIs por nivel de engagement y valor de cliente.

    Usa groupby con múltiples dimensiones y agregaciones.
    """
    kpis = (
        df.groupby(["nivel_engagement", "valor_cliente"], observed=False)
        .agg(
            usuarios=("id_cliente", "nunique"),
            gasto_promedio=("gasto_mensual", "mean"),
            consumo_promedio_horas=("horas_consumo_mensual", "mean"),
            finalizacion_promedio=("porcentaje_finalizacion", "mean"),
            promociones_promedio=("porcentaje_uso_promociones", "mean"),
            antiguedad_promedio=("antiguedad_cliente_meses", "mean"),
        )
        .reset_index()
    )

    kpis["porcentaje_usuarios"] = (kpis["usuarios"] / len(df) * 100).round(2)

    columnas_promedio = [
        "gasto_promedio",
        "consumo_promedio_horas",
        "finalizacion_promedio",
        "promociones_promedio",
        "antiguedad_promedio",
    ]
    kpis[columnas_promedio] = kpis[columnas_promedio].round(2)

    logger.info("KPIs de negocio generados: %s filas", len(kpis))
    return kpis


def generar_reporte_calidad(df_original: pd.DataFrame, df_final: pd.DataFrame) -> dict:
    """Genera un reporte resumido de calidad y transformación del dataset."""
    memoria_original_mb = df_original.memory_usage(deep=True).sum() / 1024**2
    memoria_final_mb = df_final.memory_usage(deep=True).sum() / 1024**2

    reporte = {
        "filas_originales": int(df_original.shape[0]),
        "columnas_originales": int(df_original.shape[1]),
        "filas_finales": int(df_final.shape[0]),
        "columnas_finales": int(df_final.shape[1]),
        "ids_nulos_eliminados": int(df_final.attrs.get("ids_nulos_eliminados", 0)),
        "duplicados_eliminados": int(df_final.attrs.get("duplicados_eliminados", 0)),
        "nulos_antes_limpieza": df_final.attrs.get("nulos_antes", {}),
        "nulos_finales": df_final.isna().sum().astype(int).to_dict(),
        "outliers_iqr": df_final.attrs.get("outliers_iqr", {}),
        "memoria_original_mb": round(float(memoria_original_mb), 4),
        "memoria_final_mb": round(float(memoria_final_mb), 4),
        "variables_derivadas": [
            "sesiones_mes_estimadas",
            "contenidos_por_sesion",
            "gasto_por_hora",
            "minutos_totales_estimados",
            "soporte_por_dispositivo",
            "generos_por_contenido",
            "engagement_score",
            "nivel_engagement",
            "cliente_antiguo",
            "uso_promociones_alto",
            "valor_cliente",
        ],
        "nota_metodologica": (
            "id_cliente se trata como identificador y no se imputa. "
            "Los outliers se tratan mediante IQR con winsorización para conservar registros."
        ),
    }

    return reporte


def preparar_dataset() -> pd.DataFrame:
    """Ejecuta el flujo completo de preparación del dataset analítico."""
    df_original = cargar_dataset()
    validar_columnas_base(df_original)

    df_limpio = limpiar_dataset(df_original)
    df_preparado = crear_variables_derivadas(df_limpio)
    df_preparado = optimizar_tipos(df_preparado)

    # Mantener atributos para el reporte después de copiar/optimizar el DataFrame.
    for clave, valor in df_limpio.attrs.items():
        df_preparado.attrs[clave] = valor

    kpis = generar_kpis_negocio(df_preparado)
    reporte = generar_reporte_calidad(df_original, df_preparado)

    RUTA_DATASET_MODELO.parent.mkdir(parents=True, exist_ok=True)

    df_preparado.to_csv(RUTA_DATASET_MODELO, index=False)
    kpis.to_csv(RUTA_KPIS, index=False)

    with open(RUTA_REPORTE, "w", encoding="utf-8") as archivo:
        json.dump(reporte, archivo, indent=4, ensure_ascii=False)

    logger.info("Dataset de modelado guardado en %s", RUTA_DATASET_MODELO)
    logger.info("KPIs de negocio guardados en %s", RUTA_KPIS)
    logger.info("Reporte de calidad guardado en %s", RUTA_REPORTE)

    return df_preparado


if __name__ == "__main__":
    preparar_dataset()