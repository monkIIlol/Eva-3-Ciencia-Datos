"""
Preparación del dataset base y del dataset analítico.

Esta etapa puede recibir directamente el DataFrame producido por la
integración. También conserva compatibilidad con el flujo anterior,
leyendo data/data_consolidada.csv cuando no se entrega un DataFrame.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.settings import Settings, settings
from etl.contracts import (
    BASE_COLUMN_ORDER,
    BASE_REQUIRED_COLUMNS,
    COLUMN_RANGES,
    DataQualityError,
    INTEGER_COLUMNS,
)
from etl.validate import validar_dataset_limpio


logger = logging.getLogger(__name__)


# Alias conservados para no romper los tests existentes.
RUTA_ENTRADA = settings.consolidated_dataset
RUTA_DATASET_BASE = settings.clean_base_dataset
RUTA_DATASET_ANALITICO = settings.analytical_dataset
RUTA_DATASET_MODELO = settings.legacy_model_dataset
RUTA_KPIS = settings.business_kpis
RUTA_REPORTE = settings.quality_report


VARIABLES_DERIVADAS = [
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
]


@dataclass
class PreparationResult:
    """Productos generados por la etapa de preparación."""

    dataset_base_limpio: pd.DataFrame
    dataset_analitico: pd.DataFrame
    kpis: pd.DataFrame
    reporte_calidad: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


def cargar_dataset(
    ruta: str | Path | None = None,
    config: Settings = settings,
) -> pd.DataFrame:
    """Carga el dataset consolidado para compatibilidad con el flujo anterior."""
    ruta_entrada = (
        Path(ruta)
        if ruta is not None
        else config.consolidated_dataset
    )

    if not ruta_entrada.exists():
        raise FileNotFoundError(
            f"No se encontró {ruta_entrada}. "
            "Primero debe ejecutarse la extracción e integración."
        )

    try:
        df = pd.read_csv(ruta_entrada)
    except (OSError, pd.errors.ParserError) as exc:
        raise DataQualityError(
            f"No fue posible cargar el dataset consolidado: {exc}"
        ) from exc

    if df.empty:
        raise DataQualityError(
            "El dataset consolidado está vacío."
        )

    logger.info(
        "Dataset consolidado cargado: %s filas, %s columnas.",
        df.shape[0],
        df.shape[1],
    )

    return df


def validar_columnas_base(df: pd.DataFrame) -> None:
    """Comprueba que el DataFrame contiene el esquema base completo."""
    faltantes = sorted(
        BASE_REQUIRED_COLUMNS - set(df.columns)
    )

    if faltantes:
        raise DataQualityError(
            "Faltan columnas requeridas para la preparación: "
            f"{faltantes}"
        )


def detectar_outliers_iqr(df: pd.DataFrame) -> dict[str, int]:
    """
    Cuenta posibles outliers mediante IQR.

    Esta función solo diagnostica. No modifica automáticamente valores,
    porque su tratamiento debe justificarse según cada variable.
    """
    resultado: dict[str, int] = {}

    for columna in BASE_COLUMN_ORDER:
        if columna == "id_cliente":
            continue

        serie = pd.to_numeric(
            df[columna],
            errors="coerce",
        ).dropna()

        if serie.empty:
            resultado[columna] = 0
            continue

        q1 = serie.quantile(0.25)
        q3 = serie.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            resultado[columna] = 0
            continue

        limite_inferior = q1 - 1.5 * iqr
        limite_superior = q3 + 1.5 * iqr

        cantidad = int(
            (
                (serie < limite_inferior)
                | (serie > limite_superior)
            ).sum()
        )

        resultado[columna] = cantidad

    return resultado


def limpiar_dataset_base(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Limpia las 16 variables originales del dataset.

    Políticas:
    - elimina registros exactamente duplicados;
    - nunca imputa id_cliente;
    - bloquea IDs inválidos o repetidos;
    - convierte las variables analíticas a número;
    - reemplaza infinitos;
    - transforma valores fuera del dominio en nulos;
    - imputa variables analíticas mediante mediana;
    - conserva el esquema base oficial.
    """
    validar_columnas_base(df)

    original = df[BASE_COLUMN_ORDER].copy()

    filas_iniciales = int(len(original))
    duplicados_exactos = int(original.duplicated().sum())

    limpio = original.drop_duplicates().copy()

    # El identificador tiene un tratamiento diferente:
    # no se imputa ni se corrige mediante estadísticas.
    ids = pd.to_numeric(
        limpio["id_cliente"],
        errors="coerce",
    )

    ids_invalidos = int(ids.isna().sum())

    if ids_invalidos > 0:
        raise DataQualityError(
            "No es posible preparar el dataset: "
            f"existen {ids_invalidos} IDs nulos o no numéricos."
        )

    ids_no_enteros = int(
        ((ids % 1) != 0).sum()
    )

    if ids_no_enteros > 0:
        raise DataQualityError(
            "id_cliente contiene valores no enteros."
        )

    ids_no_positivos = int(
        (ids <= 0).sum()
    )

    if ids_no_positivos > 0:
        raise DataQualityError(
            "id_cliente contiene valores menores o iguales a cero."
        )

    limpio["id_cliente"] = ids.astype("int64")

    ids_duplicados = int(
        limpio["id_cliente"].duplicated().sum()
    )

    if ids_duplicados > 0:
        raise DataQualityError(
            "Existen IDs duplicados con información potencialmente "
            f"conflictiva: {ids_duplicados}."
        )

    nulos_antes: dict[str, int] = {}
    valores_fuera_rango: dict[str, int] = {}
    valores_imputados: dict[str, int] = {}

    for columna in BASE_COLUMN_ORDER:
        if columna == "id_cliente":
            continue

        serie = pd.to_numeric(
            limpio[columna],
            errors="coerce",
        )

        serie = serie.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        nulos_antes[columna] = int(
            serie.isna().sum()
        )

        minimo, maximo = COLUMN_RANGES.get(
            columna,
            (None, None),
        )

        mascara_invalida = pd.Series(
            False,
            index=serie.index,
        )

        if minimo is not None:
            mascara_invalida |= serie < minimo

        if maximo is not None:
            mascara_invalida |= serie > maximo

        cantidad_fuera_rango = int(
            mascara_invalida.sum()
        )

        valores_fuera_rango[columna] = (
            cantidad_fuera_rango
        )

        # Los valores imposibles se consideran faltantes,
        # en lugar de recortarlos artificialmente al límite.
        serie = serie.mask(mascara_invalida)

        cantidad_a_imputar = int(
            serie.isna().sum()
        )

        valores_imputados[columna] = (
            cantidad_a_imputar
        )

        if cantidad_a_imputar > 0:
            mediana = serie.median()

            if pd.isna(mediana):
                raise DataQualityError(
                    f"No es posible imputar la columna {columna}: "
                    "no contiene valores válidos."
                )

            serie = serie.fillna(mediana)

        limpio[columna] = serie

    # Se restauran tipos enteros solo después de completar la limpieza.
    for columna in INTEGER_COLUMNS:
        if columna == "id_cliente":
            continue

        limpio[columna] = (
            limpio[columna]
            .round()
            .astype("int64")
        )

    limpio = limpio[BASE_COLUMN_ORDER].copy()

    metadata = {
        "filas_iniciales": filas_iniciales,
        "filas_limpias": int(len(limpio)),
        "duplicados_exactos_eliminados": duplicados_exactos,
        "nulos_antes": nulos_antes,
        "valores_fuera_rango": valores_fuera_rango,
        "valores_imputados": valores_imputados,
        "outliers_iqr_detectados": detectar_outliers_iqr(
            limpio
        ),
    }

    logger.info(
        "Limpieza completada: %s filas; "
        "%s duplicados exactos eliminados.",
        len(limpio),
        duplicados_exactos,
    )

    return limpio, metadata


def crear_variables_derivadas(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Construye las variables analíticas y de negocio."""
    analitico = df.copy()

    # Conversión aproximada de sesiones semanales a mensuales.
    sesiones_mensuales_estimadas = (
        analitico["sesiones_semana"] * 4.345
    )

    analitico["contenidos_por_sesion"] = (
        analitico["cantidad_contenidos_vistos"]
        .div(
            sesiones_mensuales_estimadas.replace(
                0,
                np.nan,
            )
        )
        .fillna(0)
        .round(3)
    )

    analitico["gasto_por_hora"] = (
        analitico["gasto_mensual"]
        .div(
            analitico[
                "horas_consumo_mensual"
            ].replace(0, np.nan)
        )
        .fillna(0)
        .round(3)
    )

    # Se conserva por compatibilidad y visualización,
    # pero no debe entrar junto con horas al entrenamiento.
    analitico["minutos_totales_estimados"] = (
        analitico["horas_consumo_mensual"] * 60
    ).round(2)

    analitico["soporte_por_dispositivo"] = (
        analitico["interacciones_mensuales_soporte"]
        .div(
            analitico[
                "dispositivos_registrados"
            ].replace(0, np.nan)
        )
        .fillna(0)
        .round(3)
    )

    analitico["generos_por_contenido"] = (
        analitico["cantidad_generos_consumidos"]
        .div(
            analitico[
                "cantidad_contenidos_vistos"
            ].replace(0, np.nan)
        )
        .fillna(0)
        .round(3)
    )

    analitico["engagement_score"] = (
        analitico[
            "horas_consumo_mensual"
        ].rank(pct=True) * 0.30
        + analitico[
            "sesiones_semana"
        ].rank(pct=True) * 0.25
        + analitico[
            "porcentaje_finalizacion"
        ].rank(pct=True) * 0.25
        + analitico[
            "antiguedad_cliente_meses"
        ].rank(pct=True) * 0.20
    ).round(4)

    analitico["nivel_engagement"] = (
        pd.cut(
            analitico["engagement_score"],
            bins=[-0.01, 0.33, 0.66, 1.0],
            labels=["bajo", "medio", "alto"],
        )
        .astype("string")
    )

    analitico["cliente_antiguo"] = (
        analitico[
            "antiguedad_cliente_meses"
        ] >= 36
    ).astype("int8")

    analitico["uso_promociones_alto"] = (
        analitico[
            "porcentaje_uso_promociones"
        ] >= 0.50
    ).astype("int8")

    gasto_q75 = analitico[
        "gasto_mensual"
    ].quantile(0.75)

    gasto_q25 = analitico[
        "gasto_mensual"
    ].quantile(0.25)

    engagement_q60 = analitico[
        "engagement_score"
    ].quantile(0.60)

    engagement_q33 = analitico[
        "engagement_score"
    ].quantile(0.33)

    analitico["valor_cliente"] = np.select(
        [
            (
                analitico["gasto_mensual"] >= gasto_q75
            )
            & (
                analitico[
                    "engagement_score"
                ] >= engagement_q60
            ),
            (
                analitico["gasto_mensual"] <= gasto_q25
            )
            | (
                analitico[
                    "engagement_score"
                ] <= engagement_q33
            ),
        ],
        [
            "alto_valor",
            "valor_en_riesgo",
        ],
        default="valor_medio",
    )

    logger.info(
        "Variables derivadas creadas correctamente."
    )

    return analitico


def optimizar_tipos(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Reduce el uso de memoria sin alterar los valores."""
    optimizado = df.copy()

    for columna in optimizado.columns:
        if columna in {
            "nivel_engagement",
            "valor_cliente",
        }:
            optimizado[columna] = (
                optimizado[columna].astype("category")
            )
            continue

        if pd.api.types.is_integer_dtype(
            optimizado[columna]
        ):
            optimizado[columna] = pd.to_numeric(
                optimizado[columna],
                downcast="integer",
            )

        elif pd.api.types.is_float_dtype(
            optimizado[columna]
        ):
            optimizado[columna] = pd.to_numeric(
                optimizado[columna],
                downcast="float",
            )

    return optimizado


def generar_kpis_negocio(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Genera KPIs por engagement y valor de cliente."""
    kpis = (
        df.groupby(
            [
                "nivel_engagement",
                "valor_cliente",
            ],
            observed=False,
        )
        .agg(
            usuarios=("id_cliente", "nunique"),
            gasto_promedio=("gasto_mensual", "mean"),
            consumo_promedio_horas=(
                "horas_consumo_mensual",
                "mean",
            ),
            finalizacion_promedio=(
                "porcentaje_finalizacion",
                "mean",
            ),
            promociones_promedio=(
                "porcentaje_uso_promociones",
                "mean",
            ),
            antiguedad_promedio=(
                "antiguedad_cliente_meses",
                "mean",
            ),
        )
        .reset_index()
    )

    kpis["porcentaje_usuarios"] = (
        kpis["usuarios"] / len(df) * 100
    ).round(2)

    columnas_promedio = [
        "gasto_promedio",
        "consumo_promedio_horas",
        "finalizacion_promedio",
        "promociones_promedio",
        "antiguedad_promedio",
    ]

    kpis[columnas_promedio] = (
        kpis[columnas_promedio].round(2)
    )

    return kpis


def generar_reporte_calidad(
    df_original: pd.DataFrame,
    df_base_limpio: pd.DataFrame,
    df_analitico: pd.DataFrame,
    metadata_limpieza: dict[str, Any],
) -> dict[str, Any]:
    """Construye el reporte previo y posterior a la preparación."""
    memoria_original = (
        df_original[BASE_COLUMN_ORDER]
        .memory_usage(deep=True)
        .sum()
        / 1024**2
    )

    memoria_base_limpia = (
        df_base_limpio
        .memory_usage(deep=True)
        .sum()
        / 1024**2
    )

    memoria_analitica = (
        df_analitico
        .memory_usage(deep=True)
        .sum()
        / 1024**2
    )

    return {
        # Claves anteriores conservadas por compatibilidad.
        "filas_originales": int(df_original.shape[0]),
        "columnas_originales": int(
            df_original.shape[1]
        ),
        "filas_finales": int(
            df_analitico.shape[0]
        ),
        "columnas_finales": int(
            df_analitico.shape[1]
        ),
        "variables_derivadas": VARIABLES_DERIVADAS,

        # Reporte ampliado.
        "dataset_base_limpio": {
            "filas": int(df_base_limpio.shape[0]),
            "columnas": int(
                df_base_limpio.shape[1]
            ),
            "nulos": (
                df_base_limpio
                .isna()
                .sum()
                .astype(int)
                .to_dict()
            ),
        },
        "dataset_analitico": {
            "filas": int(df_analitico.shape[0]),
            "columnas": int(
                df_analitico.shape[1]
            ),
            "nulos": (
                df_analitico
                .isna()
                .sum()
                .astype(int)
                .to_dict()
            ),
        },
        "limpieza": metadata_limpieza,
        "memoria_mb": {
            "original_16_columnas": round(
                float(memoria_original),
                4,
            ),
            "base_limpia_16_columnas": round(
                float(memoria_base_limpia),
                4,
            ),
            "dataset_analitico": round(
                float(memoria_analitica),
                4,
            ),
        },
    }


def _guardar_csv_atomico(
    df: pd.DataFrame,
    ruta: Path,
) -> None:
    """Guarda un CSV sin reemplazar el anterior hasta finalizar."""
    ruta.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporal = ruta.with_suffix(
        f"{ruta.suffix}.tmp"
    )

    try:
        df.to_csv(
            temporal,
            index=False,
        )
        temporal.replace(ruta)

    except OSError as exc:
        if temporal.exists():
            temporal.unlink()

        raise DataQualityError(
            f"No fue posible guardar {ruta}: {exc}"
        ) from exc


def _guardar_json_atomico(
    contenido: dict[str, Any],
    ruta: Path,
) -> None:
    """Guarda un JSON mediante archivo temporal."""
    ruta.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporal = ruta.with_suffix(
        f"{ruta.suffix}.tmp"
    )

    try:
        with temporal.open(
            "w",
            encoding="utf-8",
        ) as archivo:
            json.dump(
                contenido,
                archivo,
                indent=4,
                ensure_ascii=False,
            )

        temporal.replace(ruta)

    except OSError as exc:
        if temporal.exists():
            temporal.unlink()

        raise DataQualityError(
            f"No fue posible guardar {ruta}: {exc}"
        ) from exc


def guardar_resultados(
    resultado: PreparationResult,
    config: Settings = settings,
) -> None:
    """Persiste todos los productos de una preparación válida."""
    _guardar_csv_atomico(
        resultado.dataset_base_limpio,
        config.clean_base_dataset,
    )

    _guardar_csv_atomico(
        resultado.dataset_analitico,
        config.analytical_dataset,
    )

    # Archivo temporal de compatibilidad con módulos y tests anteriores.
    _guardar_csv_atomico(
        resultado.dataset_analitico,
        config.legacy_model_dataset,
    )

    _guardar_csv_atomico(
        resultado.kpis,
        config.business_kpis,
    )

    _guardar_json_atomico(
        resultado.reporte_calidad,
        config.quality_report,
    )


def preparar_desde_dataframe(
    df_integrado: pd.DataFrame,
    config: Settings = settings,
    persistir: bool = True,
) -> PreparationResult:
    """
    Recibe directamente la salida de integración y genera los productos.

    Esta es la función que utilizará posteriormente el orquestador.
    """
    if not isinstance(
        df_integrado,
        pd.DataFrame,
    ):
        raise DataQualityError(
            "La preparación requiere un DataFrame integrado."
        )

    df_original = df_integrado.copy()

    df_limpio, metadata_limpieza = (
        limpiar_dataset_base(df_original)
    )

    # Garantía posterior: ninguna etapa de modelado puede recibir
    # el dataset si la limpieza no cumplió el contrato.
    reporte_validacion = validar_dataset_limpio(
        df_limpio
    )

    reporte_validacion.raise_if_invalid()

    df_limpio = optimizar_tipos(
        df_limpio
    )

    df_analitico = crear_variables_derivadas(
        df_limpio
    )

    df_analitico = optimizar_tipos(
        df_analitico
    )

    kpis = generar_kpis_negocio(
        df_analitico
    )

    reporte = generar_reporte_calidad(
        df_original=df_original,
        df_base_limpio=df_limpio,
        df_analitico=df_analitico,
        metadata_limpieza=metadata_limpieza,
    )

    resultado = PreparationResult(
        dataset_base_limpio=df_limpio,
        dataset_analitico=df_analitico,
        kpis=kpis,
        reporte_calidad=reporte,
        metadata={
            "validacion_post_limpieza": {
                "is_valid": (
                    reporte_validacion.is_valid
                ),
                "errors": len(
                    reporte_validacion.errors
                ),
                "warnings": len(
                    reporte_validacion.warnings
                ),
            }
        },
    )

    if persistir:
        guardar_resultados(
            resultado,
            config=config,
        )

    return resultado


def preparar_dataset(
    df_integrado: pd.DataFrame | None = None,
    config: Settings = settings,
    persistir: bool = True,
) -> pd.DataFrame:
    """
    Wrapper compatible con el uso anterior.

    Cuando no recibe un DataFrame, lee data_consolidada.csv.
    Retorna el dataset analítico para no romper tests existentes.
    """
    if df_integrado is None:
        df_integrado = cargar_dataset(
            config=config
        )

    resultado = preparar_desde_dataframe(
        df_integrado=df_integrado,
        config=config,
        persistir=persistir,
    )

    return resultado.dataset_analitico


def main() -> None:
    """Ejecuta la preparación usando el consolidado existente."""
    resultado = preparar_desde_dataframe(
        df_integrado=cargar_dataset(),
        persistir=True,
    )

    print(
        "Preparación completada | "
        f"Base limpia: "
        f"{resultado.dataset_base_limpio.shape} | "
        f"Analítico: "
        f"{resultado.dataset_analitico.shape} | "
        f"KPIs: {resultado.kpis.shape}"
    )


if __name__ == "__main__":
    main()