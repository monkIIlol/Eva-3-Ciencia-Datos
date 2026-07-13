"""
Integración controlada de las fuentes del proyecto.

Esta etapa recibe exclusivamente las fuentes extraídas y su reporte
de validación estructural. No realiza limpieza ni entrenamiento.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from pandas.errors import MergeError

from etl.contracts import (
    BASE_COLUMN_ORDER,
    BASE_REQUIRED_COLUMNS,
    DataIntegrationError,
    ValidationReport,
)


logger = logging.getLogger(__name__)


@dataclass
class IntegrationResult:
    """Resultado completo de una integración válida."""

    dataframe: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)


def _normalizar_nombres_columnas(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convierte todos los nombres de columnas a str nativo de Python.

    SQLAlchemy puede devolver nombres de tipo quoted_name, que se comportan
    como texto pero Scikit-learn rechaza cuando se mezclan con str.
    """
    normalizado = df.copy()
    normalizado.columns = [
        str(columna)
        for columna in normalizado.columns
    ]
    return normalizado


def integrar_fuentes(
    df_streaming: pd.DataFrame,
    df_perfil: pd.DataFrame,
    validation_report: ValidationReport,
) -> IntegrationResult:
    """
    Integra las fuentes mediante una relación estricta uno a uno.

    Parameters
    ----------
    df_streaming:
        Fuente de consumo previamente extraída.
    df_perfil:
        Fuente de perfiles previamente extraída.
    validation_report:
        Reporte producido por validar_fuentes().

    Returns
    -------
    IntegrationResult
        Dataset integrado y metadatos de la operación.

    Raises
    ------
    DataIntegrationError
        Si las fuentes no fueron validadas o si el join incumple
        el contrato esperado.
    """
    if not isinstance(validation_report, ValidationReport):
        raise DataIntegrationError(
            "La integración requiere un ValidationReport válido."
        )

    if not validation_report.is_valid:
        codigos = [
            issue.code
            for issue in validation_report.errors
        ]

        raise DataIntegrationError(
            "Las fuentes no pueden integrarse porque la validación "
            f"estructural contiene errores: {codigos}"
        )

    if not isinstance(df_streaming, pd.DataFrame):
        raise DataIntegrationError(
            "La fuente de streaming no es un DataFrame."
        )

    if not isinstance(df_perfil, pd.DataFrame):
        raise DataIntegrationError(
            "La fuente de perfiles no es un DataFrame."
        )

    # Canonicalización del esquema procedente de fuentes heterogéneas.
    # Evita mezclar str con sqlalchemy.sql.elements.quoted_name.
    df_streaming = _normalizar_nombres_columnas(
        df_streaming
    )
    df_perfil = _normalizar_nombres_columnas(
        df_perfil
    )

    columnas_repetidas = (
        set(df_streaming.columns)
        .intersection(df_perfil.columns)
        - {"id_cliente"}
    )

    if columnas_repetidas:
        raise DataIntegrationError(
            "Las fuentes contienen columnas homónimas no autorizadas: "
            f"{sorted(columnas_repetidas)}"
        )

    filas_streaming = int(df_streaming.shape[0])
    filas_perfil = int(df_perfil.shape[0])

    ids_streaming = set(df_streaming["id_cliente"])
    ids_perfil = set(df_perfil["id_cliente"])

    ids_compartidos = ids_streaming.intersection(ids_perfil)
    ids_solo_streaming = ids_streaming - ids_perfil
    ids_solo_perfil = ids_perfil - ids_streaming

    if ids_solo_streaming or ids_solo_perfil:
        raise DataIntegrationError(
            "Las fuentes no representan exactamente el mismo "
            "universo de usuarios. "
            f"Solo streaming: {len(ids_solo_streaming)}. "
            f"Solo perfiles: {len(ids_solo_perfil)}."
        )

    try:
        integrado = df_streaming.merge(
            df_perfil,
            on="id_cliente",
            how="inner",
            validate="one_to_one",
            sort=False,
        )

    except MergeError as exc:
        raise DataIntegrationError(
            "La integración no cumple una relación one-to-one "
            "por id_cliente."
        ) from exc

    if integrado.empty:
        raise DataIntegrationError(
            "La integración produjo un dataset vacío."
        )

    filas_integradas = int(integrado.shape[0])

    if filas_integradas != len(ids_compartidos):
        raise DataIntegrationError(
            "La cantidad de filas integradas no coincide con la "
            "cantidad de identificadores compartidos. "
            f"Filas: {filas_integradas}. "
            f"IDs compartidos: {len(ids_compartidos)}."
        )

    columnas_faltantes = (
        BASE_REQUIRED_COLUMNS - set(integrado.columns)
    )

    if columnas_faltantes:
        raise DataIntegrationError(
            "El dataset integrado no contiene todas las columnas "
            f"del contrato base: {sorted(columnas_faltantes)}"
        )

    columnas_adicionales = sorted(
        set(integrado.columns) - BASE_REQUIRED_COLUMNS
    )

    # Orden determinista de las 16 variables oficiales.
    integrado = integrado[BASE_COLUMN_ORDER].copy()

    # Garantía final: todos los nombres son str nativos, no subclases
    # entregadas por conectores o librerías externas.
    integrado.columns = [
        str(columna)
        for columna in integrado.columns
    ]

    metadata = {
        "filas_streaming": filas_streaming,
        "filas_perfil": filas_perfil,
        "ids_compartidos": len(ids_compartidos),
        "ids_solo_streaming": len(ids_solo_streaming),
        "ids_solo_perfil": len(ids_solo_perfil),
        "filas_integradas": filas_integradas,
        "columnas_integradas": int(integrado.shape[1]),
        "columnas_adicionales_descartadas": columnas_adicionales,
        "tipo_join": "inner",
        "cardinalidad": "one_to_one",
        "tipo_nombres_columnas": "str",
    }

    logger.info(
        "Integración completada: %s filas y %s columnas.",
        integrado.shape[0],
        integrado.shape[1],
    )

    return IntegrationResult(
        dataframe=integrado,
        metadata=metadata,
    )


def main() -> None:
    """Ejecuta una prueba manual de la integración contractual."""
    from etl.extract import extraer_fuentes
    from etl.validate import validar_fuentes

    df_streaming, df_perfil = extraer_fuentes()

    validation_report = validar_fuentes(
        df_streaming,
        df_perfil,
    )

    validation_report.raise_if_invalid()

    resultado = integrar_fuentes(
        df_streaming=df_streaming,
        df_perfil=df_perfil,
        validation_report=validation_report,
    )

    print(
        "Integración correcta | "
        f"Dataset: {resultado.dataframe.shape} | "
        f"Cardinalidad: {resultado.metadata['cardinalidad']}"
    )


if __name__ == "__main__":
    main()
