"""
Extracción de las fuentes de datos del proyecto.

Fuente 1:
    usuarios_streaming.csv

Fuente 2:
    tabla perfil_usuarios en PostgreSQL

Este módulo obtiene las fuentes y las devuelve como DataFrames.
La ejecución directa conserva temporalmente la generación del dataset
consolidado para mantener compatibilidad con el flujo existente.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

from config.settings import Settings, settings
from etl.contracts import (
    SourceExtractionError,
)

from etl.integrate import integrar_fuentes
from etl.validate import validar_fuentes


logger = logging.getLogger(__name__)


def extraer_csv(
    ruta: str | Path | None = None,
    config: Settings = settings,
) -> pd.DataFrame:
    """
    Extrae la fuente de consumo desde un archivo CSV.

    Parameters
    ----------
    ruta:
        Ruta opcional del archivo. Cuando no se especifica,
        se utiliza la definida en la configuración.
    config:
        Configuración central del proyecto.

    Returns
    -------
    pd.DataFrame
        Datos de consumo de streaming.

    Raises
    ------
    SourceExtractionError
        Si el archivo no existe, está vacío o no puede leerse.
    """
    ruta_csv = Path(ruta) if ruta is not None else config.streaming_csv

    try:
        df = pd.read_csv(ruta_csv)

    except FileNotFoundError as exc:
        raise SourceExtractionError(
            f"No se encontró la fuente CSV: {ruta_csv}"
        ) from exc

    except pd.errors.EmptyDataError as exc:
        raise SourceExtractionError(
            f"La fuente CSV está vacía: {ruta_csv}"
        ) from exc

    except (OSError, pd.errors.ParserError) as exc:
        raise SourceExtractionError(
            f"No fue posible leer la fuente CSV {ruta_csv}: {exc}"
        ) from exc

    if df.empty:
        raise SourceExtractionError(
            f"La fuente CSV no contiene registros: {ruta_csv}"
        )

    logger.info(
        "Fuente CSV extraída: %s filas, %s columnas.",
        df.shape[0],
        df.shape[1],
    )

    return df


def extraer_postgres(
    config: Settings = settings,
) -> pd.DataFrame:
    """
    Extrae la tabla de perfiles desde PostgreSQL.

    Parameters
    ----------
    config:
        Configuración central del proyecto.

    Returns
    -------
    pd.DataFrame
        Datos de perfil de usuarios.

    Raises
    ------
    SourceExtractionError
        Si no es posible conectar, consultar la tabla o si está vacía.
    """
    engine = None

    try:
        engine = create_engine(
            config.database_url,
            pool_pre_ping=True,
        )

        df = pd.read_sql_table(
            table_name=config.postgres_table,
            con=engine,
        )

    except (SQLAlchemyError, ValueError) as exc:
        raise SourceExtractionError(
            "No fue posible extraer la tabla "
            f"'{config.postgres_table}' desde PostgreSQL: {exc}"
        ) from exc

    finally:
        if engine is not None:
            engine.dispose()

    if df.empty:
        raise SourceExtractionError(
            f"La tabla PostgreSQL '{config.postgres_table}' está vacía."
        )

    logger.info(
        "Fuente PostgreSQL extraída: %s filas, %s columnas.",
        df.shape[0],
        df.shape[1],
    )

    return df


def extraer_fuentes(
    config: Settings = settings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extrae las dos fuentes sin integrarlas ni modificarlas.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Primero el DataFrame de streaming y luego el DataFrame de perfiles.
    """
    logger.info("Iniciando extracción de fuentes.")

    df_streaming = extraer_csv(config=config)
    df_perfil = extraer_postgres(config=config)

    logger.info("Extracción de fuentes finalizada correctamente.")

    return df_streaming, df_perfil


def guardar_dataset_consolidado(
    df: pd.DataFrame,
    ruta: str | Path | None = None,
    config: Settings = settings,
) -> Path:
    """
    Guarda el dataset consolidado mediante escritura temporal.

    El archivo definitivo solo reemplaza al anterior cuando la escritura
    termina correctamente.
    """
    ruta_salida = (
        Path(ruta)
        if ruta is not None
        else config.consolidated_dataset
    )

    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    ruta_temporal = ruta_salida.with_suffix(
        f"{ruta_salida.suffix}.tmp"
    )

    try:
        df.to_csv(ruta_temporal, index=False)
        ruta_temporal.replace(ruta_salida)

    except OSError as exc:
        if ruta_temporal.exists():
            ruta_temporal.unlink()

        raise SourceExtractionError(
            f"No fue posible guardar el dataset consolidado: {exc}"
        ) from exc

    logger.info(
        "Dataset consolidado guardado en %s.",
        ruta_salida,
    )

    return ruta_salida


def main() -> None:
    """
    Ejecuta el flujo contractual inicial de datos.

    Flujo:
        extracción
        → validación estructural
        → integración controlada
        → guardado del dataset consolidado
    """
    settings.create_directories()

    # 1. Extracción real de las dos fuentes.
    df_streaming, df_perfil = extraer_fuentes(settings)

    # 2. La salida de extracción pasa directamente a validación.
    validation_report = validar_fuentes(
        df_streaming,
        df_perfil,
    )

    # 3. El flujo se detiene si existen errores estructurales.
    validation_report.raise_if_invalid()

    # 4. La integración solo recibe fuentes validadas.
    integration_result = integrar_fuentes(
        df_streaming=df_streaming,
        df_perfil=df_perfil,
        validation_report=validation_report,
    )

    # 5. Solo se guarda la salida contractual de integración.
    ruta_salida = guardar_dataset_consolidado(
        integration_result.dataframe,
        config=settings,
    )

    print(
        "ETL inicial completado | "
        f"Dataset: {integration_result.dataframe.shape} | "
        f"Cardinalidad: "
        f"{integration_result.metadata['cardinalidad']} | "
        f"Salida: {ruta_salida}"
    )


if __name__ == "__main__":
    main()