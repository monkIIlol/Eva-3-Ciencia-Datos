"""
Extracción e integración de las dos fuentes de datos del proyecto.

Fuente 1: usuarios_streaming.csv  (archivo plano, datos de consumo)
Fuente 2: perfil_usuarios         (tabla en Postgres, datos de perfil)

Ambas se unen por id_cliente para formar el dataset consolidado.
"""
import os
import pandas as pd
from sqlalchemy import create_engine
import logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def extraer_csv(ruta="data/usuarios_streaming.csv"):
    """Lee la fuente 1: el CSV de consumo de streaming."""
    try:
        df = pd.read_csv(ruta)
    except FileNotFoundError:
        logger.error("No se encontró el archivo CSV en %s", ruta)
        raise
    except pd.errors.EmptyDataError:
        logger.error("El archivo CSV en %s está vacío", ruta)
        raise

    logger.info("CSV leído: %s filas, %s columnas", df.shape[0], df.shape[1])
    return df


def extraer_postgres():
    """Lee la fuente 2: la tabla perfil_usuarios desde Postgres."""
    usuario = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    host = os.getenv("POSTGRES_HOST")
    puerto = os.getenv("POSTGRES_PORT")
    base_datos = os.getenv("POSTGRES_DB")

    faltantes = [
        nombre
        for nombre, valor in [
            ("POSTGRES_USER", usuario),
            ("POSTGRES_PASSWORD", password),
            ("POSTGRES_HOST", host),
            ("POSTGRES_PORT", puerto),
            ("POSTGRES_DB", base_datos),
        ]
        if not valor
    ]
    if faltantes:
        logger.error("Faltan variables de entorno requeridas: %s", faltantes)
        raise EnvironmentError(f"Variables de entorno faltantes: {faltantes}")

    url = f"postgresql://{usuario}:{password}@{host}:{puerto}/{base_datos}"

    try:
        engine = create_engine(url)
        df = pd.read_sql("SELECT * FROM perfil_usuarios", engine)
    except SQLAlchemyError:
        logger.exception("Error al conectar o consultar Postgres en %s:%s", host, puerto)
        raise

    logger.info("Postgres leído: %s filas, %s columnas", df.shape[0], df.shape[1])
    return df


def integrar(df_streaming, df_perfil):
    """Une las dos fuentes por id_cliente."""
    data = df_streaming.merge(df_perfil, on="id_cliente")
    logger.info("Datos integrados: %s filas, %s columnas", data.shape[0], data.shape[1])
    return data


if __name__ == "__main__":
    # Flujo de extracción: leer ambas fuentes y unirlas
    streaming = extraer_csv()
    perfil = extraer_postgres()
    data = integrar(streaming, perfil)

    # Guardar el dataset consolidado para que el modelo lo use
    data.to_csv("data/data_consolidada.csv", index=False)
    logger.info("Dataset consolidado guardado en data/data_consolidada.csv")
