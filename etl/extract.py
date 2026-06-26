"""
Extracción e integración de las dos fuentes de datos del proyecto de manera robusta.
"""
import pandas as pd
from sqlalchemy import create_engine
import logging
import os
import sys

#Sistema de registro de logs para monitorear el flujo de ETL y capturar errores criticos 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def extraer_csv(ruta="data/usuarios_streaming.csv"):
    try:
        if not os.path.exists(ruta):
            raise FileNotFoundError(f"El archivo esencial {ruta} no existe.")
            
        df = pd.read_csv(ruta)
        logging.info(f"Fuente 1 (CSV) extraída exitosamente: {df.shape[0]} filas, {df.shape[1]} columnas")
        return df
    except Exception as e:
        logging.error(f"Error crítico al leer el archivo CSV: {str(e)}")
        raise e

def extraer_postgres():
    try:
        #Intentamos leer las credenciales desde el entorno de Docker.si no se encuentra la variable, usamos por defecto la ruta local .
        DATABASE_URL = os.getenv(
            "DATABASE_URL", 
            "postgresql://admin:admin@postgres:5432/streaming_db"
        )
        ##Creacion de el motor de conexión e iniciación de la consulta SQL tradicional
        engine = create_engine(DATABASE_URL)
        df = pd.read_sql("SELECT * FROM perfil_usuarios", engine)
        
        if df.empty:
            raise ValueError("La tabla perfil_usuarios se encuentra vacía.")
            
        logging.info(f"Fuente 2 (Postgres) extraída exitosamente: {df.shape[0]} filas, {df.shape[1]} columnas")
        return df
    except Exception as e:
        logging.error(f"Error en la conexión o extracción desde PostgreSQL: {str(e)}")
        raise e

def integrar(df_streaming, df_perfil):
    try:
        #Verificación de la existencia de la columna clave para la integración
        if "id_cliente" not in df_streaming.columns or "id_cliente" not in df_perfil.columns:
            raise KeyError("La columna de integración 'id_cliente' no se encuentra en ambas fuentes.")
        #Realizamos la integración de los datasets mediante un merge interno basado en la columna 'id_cliente'
        data = df_streaming.merge(df_perfil, on="id_cliente", how="inner")
        logging.info(f"Integración completada: {data.shape[0]} filas consolidadas.")
        
        if data.isnull().sum().sum() > 0:
            logging.warning("Datos faltantes detectados post-merge. Aplicando imputación automática por mediana.")
            data = data.fillna(data.median(numeric_only=True))
            
        return data
    except Exception as e:
        logging.error(f"Error durante el proceso de integración/merge: {str(e)}")
        raise e

if __name__ == "__main__":
    try:
        logging.info("--- Iniciando Pipeline ETL Automatizado ---")
        os.makedirs("data", exist_ok=True)
        
        streaming = extraer_csv()
        perfil = extraer_postgres()
        data_consolidada = integrar(streaming, perfil)


        ruta_salida = "data/data_consolidada.csv"
        data_consolidada.to_csv(ruta_salida, index=False)
        logging.info(f"🎉 Dataset consolidado guardado exitosamente en: {ruta_salida}")
        
    except Exception as main_error:
        logging.critical(f"El Pipeline ETL ha fallado de forma irreversible: {str(main_error)}")
        sys.exit(1)