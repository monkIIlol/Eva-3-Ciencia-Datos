"""
API REST que expone los resultados del modelo de segmentación.

No es una fuente de datos: sirve los resultados ya calculados por
model/train.py (clusters, centroides, métricas) para que el dashboard
(u otro cliente) los consuma vía HTTP.
"""
import pandas as pd
import json
import pickle
import os
from fastapi import FastAPI, HTTPException
import logging


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="API de Segmentación de Usuarios — Streaming")

# Cargar al iniciar: los resultados que dejó el entrenamiento
try:
    modelo = pickle.load(open("models/modelo_kmeans.pkl", "rb"))
    scaler = pickle.load(open("models/scaler.pkl", "rb"))
    with open("models/metricas.json") as f:
        metricas = json.load(f)
    logger.info("Modelo, scaler y métricas cargados correctamente al iniciar la API.")
except FileNotFoundError:
    logger.exception(
        "No se encontraron los artefactos del modelo en /models. "
        "¿Se ejecutó model/train.py antes de levantar la API?"
    )
    raise


@app.get("/")
def inicio():
    """Endpoint raíz: confirma que el servicio está corriendo."""
    return {"mensaje": "API de segmentación de usuarios funcionando"}


@app.get("/dashboard-data")
def dashboard_data():
    """Devuelve todo lo que el dashboard necesita para graficar:
    usuarios con su cluster asignado, centroides y métricas del modelo."""
    try:
        usuarios = pd.read_csv("data/usuarios_segmentados.csv")
        centroides = pd.read_csv("data/centroides.csv")
        evaluacion_k = pd.read_csv("data/evaluacion_k.csv")
    except FileNotFoundError as error:
        logger.exception("Faltan archivos de resultados del modelo.")
        raise HTTPException(
            status_code=503,
            detail=f"Resultados del modelo no disponibles todavía: {error.filename}",
        )

    return {
        "usuarios": usuarios.to_dict(orient="records"),
        "centroides": centroides.to_dict(orient="records"),
        "evaluacion_k": evaluacion_k.to_dict(orient="records"),
        "metricas": metricas,
    }


@app.post("/predict")
def predict(datos: dict):
    """Clasifica un usuario nuevo: recibe sus variables y devuelve
    a qué cluster pertenece, según el modelo ya entrenado."""
    try:
        fila = pd.DataFrame([datos])
        X = scaler.transform(fila)
    except (KeyError, ValueError) as error:
        logger.warning("Datos de entrada inválidos en /predict: %s", error)
        raise HTTPException(
            status_code=422,
            detail=f"Datos de entrada inválidos o incompletos: {error}",
        )

    cluster = modelo.predict(X)
    logger.info("Predicción generada: cluster=%s", int(cluster[0]))
    return {"cluster": int(cluster[0])}