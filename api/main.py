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
#si no están entrenados (train_supervisado.py), devuelven 503.
modelo_regresion = None
modelo_clasificacion = None
metricas_supervisado = None
try:
    modelo_regresion = pickle.load(open("models/modelo_regresion_gasto.pkl", "rb"))
    modelo_clasificacion = pickle.load(open("models/modelo_clasificacion_riesgo.pkl", "rb"))
    with open("models/metricas_supervisado.json") as f:
        metricas_supervisado = json.load(f)
    logger.info("Modelos supervisados (regresión y clasificación) cargados correctamente.")
except FileNotFoundError:
    logger.warning(
        "No se encontraron los modelos supervisados en /models. "
        "Ejecuta model/train_supervisado.py para habilitar /predict-gasto y /predict-riesgo."
    )
FEATURES_SUPERVISADO = [
    "horas_consumo_mensual",
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
# Se excluyen porque son las variables usadas para crear la etiqueta
FEATURES_CLASIFICACION = [
    f for f in FEATURES_SUPERVISADO
    if f not in ("porcentaje_finalizacion", "sesiones_semana")
]



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
@app.get("/metricas-supervisado")
def obtener_metricas_supervisado():
    """Devuelve las métricas de comparación de los modelos de
    regresión y clasificación entrenados en model/train_supervisado.py."""
    if metricas_supervisado is None:
        raise HTTPException(
            status_code=503,
            detail="Métricas supervisadas no disponibles. Ejecuta model/train_supervisado.py.",
        )
    return metricas_supervisado


@app.post("/predict-gasto")
def predict_gasto(datos: dict):
    """Regresión: predice el gasto_mensual esperado de un usuario a
    partir de su comportamiento y perfil (modelo ganador: Random Forest)."""
    if modelo_regresion is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo de regresión no disponible. Ejecuta model/train_supervisado.py.",
        )
    try:
        fila = pd.DataFrame([{campo: datos[campo] for campo in FEATURES_SUPERVISADO}])
    except KeyError as error:
        raise HTTPException(
            status_code=422,
            detail=f"Falta el campo requerido: {error}",
        )

    gasto_predicho = modelo_regresion.predict(fila)[0]
    logger.info("Predicción de gasto generada: %.2f", gasto_predicho)
    return {"gasto_mensual_predicho": round(float(gasto_predicho), 2)}


@app.post("/predict-riesgo")
def predict_riesgo(datos: dict):
    """Clasificación: predice si un usuario tiene riesgo de bajo
    compromiso (posible señal temprana de abandono). Modelo ganador:
    Random Forest Classifier, elegido por mejor F1 en validación cruzada."""
    if modelo_clasificacion is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo de clasificación no disponible. Ejecuta model/train_supervisado.py.",
        )
    try:
        fila = pd.DataFrame([{campo: datos[campo] for campo in FEATURES_CLASIFICACION}])
    except KeyError as error:
        raise HTTPException(
            status_code=422,
            detail=f"Falta el campo requerido: {error}",
        )

    riesgo = modelo_clasificacion.predict(fila)[0]
    probabilidad = modelo_clasificacion.predict_proba(fila)[0][1]
    logger.info("Predicción de riesgo generada: riesgo=%s prob=%.3f", int(riesgo), probabilidad)
    return {
        "riesgo_bajo_compromiso": int(riesgo),
        "probabilidad": round(float(probabilidad), 3),
    }