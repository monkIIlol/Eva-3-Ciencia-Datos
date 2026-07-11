"""
Entrenamiento de los modelos supervisados del proyecto.
Acá se entrenan dos modelos con objetivos distintos:
- Regresión: estima el gasto_mensual de un usuario según su comportamiento
  y perfil.
- Clasificación: predice si un usuario tiene riesgo de bajo compromiso.
Para cada tarea se prueban dos algoritmos distintos con pipeline
(escalamiento + modelo) y se ajustan hiperparámetros con GridSearchCV.
"""
import os
import json
import pickle
import logging

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

os.makedirs("models", exist_ok=True)

RANDOM_STATE = 29

#1. Cargar el dataset consolidado
data = pd.read_csv("data/data_consolidada.csv")

FEATURES = [
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
# `riesgo_bajo_compromiso` se construye a partir de `porcentaje_finalizacion`
# y `sesiones_semana`. Si esas dos variables se
# dejan como features, el modelo prácticamente memoriza.
FEATURES_CLASIFICACION = [
    f for f in FEATURES
    if f not in ("porcentaje_finalizacion", "sesiones_semana")
]

resultados = {"regresion": {}, "clasificacion": {}}


#2.TAREA DE REGRESIÓN: predecir gasto_mensual
def entrenar_regresion(data):
    logger.info("=== Entrenando modelos de regresión (target: gasto_mensual) ===")

    X = data[FEATURES]
    y = data["gasto_mensual"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    candidatos = {
        "regresion_lineal": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("modelo", LinearRegression()),
            ]),
            "grid": {},
        },
        "random_forest_regressor": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("modelo", RandomForestRegressor(random_state=RANDOM_STATE)),
            ]),
            "grid": {
                "modelo__n_estimators": [100, 200],
                "modelo__max_depth": [None, 5, 10],
            },
        },
    }

    mejor_modelo = None
    mejor_r2 = -np.inf
    mejor_nombre = None

    for nombre, config in candidatos.items():
        if config["grid"]:
            buscador = GridSearchCV(
                config["pipeline"], config["grid"], cv=5, scoring="r2", n_jobs=-1
            )
            buscador.fit(X_train, y_train)
            modelo = buscador.best_estimator_
            logger.info("%s: mejores hiperparámetros %s", nombre, buscador.best_params_)
        else:
            modelo = config["pipeline"]
            modelo.fit(X_train, y_train)

        pred = modelo.predict(X_test)
        metricas = {
            "r2": float(r2_score(y_test, pred)),
            "mae": float(mean_absolute_error(y_test, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        }
        resultados["regresion"][nombre] = metricas
        logger.info("%s: R2=%.3f | MAE=%.2f | RMSE=%.2f",
                    nombre, metricas["r2"], metricas["mae"], metricas["rmse"])

        if metricas["r2"] > mejor_r2:
            mejor_r2 = metricas["r2"]
            mejor_modelo = modelo
            mejor_nombre = nombre

    logger.info("Mejor modelo de regresión: %s (R2=%.3f)", mejor_nombre, mejor_r2)
    resultados["regresion"]["mejor_modelo"] = mejor_nombre

    pickle.dump(mejor_modelo, open("models/modelo_regresion_gasto.pkl", "wb"))
    return mejor_modelo


#3. TAREA DE CLASIFICACIÓN: predecir riesgo_bajo_compromiso
def entrenar_clasificacion(data):
    logger.info("=== Entrenando modelos de clasificación (target: riesgo_bajo_compromiso) ===")

    umbral_finalizacion = data["porcentaje_finalizacion"].median()
    umbral_sesiones = data["sesiones_semana"].median()

    data = data.copy()
    data["riesgo_bajo_compromiso"] = (
        (data["porcentaje_finalizacion"] < umbral_finalizacion)
        & (data["sesiones_semana"] < umbral_sesiones)
    ).astype(int)

    logger.info(
        "Umbrales usados -> finalización < %.1f y sesiones/semana < %.1f. "
        "Positivos (riesgo=1): %d de %d usuarios",
        umbral_finalizacion, umbral_sesiones,
        data["riesgo_bajo_compromiso"].sum(), len(data),
    )

    X = data[FEATURES_CLASIFICACION]
    y = data["riesgo_bajo_compromiso"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    candidatos = {
        "regresion_logistica": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("modelo", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
            ]),
            "grid": {
                "modelo__C": [0.1, 1.0, 10.0],
            },
        },
        "random_forest_classifier": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("modelo", RandomForestClassifier(random_state=RANDOM_STATE)),
            ]),
            "grid": {
                "modelo__n_estimators": [100, 200],
                "modelo__max_depth": [None, 5, 10],
            },
        },
    }

    mejor_modelo = None
    mejor_f1 = -np.inf
    mejor_nombre = None

    for nombre, config in candidatos.items():
        buscador = GridSearchCV(
            config["pipeline"], config["grid"], cv=5, scoring="f1", n_jobs=-1
        )
        buscador.fit(X_train, y_train)
        modelo = buscador.best_estimator_
        logger.info("%s: mejores hiperparámetros %s", nombre, buscador.best_params_)

        pred = modelo.predict(X_test)
        proba = modelo.predict_proba(X_test)[:, 1]
        metricas = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, proba)),
        }
        resultados["clasificacion"][nombre] = metricas
        logger.info(
            "%s: acc=%.3f | precision=%.3f | recall=%.3f | f1=%.3f | roc_auc=%.3f",
            nombre, metricas["accuracy"], metricas["precision"],
            metricas["recall"], metricas["f1"], metricas["roc_auc"],
        )

        # Priorizamos F1 porque hay desbalance moderado y nos interesa
        # equilibrar falsos positivos y falsos negativos en la señal de riesgo.
        if metricas["f1"] > mejor_f1:
            mejor_f1 = metricas["f1"]
            mejor_modelo = modelo
            mejor_nombre = nombre

    logger.info("Mejor modelo de clasificación: %s (F1=%.3f)", mejor_nombre, mejor_f1)
    resultados["clasificacion"]["mejor_modelo"] = mejor_nombre
    resultados["clasificacion"]["umbral_finalizacion"] = float(umbral_finalizacion)
    resultados["clasificacion"]["umbral_sesiones"] = float(umbral_sesiones)

    pickle.dump(mejor_modelo, open("models/modelo_clasificacion_riesgo.pkl", "wb"))
    return mejor_modelo


if __name__ == "__main__":
    entrenar_regresion(data)
    entrenar_clasificacion(data)

    with open("models/metricas_supervisado.json", "w") as f:
        json.dump(resultados, f, indent=4, ensure_ascii=False)

    logger.info("Modelos supervisados y métricas guardados en /models")
