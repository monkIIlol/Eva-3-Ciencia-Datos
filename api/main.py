"""API REST contractual para segmentación y modelos supervisados.

La API no entrena modelos. Carga artefactos previamente publicados y
valida las entradas con esquemas generados desde config/features.py.
"""

import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pydantic
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, create_model

from config.features import (
    CLASSIFICATION_FEATURES,
    CLASSIFICATION_TARGET_SOURCE_COLUMNS,
    KMEANS_FEATURES,
    REGRESSION_FEATURES,
)
from config.settings import Settings, settings
from etl.contracts import COLUMN_RANGES, INTEGER_COLUMNS


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Esquemas de entrada
# ---------------------------------------------------------------------------

if int(pydantic.__version__.split(".", 1)[0]) >= 2:
    from pydantic import ConfigDict

    class StrictInputModel(BaseModel):
        """Modelo base que rechaza campos no contemplados por el contrato."""

        model_config = ConfigDict(extra="forbid")

else:

    class StrictInputModel(BaseModel):
        """Compatibilidad con Pydantic 1.x."""

        class Config:
            extra = "forbid"


def _campo_numerico(nombre: str, requerido: bool = True) -> tuple[Any, Any]:
    """Construye un campo tipado usando los rangos del contrato de datos."""
    tipo_base: Any = int if nombre in INTEGER_COLUMNS else float
    minimo, maximo = COLUMN_RANGES.get(nombre, (None, None))

    restricciones: dict[str, float] = {}
    if minimo is not None:
        restricciones["ge"] = minimo
    if maximo is not None:
        restricciones["le"] = maximo

    if requerido:
        return tipo_base, Field(..., **restricciones)

    return tipo_base | None, Field(None, **restricciones)


def _crear_modelo_entrada(
    nombre: str,
    campos_requeridos: list[str],
    campos_opcionales: list[str] | None = None,
) -> type[BaseModel]:
    """Crea un esquema Pydantic directamente desde las listas oficiales."""
    definiciones: dict[str, tuple[Any, Any]] = {
        campo: _campo_numerico(campo, requerido=True)
        for campo in campos_requeridos
    }

    for campo in campos_opcionales or []:
        if campo not in definiciones:
            definiciones[campo] = _campo_numerico(campo, requerido=False)

    return create_model(
        nombre,
        __base__=StrictInputModel,
        **definiciones,
    )


KMeansInput = _crear_modelo_entrada(
    "KMeansInput",
    KMEANS_FEATURES,
)

RegressionInput = _crear_modelo_entrada(
    "RegressionInput",
    REGRESSION_FEATURES,
)

# Las dos columnas usadas para construir la etiqueta histórica se aceptan
# opcionalmente por compatibilidad con el dashboard, pero nunca se entregan
# al clasificador.
ClassificationInput = _crear_modelo_entrada(
    "ClassificationInput",
    CLASSIFICATION_FEATURES,
    CLASSIFICATION_TARGET_SOURCE_COLUMNS,
)


class ClusterResponse(BaseModel):
    cluster: int


class RegressionResponse(BaseModel):
    gasto_mensual_predicho: float


class ClassificationResponse(BaseModel):
    riesgo_bajo_compromiso: int
    probabilidad: float


# ---------------------------------------------------------------------------
# Carga controlada de artefactos
# ---------------------------------------------------------------------------

@dataclass
class ArtifactRegistry:
    modelo_kmeans: Any | None = None
    scaler_kmeans: Any | None = None
    metricas_kmeans: dict[str, Any] | None = None
    modelo_regresion: Any | None = None
    modelo_clasificacion: Any | None = None
    metricas_supervisado: dict[str, Any] | None = None
    errores: dict[str, str] = field(default_factory=dict)

    @property
    def kmeans_disponible(self) -> bool:
        return self.modelo_kmeans is not None and self.scaler_kmeans is not None

    @property
    def dashboard_disponible(self) -> bool:
        return self.kmeans_disponible and self.metricas_kmeans is not None

    @property
    def regresion_disponible(self) -> bool:
        return self.modelo_regresion is not None

    @property
    def clasificacion_disponible(self) -> bool:
        return self.modelo_clasificacion is not None

    @property
    def metricas_supervisadas_disponibles(self) -> bool:
        return self.metricas_supervisado is not None

    def estado(self) -> dict[str, bool]:
        return {
            "kmeans": self.kmeans_disponible,
            "dashboard": self.dashboard_disponible,
            "regresion": self.regresion_disponible,
            "clasificacion": self.clasificacion_disponible,
            "metricas_supervisadas": self.metricas_supervisadas_disponibles,
        }


def _cargar_pickle(ruta: Path) -> Any:
    with ruta.open("rb") as archivo:
        return pickle.load(archivo)


def _cargar_json(ruta: Path) -> dict[str, Any]:
    with ruta.open("r", encoding="utf-8") as archivo:
        contenido = json.load(archivo)

    if not isinstance(contenido, dict):
        raise ValueError(f"El artefacto {ruta.name} no contiene un objeto JSON.")

    return contenido


def _intentar_carga(
    registro: ArtifactRegistry,
    nombre: str,
    ruta: Path,
    cargador: Any,
) -> Any | None:
    try:
        return cargador(ruta)
    except Exception as exc:  # la API debe iniciar en modo degradado
        registro.errores[nombre] = f"{type(exc).__name__}: {exc}"
        logger.warning("Artefacto %s no disponible en %s: %s", nombre, ruta, exc)
        return None


def cargar_artefactos(config: Settings = settings) -> ArtifactRegistry:
    """Carga cada artefacto de manera independiente y registra fallos."""
    registro = ArtifactRegistry()
    modelos = config.models_dir

    registro.modelo_kmeans = _intentar_carga(
        registro,
        "modelo_kmeans",
        modelos / "modelo_kmeans.pkl",
        _cargar_pickle,
    )
    registro.scaler_kmeans = _intentar_carga(
        registro,
        "scaler_kmeans",
        modelos / "scaler.pkl",
        _cargar_pickle,
    )
    registro.metricas_kmeans = _intentar_carga(
        registro,
        "metricas_kmeans",
        modelos / "metricas.json",
        _cargar_json,
    )
    registro.modelo_regresion = _intentar_carga(
        registro,
        "modelo_regresion",
        modelos / "modelo_regresion_gasto.pkl",
        _cargar_pickle,
    )
    registro.modelo_clasificacion = _intentar_carga(
        registro,
        "modelo_clasificacion",
        modelos / "modelo_clasificacion_riesgo.pkl",
        _cargar_pickle,
    )
    registro.metricas_supervisado = _intentar_carga(
        registro,
        "metricas_supervisado",
        modelos / "metricas_supervisado.json",
        _cargar_json,
    )

    estado = registro.estado()
    logger.info(
        "Artefactos cargados | KMeans=%s | Regresión=%s | Clasificación=%s | "
        "Métricas supervisadas=%s",
        estado["kmeans"],
        estado["regresion"],
        estado["clasificacion"],
        estado["metricas_supervisadas"],
    )
    return registro


artefactos = cargar_artefactos()


def recargar_artefactos(config: Settings = settings) -> dict[str, bool]:
    """Recarga manualmente los artefactos y devuelve su nuevo estado."""
    global artefactos
    artefactos = cargar_artefactos(config=config)
    return artefactos.estado()


# ---------------------------------------------------------------------------
# Aplicación y utilidades de inferencia
# ---------------------------------------------------------------------------

app = FastAPI(
    title="API de Analítica de Usuarios — Streaming",
    version="2.0.0",
    description=(
        "Segmentación KMeans, predicción de gasto mensual y clasificación "
        "del proxy de bajo compromiso."
    ),
)


def _model_dump(datos: BaseModel) -> dict[str, Any]:
    if hasattr(datos, "model_dump"):
        return datos.model_dump(exclude_none=True)
    return datos.dict(exclude_none=True)


def _fila_modelo(datos: BaseModel, features: list[str]) -> pd.DataFrame:
    payload = _model_dump(datos)
    return pd.DataFrame(
        [{campo: payload[campo] for campo in features}],
        columns=features,
    )


def _verificar_salida_finita(valor: Any, nombre: str) -> float:
    numero = float(valor)
    if not np.isfinite(numero):
        logger.error("El modelo produjo una salida no finita para %s.", nombre)
        raise HTTPException(
            status_code=500,
            detail="El modelo produjo una salida numérica inválida.",
        )
    return numero


@app.get("/")
def inicio() -> dict[str, Any]:
    """Confirma que la API está activa e informa artefactos disponibles."""
    return {
        "mensaje": "API de analítica de usuarios funcionando",
        "artefactos": artefactos.estado(),
    }


@app.get("/health")
def health() -> dict[str, Any]:
    """Estado técnico sin exponer trazas ni rutas internas."""
    estado = artefactos.estado()
    return {
        "status": "ok" if all(estado.values()) else "degraded",
        "artefactos": estado,
        "errores_detectados": sorted(artefactos.errores),
    }


@app.get("/dashboard-data")
def dashboard_data() -> dict[str, Any]:
    """Devuelve los resultados persistidos que utiliza el dashboard."""
    if not artefactos.dashboard_disponible:
        raise HTTPException(
            status_code=503,
            detail="Los artefactos de segmentación no están disponibles.",
        )

    rutas = {
        "usuarios": settings.data_dir / "usuarios_segmentados.csv",
        "centroides": settings.data_dir / "centroides.csv",
        "evaluacion_k": settings.data_dir / "evaluacion_k.csv",
    }

    try:
        usuarios = pd.read_csv(rutas["usuarios"])
        centroides = pd.read_csv(rutas["centroides"])
        evaluacion_k = pd.read_csv(rutas["evaluacion_k"])
    except (OSError, pd.errors.ParserError) as exc:
        logger.exception("No fue posible leer los productos del dashboard.")
        raise HTTPException(
            status_code=503,
            detail="Los resultados de segmentación no pueden leerse.",
        ) from exc

    return {
        "usuarios": usuarios.to_dict(orient="records"),
        "centroides": centroides.to_dict(orient="records"),
        "evaluacion_k": evaluacion_k.to_dict(orient="records"),
        "metricas": artefactos.metricas_kmeans,
    }


@app.post("/predict", response_model=ClusterResponse)
def predict(datos: KMeansInput) -> ClusterResponse:
    """Asigna un usuario nuevo a uno de los clusters entrenados."""
    if not artefactos.kmeans_disponible:
        raise HTTPException(
            status_code=503,
            detail="El modelo KMeans no está disponible.",
        )

    fila = _fila_modelo(datos, KMEANS_FEATURES)

    try:
        escalada = artefactos.scaler_kmeans.transform(fila)

        # KMeans exige que la entrada tenga el mismo dtype que sus centroides.
        # Esto mantiene compatibilidad tanto con artefactos float64 nuevos como
        # con artefactos float32 generados por ejecuciones anteriores.
        dtype_modelo = artefactos.modelo_kmeans.cluster_centers_.dtype
        escalada = np.asarray(escalada, dtype=dtype_modelo)

        cluster = artefactos.modelo_kmeans.predict(escalada)[0]
    except (ValueError, TypeError, AttributeError) as exc:
        logger.exception("Fallo de inferencia en KMeans.")
        raise HTTPException(
            status_code=500,
            detail="Los artefactos KMeans no son compatibles con la entrada.",
        ) from exc

    logger.info("Predicción KMeans generada: cluster=%s", int(cluster))
    return ClusterResponse(cluster=int(cluster))


@app.get("/metricas-supervisado")
def obtener_metricas_supervisado() -> dict[str, Any]:
    """Devuelve métricas y evidencia de selección de modelos supervisados."""
    if not artefactos.metricas_supervisadas_disponibles:
        raise HTTPException(
            status_code=503,
            detail="Las métricas supervisadas no están disponibles.",
        )
    return artefactos.metricas_supervisado


@app.post("/predict-gasto", response_model=RegressionResponse)
def predict_gasto(datos: RegressionInput) -> RegressionResponse:
    """Predice el gasto mensual usando el modelo de regresión publicado."""
    if not artefactos.regresion_disponible:
        raise HTTPException(
            status_code=503,
            detail="El modelo de regresión no está disponible.",
        )

    fila = _fila_modelo(datos, REGRESSION_FEATURES)

    try:
        prediccion = artefactos.modelo_regresion.predict(fila)[0]
    except (ValueError, TypeError, AttributeError) as exc:
        logger.exception("Fallo de inferencia en regresión.")
        raise HTTPException(
            status_code=500,
            detail="El artefacto de regresión no es compatible con la entrada.",
        ) from exc

    gasto = _verificar_salida_finita(prediccion, "gasto_mensual")
    logger.info("Predicción de gasto generada: %.2f", gasto)
    return RegressionResponse(
        gasto_mensual_predicho=round(max(0.0, gasto), 2),
    )


@app.post("/predict-riesgo", response_model=ClassificationResponse)
def predict_riesgo(datos: ClassificationInput) -> ClassificationResponse:
    """Predice el proxy de riesgo sin usar las columnas que crean su etiqueta."""
    if not artefactos.clasificacion_disponible:
        raise HTTPException(
            status_code=503,
            detail="El modelo de clasificación no está disponible.",
        )

    fila = _fila_modelo(datos, CLASSIFICATION_FEATURES)

    try:
        riesgo = int(artefactos.modelo_clasificacion.predict(fila)[0])
        probabilidades = artefactos.modelo_clasificacion.predict_proba(fila)[0]
        clases = list(artefactos.modelo_clasificacion.classes_)
        indice_positivo = clases.index(1)
        probabilidad = probabilidades[indice_positivo]
    except (ValueError, TypeError, AttributeError) as exc:
        logger.exception("Fallo de inferencia en clasificación.")
        raise HTTPException(
            status_code=500,
            detail="El artefacto de clasificación no es compatible con la entrada.",
        ) from exc

    probabilidad_finita = _verificar_salida_finita(
        probabilidad,
        "probabilidad_riesgo",
    )
    probabilidad_finita = min(max(probabilidad_finita, 0.0), 1.0)

    logger.info(
        "Predicción de riesgo generada: riesgo=%s prob=%.3f",
        riesgo,
        probabilidad_finita,
    )
    return ClassificationResponse(
        riesgo_bajo_compromiso=riesgo,
        probabilidad=round(probabilidad_finita, 3),
    )


@app.get("/kpis-negocio")
def obtener_kpis_negocio() -> dict[str, Any]:
    """Devuelve los KPIs de negocio generados por el pipeline."""
    ruta = settings.data_dir / "kpis_negocio.csv"

    try:
        kpis = pd.read_csv(ruta)
    except Exception as exc:
        logger.exception("No fue posible leer los KPIs de negocio.")
        raise HTTPException(
            status_code=503,
            detail="Los KPIs de negocio no están disponibles. Ejecuta el pipeline.",
        ) from exc

    return {
        "kpis": kpis.to_dict(orient="records"),
        "total_filas": len(kpis),
    }


@app.get("/reporte-calidad")
def obtener_reporte_calidad() -> dict[str, Any]:
    """Devuelve el reporte de calidad generado por el pipeline."""
    ruta = settings.data_dir / "reporte_calidad.json"

    try:
        return _cargar_json(ruta)
    except Exception as exc:
        logger.exception("No fue posible leer el reporte de calidad.")
        raise HTTPException(
            status_code=503,
            detail="El reporte de calidad no está disponible. Ejecuta el pipeline.",
        ) from exc