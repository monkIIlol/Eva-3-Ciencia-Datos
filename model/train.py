"""
Entrenamiento contractual del modelo KMeans.

Entrada oficial:
    dataset_base_limpio.csv o un DataFrame validado.

Artefactos conservados por compatibilidad:
    models/modelo_kmeans.pkl
    models/scaler.pkl
    models/pca.pkl
    models/metricas.json
    data/usuarios_segmentados.csv
    data/evaluacion_k.csv
    data/centroides.csv
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from config.features import ID_COLUMN, KMEANS_FEATURES
from config.settings import Settings, settings
from etl.contracts import ModelTrainingError


logger = logging.getLogger(__name__)


@dataclass
class KMeansTrainingResult:
    """Productos generados por el entrenamiento de KMeans."""

    modelo: KMeans
    scaler: StandardScaler
    pca: PCA
    usuarios_segmentados: pd.DataFrame
    evaluacion_k: pd.DataFrame
    centroides: pd.DataFrame
    metricas: dict[str, Any]


def cargar_dataset_base(
    ruta: str | Path | None = None,
    config: Settings = settings,
) -> pd.DataFrame:
    """
    Carga el dataset base limpio.

    No utiliza data_consolidada.csv como fallback, porque los modelos
    solo deben consumir la salida aprobada de preparación.
    """
    ruta_dataset = (
        Path(ruta)
        if ruta is not None
        else config.clean_base_dataset
    )

    if not ruta_dataset.exists():
        raise ModelTrainingError(
            f"No se encontró el dataset base limpio: {ruta_dataset}. "
            "Ejecuta primero la etapa de preparación."
        )

    try:
        df = pd.read_csv(ruta_dataset)

    except (OSError, pd.errors.ParserError) as exc:
        raise ModelTrainingError(
            f"No fue posible cargar el dataset base limpio: {exc}"
        ) from exc

    if df.empty:
        raise ModelTrainingError(
            "El dataset base limpio está vacío."
        )

    return df


def validar_entrada_kmeans(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Valida y normaliza la vista utilizada por KMeans.

    Returns
    -------
    pd.DataFrame
        Dataset con id_cliente y las 15 variables oficiales.
    """
    if not isinstance(df, pd.DataFrame):
        raise ModelTrainingError(
            "KMeans requiere un DataFrame como entrada."
        )

    if df.empty:
        raise ModelTrainingError(
            "KMeans no puede entrenarse con un dataset vacío."
        )

    columnas_requeridas = {
        ID_COLUMN,
        *KMEANS_FEATURES,
    }

    faltantes = sorted(
        columnas_requeridas - set(df.columns)
    )

    if faltantes:
        raise ModelTrainingError(
            "Faltan columnas requeridas para KMeans: "
            f"{faltantes}"
        )

    data = df[
        [ID_COLUMN] + KMEANS_FEATURES
    ].copy()

    ids = pd.to_numeric(
        data[ID_COLUMN],
        errors="coerce",
    )

    if ids.isna().any():
        raise ModelTrainingError(
            "id_cliente contiene valores inválidos."
        )

    if ids.duplicated().any():
        raise ModelTrainingError(
            "id_cliente contiene valores duplicados."
        )

    data[ID_COLUMN] = ids.astype("int64")

    for columna in KMEANS_FEATURES:
        serie = pd.to_numeric(
            data[columna],
            errors="coerce",
        )

        if serie.isna().any():
            raise ModelTrainingError(
                f"La variable {columna} contiene valores "
                "nulos o no numéricos."
            )

        if np.isinf(serie).any():
            raise ModelTrainingError(
                f"La variable {columna} contiene valores infinitos."
            )

        data[columna] = serie

    if len(data) < 3:
        raise ModelTrainingError(
            "KMeans requiere al menos tres registros."
        )

    return data


def evaluar_cantidades_clusters(
    x_escalado: np.ndarray,
    random_state: int,
    k_minimo: int = 2,
    k_maximo: int = 10,
) -> pd.DataFrame:
    """Calcula inercia y Silhouette para diferentes valores de k."""
    maximo_permitido = min(
        k_maximo,
        len(x_escalado) - 1,
    )

    if maximo_permitido < k_minimo:
        raise ModelTrainingError(
            "No existen suficientes registros para evaluar clusters."
        )

    resultados: list[dict[str, float | int]] = []

    for k in range(
        k_minimo,
        maximo_permitido + 1,
    ):
        modelo = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=10,
        )

        etiquetas = modelo.fit_predict(
            x_escalado
        )

        resultados.append(
            {
                "k": k,
                "inertia": float(modelo.inertia_),
                "silhouette": float(
                    silhouette_score(
                        x_escalado,
                        etiquetas,
                    )
                ),
            }
        )

    return pd.DataFrame(resultados)


def seleccionar_k_optimo(
    evaluacion_k: pd.DataFrame,
) -> tuple[int, str]:
    """
    Selecciona k mediante el método del codo.

    Si KneeLocator no encuentra un codo válido, utiliza el valor con
    mayor Silhouette como fallback controlado.
    """
    valores_k = evaluacion_k["k"].tolist()
    inercias = evaluacion_k["inertia"].tolist()

    detector = KneeLocator(
        valores_k,
        inercias,
        curve="convex",
        direction="decreasing",
    )

    if detector.elbow in valores_k:
        return int(detector.elbow), "metodo_codo"

    mejor_fila = evaluacion_k.loc[
        evaluacion_k["silhouette"].idxmax()
    ]

    return (
        int(mejor_fila["k"]),
        "silhouette_fallback",
    )


def entrenar_kmeans(
    df_base_limpio: pd.DataFrame,
    config: Settings = settings,
    persistir: bool = True,
) -> KMeansTrainingResult:
    """
    Entrena el modelo utilizando directamente el dataset limpio.

    La salida de esta función puede pasar posteriormente a validación
    de artefactos y publicación.
    """
    data = validar_entrada_kmeans(
        df_base_limpio
    )

    # Contrato numérico estable para entrenamiento e inferencia.
    # La preparación optimiza memoria y puede entregar float32; sin esta
    # normalización, KMeans puede guardar centroides float32 mientras la API
    # genera entradas float64, provocando "Buffer dtype mismatch".
    x = data[KMEANS_FEATURES].astype(np.float64)

    scaler = StandardScaler()
    x_escalado = scaler.fit_transform(x)
    x_escalado = np.asarray(x_escalado, dtype=np.float64)

    evaluacion_k = evaluar_cantidades_clusters(
        x_escalado=x_escalado,
        random_state=config.random_state,
    )

    k_optimo, metodo_seleccion = (
        seleccionar_k_optimo(evaluacion_k)
    )

    modelo = KMeans(
        n_clusters=k_optimo,
        random_state=config.random_state,
        n_init=10,
    )

    clusters = modelo.fit_predict(
        x_escalado
    )

    usuarios_segmentados = data.copy()
    usuarios_segmentados["cluster"] = clusters

    pca = PCA(n_components=2)
    componentes = pca.fit_transform(
        x_escalado
    )

    usuarios_segmentados["pc1"] = componentes[:, 0]
    usuarios_segmentados["pc2"] = componentes[:, 1]

    centroides_originales = scaler.inverse_transform(
        modelo.cluster_centers_
    )

    centroides = pd.DataFrame(
        centroides_originales,
        columns=KMEANS_FEATURES,
    )

    silhouette_final = silhouette_score(
        x_escalado,
        clusters,
    )

    metricas = {
        # Claves anteriores conservadas.
        "k_optimo": int(k_optimo),
        "silhouette_score": float(silhouette_final),
        "n_usuarios": int(len(data)),
        "varianza_pca": float(
            pca.explained_variance_ratio_.sum()
        ),

        # Metadatos nuevos.
        "metodo_seleccion_k": metodo_seleccion,
        "random_state": int(config.random_state),
        "n_features": len(KMEANS_FEATURES),
        "features": KMEANS_FEATURES,
        "dtype_entrenamiento": str(x_escalado.dtype),
    }

    resultado = KMeansTrainingResult(
        modelo=modelo,
        scaler=scaler,
        pca=pca,
        usuarios_segmentados=usuarios_segmentados,
        evaluacion_k=evaluacion_k,
        centroides=centroides,
        metricas=metricas,
    )

    if persistir:
        guardar_resultados_kmeans(
            resultado,
            config=config,
        )

    logger.info(
        "KMeans entrenado con k=%s, silhouette=%.4f.",
        k_optimo,
        silhouette_final,
    )

    return resultado


def _guardar_pickle_atomico(
    objeto: Any,
    ruta: Path,
) -> None:
    """Guarda un objeto pickle mediante un archivo temporal."""
    ruta.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporal = ruta.with_suffix(
        f"{ruta.suffix}.tmp"
    )

    try:
        with temporal.open("wb") as archivo:
            pickle.dump(
                objeto,
                archivo,
            )

        temporal.replace(ruta)

    except OSError as exc:
        if temporal.exists():
            temporal.unlink()

        raise ModelTrainingError(
            f"No fue posible guardar {ruta}: {exc}"
        ) from exc


def _guardar_csv_atomico(
    df: pd.DataFrame,
    ruta: Path,
) -> None:
    """Guarda un DataFrame mediante un archivo temporal."""
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

        raise ModelTrainingError(
            f"No fue posible guardar {ruta}: {exc}"
        ) from exc


def _guardar_json_atomico(
    contenido: dict[str, Any],
    ruta: Path,
) -> None:
    """Guarda las métricas mediante un archivo temporal."""
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

        raise ModelTrainingError(
            f"No fue posible guardar {ruta}: {exc}"
        ) from exc


def guardar_resultados_kmeans(
    resultado: KMeansTrainingResult,
    config: Settings = settings,
) -> None:
    """Persiste los mismos artefactos utilizados por API y dashboard."""
    config.models_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    config.data_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    _guardar_pickle_atomico(
        resultado.modelo,
        config.models_dir / "modelo_kmeans.pkl",
    )

    _guardar_pickle_atomico(
        resultado.scaler,
        config.models_dir / "scaler.pkl",
    )

    _guardar_pickle_atomico(
        resultado.pca,
        config.models_dir / "pca.pkl",
    )

    _guardar_json_atomico(
        resultado.metricas,
        config.models_dir / "metricas.json",
    )

    _guardar_csv_atomico(
        resultado.usuarios_segmentados,
        config.data_dir / "usuarios_segmentados.csv",
    )

    _guardar_csv_atomico(
        resultado.evaluacion_k,
        config.data_dir / "evaluacion_k.csv",
    )

    _guardar_csv_atomico(
        resultado.centroides,
        config.data_dir / "centroides.csv",
    )


def main() -> None:
    """Entrena KMeans usando el dataset base limpio persistido."""
    config = settings
    config.create_directories()

    dataset = cargar_dataset_base(
        config=config
    )

    resultado = entrenar_kmeans(
        df_base_limpio=dataset,
        config=config,
        persistir=True,
    )

    print(
        "KMeans completado | "
        f"k={resultado.metricas['k_optimo']} | "
        f"Silhouette="
        f"{resultado.metricas['silhouette_score']:.4f} | "
        f"Usuarios="
        f"{resultado.metricas['n_usuarios']}"
    )


if __name__ == "__main__":
    main()