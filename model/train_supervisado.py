"""Entrenamiento contractual de regresión y clasificación."""
from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config.features import (
    CLASSIFICATION_FEATURES,
    CLASSIFICATION_TARGET,
    CLASSIFICATION_TARGET_SOURCE_COLUMNS,
    ID_COLUMN,
    REGRESSION_FEATURES,
    REGRESSION_TARGET,
)
from config.settings import Settings, settings
from etl.contracts import ModelTrainingError

logger = logging.getLogger(__name__)

# Alias de compatibilidad con imports anteriores.
FEATURES = REGRESSION_FEATURES
FEATURES_CLASIFICACION = CLASSIFICATION_FEATURES


@dataclass
class RegressionTrainingResult:
    modelo: Any
    metricas: dict[str, Any]
    predicciones_test: pd.DataFrame


@dataclass
class ClassificationTrainingResult:
    modelo: Any
    metricas: dict[str, Any]
    predicciones_test: pd.DataFrame


@dataclass
class SupervisedTrainingResult:
    regresion: RegressionTrainingResult
    clasificacion: ClassificationTrainingResult
    metricas: dict[str, Any]


def cargar_dataset_base(ruta: str | Path | None = None, config: Settings = settings) -> pd.DataFrame:
    ruta_dataset = Path(ruta) if ruta is not None else config.clean_base_dataset
    if not ruta_dataset.exists():
        raise ModelTrainingError(
            f"No se encontró el dataset base limpio: {ruta_dataset}. "
            "Ejecuta primero la etapa de preparación."
        )
    try:
        df = pd.read_csv(ruta_dataset)
    except (OSError, pd.errors.ParserError) as exc:
        raise ModelTrainingError(f"No fue posible cargar el dataset base limpio: {exc}") from exc
    if df.empty:
        raise ModelTrainingError("El dataset base limpio está vacío.")
    return df


def validar_entrada_supervisada(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise ModelTrainingError("El entrenamiento supervisado requiere un DataFrame.")
    if df.empty:
        raise ModelTrainingError("No se puede entrenar con un dataset vacío.")

    requeridas = {
        ID_COLUMN,
        REGRESSION_TARGET,
        *REGRESSION_FEATURES,
        *CLASSIFICATION_FEATURES,
        *CLASSIFICATION_TARGET_SOURCE_COLUMNS,
    }
    faltantes = sorted(requeridas - set(df.columns))
    if faltantes:
        raise ModelTrainingError(f"Faltan columnas para los modelos supervisados: {faltantes}")

    fuga = sorted(set(CLASSIFICATION_FEATURES) & set(CLASSIFICATION_TARGET_SOURCE_COLUMNS))
    if fuga:
        raise ModelTrainingError(
            "Las variables que construyen la etiqueta de clasificación no pueden usarse como features: "
            f"{fuga}"
        )
    if REGRESSION_TARGET in REGRESSION_FEATURES:
        raise ModelTrainingError("El target de regresión aparece dentro de sus features.")

    columnas = [
        ID_COLUMN,
        REGRESSION_TARGET,
        *REGRESSION_FEATURES,
        *CLASSIFICATION_FEATURES,
        *CLASSIFICATION_TARGET_SOURCE_COLUMNS,
    ]
    columnas = list(dict.fromkeys(columnas))
    data = df[columnas].copy()

    for columna in columnas:
        serie = pd.to_numeric(data[columna], errors="coerce")
        if serie.isna().any():
            raise ModelTrainingError(f"La columna {columna} contiene valores nulos o no numéricos.")
        if np.isinf(serie).any():
            raise ModelTrainingError(f"La columna {columna} contiene valores infinitos.")
        data[columna] = serie

    if data[ID_COLUMN].duplicated().any():
        raise ModelTrainingError("id_cliente contiene valores duplicados.")
    if len(data) < 30:
        raise ModelTrainingError("Se requieren al menos 30 registros para entrenar y evaluar.")
    return data


def _ajustar_candidatos(
    candidatos: dict[str, dict[str, Any]],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    cv: Any,
    scoring: str,
    n_jobs: int,
) -> dict[str, dict[str, Any]]:
    ajustados: dict[str, dict[str, Any]] = {}
    for nombre, especificacion in candidatos.items():
        buscador = GridSearchCV(
            estimator=especificacion["pipeline"],
            param_grid=especificacion["grid"],
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            refit=True,
            return_train_score=False,
            error_score="raise",
        )
        buscador.fit(x_train, y_train)
        indice = int(buscador.best_index_)
        ajustados[nombre] = {
            "modelo": buscador.best_estimator_,
            "best_params": buscador.best_params_,
            "cv_score_mean": float(buscador.best_score_),
            "cv_score_std": float(buscador.cv_results_["std_test_score"][indice]),
        }
        logger.info(
            "%s: mejor CV=%.4f ± %.4f | params=%s",
            nombre,
            ajustados[nombre]["cv_score_mean"],
            ajustados[nombre]["cv_score_std"],
            ajustados[nombre]["best_params"],
        )
    return ajustados


def entrenar_regresion(
    data: pd.DataFrame,
    config: Settings = settings,
) -> RegressionTrainingResult:
    data = validar_entrada_supervisada(data)
    x = data[REGRESSION_FEATURES]
    y = data[REGRESSION_TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=config.random_state,
    )

    candidatos = {
        "regresion_lineal": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("modelo", LinearRegression()),
            ]),
            "grid": [{}],
        },
        "random_forest_regressor": {
            "pipeline": Pipeline([
                ("modelo", RandomForestRegressor(random_state=config.random_state)),
            ]),
            "grid": {
                "modelo__n_estimators": [100, 200],
                "modelo__max_depth": [None, 5, 10],
            },
        },
    }
    cv = KFold(n_splits=5, shuffle=True, random_state=config.random_state)
    ajustados = _ajustar_candidatos(
        candidatos, x_train, y_train, cv, "r2", config.n_jobs
    )

    metricas_candidatos: dict[str, Any] = {}
    for nombre, info in ajustados.items():
        pred = info["modelo"].predict(x_test)
        metricas_candidatos[nombre] = {
            "r2": float(r2_score(y_test, pred)),
            "mae": float(mean_absolute_error(y_test, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
            "cv_r2_mean": info["cv_score_mean"],
            "cv_r2_std": info["cv_score_std"],
            "best_params": info["best_params"],
        }

    mejor_nombre = max(
        ajustados,
        key=lambda nombre: ajustados[nombre]["cv_score_mean"],
    )
    modelo_evaluado = ajustados[mejor_nombre]["modelo"]
    pred_final = modelo_evaluado.predict(x_test)

    # Tras seleccionar y evaluar, se reajusta el ganador con todos los datos
    # para producir el artefacto usado por la API.
    modelo_produccion = clone(modelo_evaluado)
    modelo_produccion.fit(x, y)

    predicciones = pd.DataFrame({
        ID_COLUMN: data.loc[x_test.index, ID_COLUMN].astype("int64"),
        "valor_real": y_test,
        "valor_predicho": pred_final,
        "residuo": y_test - pred_final,
    }).reset_index(drop=True)

    metricas = {
        **metricas_candidatos,
        "mejor_modelo": mejor_nombre,
        "criterio_seleccion": "mayor_cv_r2_mean_en_train",
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
    }
    logger.info(
        "Regresión seleccionada por CV: %s | test R2=%.3f",
        mejor_nombre,
        metricas_candidatos[mejor_nombre]["r2"],
    )
    return RegressionTrainingResult(modelo_produccion, metricas, predicciones)


def _crear_etiqueta_riesgo(
    data: pd.DataFrame,
    umbral_finalizacion: float,
    umbral_sesiones: float,
) -> pd.Series:
    return (
        (data["porcentaje_finalizacion"] < umbral_finalizacion)
        & (data["sesiones_semana"] < umbral_sesiones)
    ).astype("int8")


def entrenar_clasificacion(
    data: pd.DataFrame,
    config: Settings = settings,
) -> ClassificationTrainingResult:
    data = validar_entrada_supervisada(data)

    indices_train, indices_test = train_test_split(
        data.index,
        test_size=0.20,
        random_state=config.random_state,
    )
    train_data = data.loc[indices_train]
    test_data = data.loc[indices_test]

    # Los umbrales se calculan solo con train. El holdout no participa
    # en la construcción de la etiqueta ni en la selección del modelo.
    umbral_finalizacion = float(train_data["porcentaje_finalizacion"].median())
    umbral_sesiones = float(train_data["sesiones_semana"].median())
    y_train = _crear_etiqueta_riesgo(train_data, umbral_finalizacion, umbral_sesiones)
    y_test = _crear_etiqueta_riesgo(test_data, umbral_finalizacion, umbral_sesiones)

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise ModelTrainingError(
            "La división train/test no contiene ambas clases de riesgo. "
            "Ajusta random_state o la definición de la etiqueta."
        )
    minimo_clase = int(y_train.value_counts().min())
    n_splits = min(5, minimo_clase)
    if n_splits < 2:
        raise ModelTrainingError("No hay suficientes ejemplos por clase para validación cruzada.")

    x_train = train_data[CLASSIFICATION_FEATURES]
    x_test = test_data[CLASSIFICATION_FEATURES]
    candidatos = {
        "regresion_logistica": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("modelo", LogisticRegression(
                    max_iter=1000,
                    random_state=config.random_state,
                )),
            ]),
            "grid": {"modelo__C": [0.1, 1.0, 10.0]},
        },
        "random_forest_classifier": {
            "pipeline": Pipeline([
                ("modelo", RandomForestClassifier(random_state=config.random_state)),
            ]),
            "grid": {
                "modelo__n_estimators": [100, 200],
                "modelo__max_depth": [None, 5, 10],
            },
        },
    }
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=config.random_state,
    )
    ajustados = _ajustar_candidatos(
        candidatos, x_train, y_train, cv, "f1", config.n_jobs
    )

    metricas_candidatos: dict[str, Any] = {}
    for nombre, info in ajustados.items():
        pred = info["modelo"].predict(x_test)
        proba = info["modelo"].predict_proba(x_test)[:, 1]
        matriz = confusion_matrix(y_test, pred, labels=[0, 1])
        metricas_candidatos[nombre] = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, proba)),
            "cv_f1_mean": info["cv_score_mean"],
            "cv_f1_std": info["cv_score_std"],
            "best_params": info["best_params"],
            "confusion_matrix": matriz.astype(int).tolist(),
        }

    mejor_nombre = max(
        ajustados,
        key=lambda nombre: ajustados[nombre]["cv_score_mean"],
    )
    modelo_evaluado = ajustados[mejor_nombre]["modelo"]
    pred_final = modelo_evaluado.predict(x_test)
    proba_final = modelo_evaluado.predict_proba(x_test)[:, 1]

    # Reajuste de producción con todos los datos, manteniendo los umbrales
    # aprendidos exclusivamente desde el conjunto train original.
    y_completo = _crear_etiqueta_riesgo(data, umbral_finalizacion, umbral_sesiones)
    modelo_produccion = clone(modelo_evaluado)
    modelo_produccion.fit(data[CLASSIFICATION_FEATURES], y_completo)

    predicciones = pd.DataFrame({
        ID_COLUMN: test_data[ID_COLUMN].astype("int64"),
        "valor_real": y_test,
        "valor_predicho": pred_final.astype(int),
        "probabilidad_riesgo": proba_final,
    }).reset_index(drop=True)

    metricas = {
        **metricas_candidatos,
        "mejor_modelo": mejor_nombre,
        "criterio_seleccion": "mayor_cv_f1_mean_en_train",
        "umbral_finalizacion": umbral_finalizacion,
        "umbral_sesiones": umbral_sesiones,
        "origen_umbrales": "solo_conjunto_train",
        "positivos_train": int(y_train.sum()),
        "positivos_test": int(y_test.sum()),
        "n_train": int(len(train_data)),
        "n_test": int(len(test_data)),
    }
    logger.info(
        "Clasificación seleccionada por CV: %s | test F1=%.3f | AUC=%.3f",
        mejor_nombre,
        metricas_candidatos[mejor_nombre]["f1"],
        metricas_candidatos[mejor_nombre]["roc_auc"],
    )
    return ClassificationTrainingResult(modelo_produccion, metricas, predicciones)


def entrenar_modelos_supervisados(
    df_base_limpio: pd.DataFrame,
    config: Settings = settings,
    persistir: bool = True,
) -> SupervisedTrainingResult:
    data = validar_entrada_supervisada(df_base_limpio)
    regresion = entrenar_regresion(data, config=config)
    clasificacion = entrenar_clasificacion(data, config=config)
    metricas = {
        "regresion": regresion.metricas,
        "clasificacion": clasificacion.metricas,
        "metadata": {
            "dataset_entrada": "dataset_base_limpio",
            "n_filas": int(len(data)),
            "random_state": int(config.random_state),
            "n_jobs": int(config.n_jobs),
            "regression_target": REGRESSION_TARGET,
            "regression_features": REGRESSION_FEATURES,
            "classification_target": CLASSIFICATION_TARGET,
            "classification_features": CLASSIFICATION_FEATURES,
            "classification_target_source_columns": CLASSIFICATION_TARGET_SOURCE_COLUMNS,
            "seleccion": "solo_validacion_cruzada_en_train",
            "test": "auditoria_final_sin_seleccion",
            "production_refit": True,
        },
    }
    resultado = SupervisedTrainingResult(regresion, clasificacion, metricas)
    if persistir:
        guardar_resultados_supervisados(resultado, config=config)
    return resultado


def _guardar_pickle_atomico(objeto: Any, ruta: Path) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_suffix(f"{ruta.suffix}.tmp")
    try:
        with temporal.open("wb") as archivo:
            pickle.dump(objeto, archivo)
        temporal.replace(ruta)
    except OSError as exc:
        if temporal.exists():
            temporal.unlink()
        raise ModelTrainingError(f"No fue posible guardar {ruta}: {exc}") from exc


def _guardar_json_atomico(contenido: dict[str, Any], ruta: Path) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_suffix(f"{ruta.suffix}.tmp")
    try:
        with temporal.open("w", encoding="utf-8") as archivo:
            json.dump(contenido, archivo, indent=4, ensure_ascii=False)
        temporal.replace(ruta)
    except OSError as exc:
        if temporal.exists():
            temporal.unlink()
        raise ModelTrainingError(f"No fue posible guardar {ruta}: {exc}") from exc


def _guardar_csv_atomico(df: pd.DataFrame, ruta: Path) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_suffix(f"{ruta.suffix}.tmp")
    try:
        df.to_csv(temporal, index=False)
        temporal.replace(ruta)
    except OSError as exc:
        if temporal.exists():
            temporal.unlink()
        raise ModelTrainingError(f"No fue posible guardar {ruta}: {exc}") from exc


def guardar_resultados_supervisados(
    resultado: SupervisedTrainingResult,
    config: Settings = settings,
) -> None:
    config.create_directories()
    _guardar_pickle_atomico(
        resultado.regresion.modelo,
        config.models_dir / "modelo_regresion_gasto.pkl",
    )
    _guardar_pickle_atomico(
        resultado.clasificacion.modelo,
        config.models_dir / "modelo_clasificacion_riesgo.pkl",
    )
    _guardar_json_atomico(
        resultado.metricas,
        config.models_dir / "metricas_supervisado.json",
    )
    _guardar_csv_atomico(
        resultado.regresion.predicciones_test,
        config.data_dir / "predicciones_regresion_test.csv",
    )
    _guardar_csv_atomico(
        resultado.clasificacion.predicciones_test,
        config.data_dir / "predicciones_clasificacion_test.csv",
    )


def main() -> None:
    config = settings
    config.create_directories()
    dataset = cargar_dataset_base(config=config)
    resultado = entrenar_modelos_supervisados(dataset, config=config, persistir=True)
    reg = resultado.metricas["regresion"]
    clf = resultado.metricas["clasificacion"]
    mejor_reg = reg["mejor_modelo"]
    mejor_clf = clf["mejor_modelo"]
    print(
        "Modelos supervisados completados | "
        f"Regresión={mejor_reg} (CV R²={reg[mejor_reg]['cv_r2_mean']:.3f}, "
        f"test R²={reg[mejor_reg]['r2']:.3f}) | "
        f"Clasificación={mejor_clf} (CV F1={clf[mejor_clf]['cv_f1_mean']:.3f}, "
        f"test F1={clf[mejor_clf]['f1']:.3f})"
    )


if __name__ == "__main__":
    main()