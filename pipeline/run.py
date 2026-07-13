"""Ejecución integrada de extremo a extremo del proyecto.

La función principal conecta las salidas en memoria entre etapas:

    extracción -> validación -> integración -> preparación
    -> KMeans -> modelos supervisados -> validación de resultados

La publicación comienza únicamente después de que todas las etapas hayan
terminado correctamente. Cada archivo se escribe de forma atómica por los
módulos responsables y el manifiesto se publica al final como marca de una
ejecución completa.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from config.features import (
    CLASSIFICATION_FEATURES,
    KMEANS_FEATURES,
    REGRESSION_FEATURES,
)
from config.settings import Settings, settings
from etl.extract import extraer_fuentes, guardar_dataset_consolidado
from etl.integrate import IntegrationResult, integrar_fuentes
from etl.prepare_dataset import (
    PreparationResult,
    guardar_resultados,
    preparar_desde_dataframe,
)
from etl.validate import ValidationReport, validar_fuentes
from model.train import (
    KMeansTrainingResult,
    entrenar_kmeans,
    guardar_resultados_kmeans,
)
from model.train_supervisado import (
    SupervisedTrainingResult,
    entrenar_modelos_supervisados,
    guardar_resultados_supervisados,
)

logger = logging.getLogger(__name__)


class PipelineExecutionError(RuntimeError):
    """La ejecución integrada no cumple sus invariantes finales."""


@dataclass
class PipelineRunResult:
    """Productos y trazabilidad de una ejecución integrada."""

    validation_report: ValidationReport
    integration_result: IntegrationResult
    preparation_result: PreparationResult
    kmeans_result: KMeansTrainingResult
    supervised_result: SupervisedTrainingResult
    manifest: dict[str, Any]


def _feature_names(modelo: Any) -> list[str] | None:
    """Obtiene nombres de entrada cuando el estimador los conserva."""
    nombres = getattr(modelo, "feature_names_in_", None)
    if nombres is None:
        return None
    return [str(nombre) for nombre in nombres]


def validar_resultados_pipeline(
    integration: IntegrationResult,
    preparation: PreparationResult,
    kmeans: KMeansTrainingResult,
    supervised: SupervisedTrainingResult,
) -> None:
    """Comprueba compatibilidad entre las salidas antes de publicarlas."""
    errores: list[str] = []

    filas_integradas = len(integration.dataframe)
    filas_base = len(preparation.dataset_base_limpio)
    filas_analiticas = len(preparation.dataset_analitico)
    filas_segmentadas = len(kmeans.usuarios_segmentados)

    if not (filas_integradas == filas_base == filas_analiticas == filas_segmentadas):
        errores.append(
            "Las etapas no conservaron la misma cantidad de usuarios: "
            f"integración={filas_integradas}, base={filas_base}, "
            f"analítico={filas_analiticas}, segmentados={filas_segmentadas}."
        )

    if preparation.dataset_base_limpio["id_cliente"].duplicated().any():
        errores.append("El dataset base limpio contiene id_cliente duplicados.")

    if kmeans.usuarios_segmentados["id_cliente"].duplicated().any():
        errores.append("La salida segmentada contiene id_cliente duplicados.")

    columnas_kmeans = _feature_names(kmeans.scaler)
    if columnas_kmeans is not None and columnas_kmeans != KMEANS_FEATURES:
        errores.append(
            "El scaler KMeans no conserva el contrato oficial de variables."
        )

    if int(kmeans.modelo.n_features_in_) != len(KMEANS_FEATURES):
        errores.append("El modelo KMeans fue entrenado con otra dimensionalidad.")

    if kmeans.centroides.columns.tolist() != KMEANS_FEATURES:
        errores.append("Los centroides no utilizan las variables oficiales de KMeans.")

    columnas_regresion = _feature_names(supervised.regresion.modelo)
    if columnas_regresion is not None and columnas_regresion != REGRESSION_FEATURES:
        errores.append("El modelo de regresión no respeta REGRESSION_FEATURES.")

    columnas_clasificacion = _feature_names(supervised.clasificacion.modelo)
    if (
        columnas_clasificacion is not None
        and columnas_clasificacion != CLASSIFICATION_FEATURES
    ):
        errores.append(
            "El modelo de clasificación no respeta CLASSIFICATION_FEATURES."
        )

    valores_numericos = [
        kmeans.metricas["silhouette_score"],
        kmeans.metricas["varianza_pca"],
    ]
    if not all(np.isfinite(float(valor)) for valor in valores_numericos):
        errores.append("KMeans produjo métricas no finitas.")

    metadata_supervisada = supervised.metricas.get("metadata", {})
    if int(metadata_supervisada.get("n_filas", -1)) != filas_base:
        errores.append(
            "Las métricas supervisadas no corresponden al dataset base completo."
        )

    if errores:
        raise PipelineExecutionError(" | ".join(errores))


def _guardar_manifest_atomico(contenido: dict[str, Any], ruta: Path) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_suffix(f"{ruta.suffix}.tmp")
    try:
        with temporal.open("w", encoding="utf-8") as archivo:
            json.dump(contenido, archivo, indent=4, ensure_ascii=False)
        temporal.replace(ruta)
    except OSError as exc:
        if temporal.exists():
            temporal.unlink()
        raise PipelineExecutionError(
            f"No fue posible publicar el manifiesto {ruta}: {exc}"
        ) from exc


def _construir_manifest(
    inicio: datetime,
    duracion_segundos: float,
    validation: ValidationReport,
    integration: IntegrationResult,
    preparation: PreparationResult,
    kmeans: KMeansTrainingResult,
    supervised: SupervisedTrainingResult,
) -> dict[str, Any]:
    reg = supervised.metricas["regresion"]
    clf = supervised.metricas["clasificacion"]
    mejor_reg = reg["mejor_modelo"]
    mejor_clf = clf["mejor_modelo"]

    return {
        "status": "completed",
        "run_id": inicio.strftime("%Y%m%dT%H%M%SZ"),
        "started_at_utc": inicio.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": round(float(duracion_segundos), 3),
        "validation": {
            "is_valid": validation.is_valid,
            "errors": len(validation.errors),
            "warnings": len(validation.warnings),
        },
        "integration": integration.metadata,
        "preparation": {
            "dataset_base_limpio": list(
                preparation.dataset_base_limpio.shape
            ),
            "dataset_analitico": list(preparation.dataset_analitico.shape),
            "kpis": list(preparation.kpis.shape),
            "post_clean_validation": preparation.metadata.get(
                "validacion_post_limpieza", {}
            ),
        },
        "kmeans": {
            "k_optimo": int(kmeans.metricas["k_optimo"]),
            "silhouette_score": float(kmeans.metricas["silhouette_score"]),
            "varianza_pca": float(kmeans.metricas["varianza_pca"]),
            "features": KMEANS_FEATURES,
        },
        "supervised": {
            "regression": {
                "best_model": mejor_reg,
                "cv_r2": float(reg[mejor_reg]["cv_r2_mean"]),
                "test_r2": float(reg[mejor_reg]["r2"]),
                "features": REGRESSION_FEATURES,
            },
            "classification": {
                "best_model": mejor_clf,
                "cv_f1": float(clf[mejor_clf]["cv_f1_mean"]),
                "test_f1": float(clf[mejor_clf]["f1"]),
                "test_roc_auc": float(clf[mejor_clf]["roc_auc"]),
                "features": CLASSIFICATION_FEATURES,
            },
        },
        "publication": {
            "strategy": "deferred_until_all_stages_pass",
            "file_writes": "atomic_per_file",
            "completion_marker": "models/pipeline_manifest.json",
        },
    }


def publicar_resultados(
    resultado: PipelineRunResult,
    config: Settings = settings,
) -> None:
    """Publica las salidas después de validar todas las etapas en memoria."""
    guardar_dataset_consolidado(
        resultado.integration_result.dataframe,
        config=config,
    )
    guardar_resultados(
        resultado.preparation_result,
        config=config,
    )
    guardar_resultados_kmeans(
        resultado.kmeans_result,
        config=config,
    )
    guardar_resultados_supervisados(
        resultado.supervised_result,
        config=config,
    )

    # El manifiesto se escribe al final. Su presencia con status=completed
    # indica que la ejecución llegó al final de la publicación.
    _guardar_manifest_atomico(
        resultado.manifest,
        config.models_dir / "pipeline_manifest.json",
    )


def ejecutar_pipeline(
    config: Settings = settings,
    persistir: bool = True,
) -> PipelineRunResult:
    """Ejecuta el flujo completo conectando DataFrames y resultados en memoria."""
    config.create_directories()
    inicio = datetime.now(timezone.utc)
    reloj = time.perf_counter()

    logger.info("1/7 Extrayendo fuentes.")
    df_streaming, df_perfil = extraer_fuentes(config)

    logger.info("2/7 Validando estructura y compatibilidad.")
    validation = validar_fuentes(df_streaming, df_perfil)
    validation.raise_if_invalid()

    logger.info("3/7 Integrando fuentes one-to-one.")
    integration = integrar_fuentes(
        df_streaming=df_streaming,
        df_perfil=df_perfil,
        validation_report=validation,
    )

    logger.info("4/7 Limpiando y preparando datasets.")
    preparation = preparar_desde_dataframe(
        integration.dataframe,
        config=config,
        persistir=False,
    )

    logger.info("5/7 Entrenando KMeans.")
    kmeans = entrenar_kmeans(
        preparation.dataset_base_limpio,
        config=config,
        persistir=False,
    )

    logger.info("6/7 Entrenando modelos supervisados.")
    supervised = entrenar_modelos_supervisados(
        preparation.dataset_base_limpio,
        config=config,
        persistir=False,
    )

    logger.info("7/7 Validando compatibilidad de productos.")
    validar_resultados_pipeline(
        integration=integration,
        preparation=preparation,
        kmeans=kmeans,
        supervised=supervised,
    )

    duracion = time.perf_counter() - reloj
    manifest = _construir_manifest(
        inicio=inicio,
        duracion_segundos=duracion,
        validation=validation,
        integration=integration,
        preparation=preparation,
        kmeans=kmeans,
        supervised=supervised,
    )

    resultado = PipelineRunResult(
        validation_report=validation,
        integration_result=integration,
        preparation_result=preparation,
        kmeans_result=kmeans,
        supervised_result=supervised,
        manifest=manifest,
    )

    if persistir:
        logger.info("Publicando artefactos de la ejecución aprobada.")
        publicar_resultados(resultado, config=config)

    logger.info("Pipeline completado en %.2f segundos.", duracion)
    return resultado


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    resultado = ejecutar_pipeline(persistir=True)
    reg = resultado.supervised_result.metricas["regresion"]
    clf = resultado.supervised_result.metricas["clasificacion"]
    print(
        "Pipeline completo | "
        f"Dataset={resultado.preparation_result.dataset_base_limpio.shape} | "
        f"KMeans k={resultado.kmeans_result.metricas['k_optimo']} | "
        f"Regresión={reg['mejor_modelo']} | "
        f"Clasificación={clf['mejor_modelo']}"
    )


if __name__ == "__main__":
    main()
