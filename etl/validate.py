"""
Validación estructural y de calidad previa de las fuentes.

Este módulo no vuelve a leer archivos ni consulta PostgreSQL.
Recibe directamente los DataFrames producidos por la extracción.

Errores estructurales:
    Detienen el pipeline.

Problemas de calidad corregibles:
    Se registran como advertencias para la etapa de limpieza.
"""

from __future__ import annotations

import logging
from typing import Collection

import numpy as np
import pandas as pd

from etl.contracts import (
    PROFILE_REQUIRED_COLUMNS,
    STREAMING_REQUIRED_COLUMNS,
    ValidationReport,
    BASE_COLUMN_ORDER,
    BASE_REQUIRED_COLUMNS,
    COLUMN_RANGES,
    INTEGER_COLUMNS,
)


logger = logging.getLogger(__name__)


def validar_fuente(
    df: pd.DataFrame,
    columnas_requeridas: Collection[str],
    nombre_fuente: str,
) -> ValidationReport:
    """
    Valida la estructura de una fuente antes de su integración.

    Los problemas que imposibilitan identificar o integrar registros
    se consideran errores. Los datos analíticos potencialmente
    corregibles se registran como advertencias.

    Parameters
    ----------
    df:
        DataFrame extraído desde la fuente real.
    columnas_requeridas:
        Columnas mínimas exigidas por el contrato.
    nombre_fuente:
        Nombre utilizado en reportes y mensajes.

    Returns
    -------
    ValidationReport
        Resultado estructurado de la validación.
    """
    report = ValidationReport(
        stage=f"estructura_{nombre_fuente}",
        metadata={
            "fuente": nombre_fuente,
        },
    )

    if not isinstance(df, pd.DataFrame):
        report.add_error(
            code="INVALID_SOURCE_TYPE",
            message=(
                f"La fuente {nombre_fuente} no fue entregada "
                "como DataFrame."
            ),
            received_type=type(df).__name__,
        )
        return report

    report.metadata.update(
        {
            "filas": int(df.shape[0]),
            "columnas": int(df.shape[1]),
        }
    )

    if df.empty:
        report.add_error(
            code="EMPTY_SOURCE",
            message=f"La fuente {nombre_fuente} no contiene registros.",
        )
        return report

    columnas_requeridas = set(columnas_requeridas)
    columnas_actuales = set(df.columns)

    faltantes = sorted(columnas_requeridas - columnas_actuales)
    adicionales = sorted(columnas_actuales - columnas_requeridas)

    if faltantes:
        report.add_error(
            code="MISSING_REQUIRED_COLUMNS",
            message=(
                f"La fuente {nombre_fuente} no contiene todas "
                "las columnas requeridas."
            ),
            missing_columns=faltantes,
        )

    if adicionales:
        report.add_warning(
            code="UNEXPECTED_COLUMNS",
            message=(
                f"La fuente {nombre_fuente} contiene columnas "
                "adicionales no contempladas por el contrato."
            ),
            additional_columns=adicionales,
        )

    # Sin el identificador no pueden ejecutarse las comprobaciones
    # de unicidad y compatibilidad.
    if "id_cliente" not in df.columns:
        return report

    ids_numericos = pd.to_numeric(
        df["id_cliente"],
        errors="coerce",
    )

    ids_invalidos = int(ids_numericos.isna().sum())

    if ids_invalidos > 0:
        report.add_error(
            code="INVALID_CUSTOMER_ID",
            message=(
                f"La fuente {nombre_fuente} contiene identificadores "
                "nulos o no convertibles a número."
            ),
            column="id_cliente",
            invalid_count=ids_invalidos,
        )

    ids_no_positivos = int(
        (ids_numericos.dropna() <= 0).sum()
    )

    if ids_no_positivos > 0:
        report.add_error(
            code="NON_POSITIVE_CUSTOMER_ID",
            message=(
                f"La fuente {nombre_fuente} contiene identificadores "
                "menores o iguales a cero."
            ),
            column="id_cliente",
            invalid_count=ids_no_positivos,
        )

    duplicados = int(
        ids_numericos.dropna().duplicated().sum()
    )

    if duplicados > 0:
        report.add_error(
            code="DUPLICATED_CUSTOMER_ID",
            message=(
                f"La fuente {nombre_fuente} contiene identificadores "
                "duplicados."
            ),
            column="id_cliente",
            duplicated_count=duplicados,
        )

    # Estas comprobaciones solo se realizan sobre columnas presentes.
    columnas_disponibles = sorted(
        columnas_requeridas.intersection(columnas_actuales)
    )

    for columna in columnas_disponibles:
        if columna == "id_cliente":
            continue

        serie = df[columna]

        cantidad_nulos = int(serie.isna().sum())

        if cantidad_nulos > 0:
            report.add_warning(
                code="NULL_VALUES",
                message=(
                    f"La columna {columna} de {nombre_fuente} "
                    "contiene valores nulos que deberán tratarse."
                ),
                column=columna,
                null_count=cantidad_nulos,
            )

        if not pd.api.types.is_numeric_dtype(serie):
            conversion = pd.to_numeric(
                serie,
                errors="coerce",
            )

            no_convertibles = int(
                conversion.isna().sum() - serie.isna().sum()
            )

            if no_convertibles > 0:
                report.add_warning(
                    code="NON_NUMERIC_VALUES",
                    message=(
                        f"La columna {columna} de {nombre_fuente} "
                        "contiene valores no convertibles a número."
                    ),
                    column=columna,
                    invalid_count=no_convertibles,
                )
            else:
                report.add_warning(
                    code="CONVERTIBLE_NUMERIC_TYPE",
                    message=(
                        f"La columna {columna} de {nombre_fuente} "
                        "no tiene tipo numérico, pero puede convertirse."
                    ),
                    column=columna,
                    current_dtype=str(serie.dtype),
                )

            serie_numerica = conversion
        else:
            serie_numerica = serie

        cantidad_infinitos = int(
            np.isinf(
                pd.to_numeric(
                    serie_numerica,
                    errors="coerce",
                )
            ).sum()
        )

        if cantidad_infinitos > 0:
            report.add_warning(
                code="INFINITE_VALUES",
                message=(
                    f"La columna {columna} de {nombre_fuente} "
                    "contiene valores infinitos."
                ),
                column=columna,
                infinite_count=cantidad_infinitos,
            )

    logger.info(
        "Fuente %s validada: %s errores, %s advertencias.",
        nombre_fuente,
        len(report.errors),
        len(report.warnings),
    )

    return report


def validar_compatibilidad_ids(
    df_streaming: pd.DataFrame,
    df_perfil: pd.DataFrame,
) -> ValidationReport:
    """
    Verifica que ambas fuentes representen el mismo universo de usuarios.

    Returns
    -------
    ValidationReport
        Reporte de compatibilidad previo a la integración.
    """
    report = ValidationReport(
        stage="compatibilidad_ids",
    )

    if "id_cliente" not in df_streaming.columns:
        report.add_error(
            code="MISSING_STREAMING_ID",
            message=(
                "La fuente de streaming no contiene id_cliente."
            ),
        )
        return report

    if "id_cliente" not in df_perfil.columns:
        report.add_error(
            code="MISSING_PROFILE_ID",
            message=(
                "La fuente de perfiles no contiene id_cliente."
            ),
        )
        return report

    ids_streaming = set(
        pd.to_numeric(
            df_streaming["id_cliente"],
            errors="coerce",
        ).dropna()
    )

    ids_perfil = set(
        pd.to_numeric(
            df_perfil["id_cliente"],
            errors="coerce",
        ).dropna()
    )

    solo_streaming = sorted(ids_streaming - ids_perfil)
    solo_perfil = sorted(ids_perfil - ids_streaming)

    report.metadata.update(
        {
            "ids_streaming": len(ids_streaming),
            "ids_perfil": len(ids_perfil),
            "ids_compartidos": len(
                ids_streaming.intersection(ids_perfil)
            ),
            "ids_solo_streaming": len(solo_streaming),
            "ids_solo_perfil": len(solo_perfil),
        }
    )

    if solo_streaming or solo_perfil:
        report.add_error(
            code="CUSTOMER_ID_MISMATCH",
            message=(
                "Las fuentes no contienen exactamente el mismo "
                "conjunto de usuarios."
            ),
            only_streaming_count=len(solo_streaming),
            only_profile_count=len(solo_perfil),
            only_streaming_sample=solo_streaming[:10],
            only_profile_sample=solo_perfil[:10],
        )

    return report


def validar_fuentes(
    df_streaming: pd.DataFrame,
    df_perfil: pd.DataFrame,
) -> ValidationReport:
    """
    Ejecuta todas las validaciones previas a la integración.

    Returns
    -------
    ValidationReport
        Reporte consolidado de ambas fuentes.
    """
    streaming_report = validar_fuente(
        df=df_streaming,
        columnas_requeridas=STREAMING_REQUIRED_COLUMNS,
        nombre_fuente="usuarios_streaming",
    )

    perfil_report = validar_fuente(
        df=df_perfil,
        columnas_requeridas=PROFILE_REQUIRED_COLUMNS,
        nombre_fuente="perfil_usuarios_postgresql",
    )

    errores_id = {
        "INVALID_CUSTOMER_ID",
        "NON_POSITIVE_CUSTOMER_ID",
        "DUPLICATED_CUSTOMER_ID",
        "MISSING_STREAMING_ID",
        "MISSING_PROFILE_ID",
    }

    issues_fuentes = (
        streaming_report.issues
        + perfil_report.issues
    )

    existe_error_id = any(
        issue.severity == "error"
        and issue.code in errores_id
        for issue in issues_fuentes
    )

    if existe_error_id:
        compatibilidad_report = ValidationReport(
            stage="compatibilidad_ids",
            metadata={
                "skipped": True,
                "reason": (
                    "La comparación de IDs fue omitida porque "
                    "alguna fuente contiene identificadores inválidos."
                ),
            },
        )
    else:
        compatibilidad_report = validar_compatibilidad_ids(
            df_streaming,
            df_perfil,
        )

    report = ValidationReport(
        stage="validacion_estructural_fuentes",
        metadata={
            "streaming": streaming_report.metadata,
            "perfil": perfil_report.metadata,
            "compatibilidad": compatibilidad_report.metadata,
        },
    )

    report.issues.extend(streaming_report.issues)
    report.issues.extend(perfil_report.issues)
    report.issues.extend(compatibilidad_report.issues)

    return report


def validar_dataset_limpio(
    df: pd.DataFrame,
) -> ValidationReport:
    """
    Verifica que el dataset base limpio pueda alimentar los modelos.

    Esta validación se ejecuta después de la limpieza. A diferencia de
    la validación previa, aquí los nulos, infinitos y valores fuera de
    rango son errores bloqueantes.
    """
    report = ValidationReport(
        stage="calidad_dataset_base_limpio",
    )

    if not isinstance(df, pd.DataFrame):
        report.add_error(
            code="INVALID_CLEAN_DATASET_TYPE",
            message="El dataset limpio no es un DataFrame.",
            received_type=type(df).__name__,
        )
        return report

    report.metadata.update(
        {
            "filas": int(df.shape[0]),
            "columnas": int(df.shape[1]),
        }
    )

    if df.empty:
        report.add_error(
            code="EMPTY_CLEAN_DATASET",
            message="El dataset limpio no contiene registros.",
        )
        return report

    columnas_actuales = set(df.columns)
    faltantes = sorted(BASE_REQUIRED_COLUMNS - columnas_actuales)
    adicionales = sorted(columnas_actuales - BASE_REQUIRED_COLUMNS)

    if faltantes:
        report.add_error(
            code="MISSING_CLEAN_COLUMNS",
            message="El dataset limpio no contiene todas las columnas base.",
            missing_columns=faltantes,
        )
        return report

    if adicionales:
        report.add_warning(
            code="UNEXPECTED_CLEAN_COLUMNS",
            message="El dataset base limpio contiene columnas adicionales.",
            additional_columns=adicionales,
        )

    if list(df.columns) != BASE_COLUMN_ORDER:
        report.add_warning(
            code="NON_STANDARD_COLUMN_ORDER",
            message="Las columnas no siguen el orden oficial del contrato.",
        )

    ids = pd.to_numeric(df["id_cliente"], errors="coerce")

    if ids.isna().any():
        report.add_error(
            code="INVALID_CLEAN_CUSTOMER_ID",
            message="El dataset limpio contiene IDs nulos o no numéricos.",
            column="id_cliente",
            invalid_count=int(ids.isna().sum()),
        )

    if ids.dropna().duplicated().any():
        report.add_error(
            code="DUPLICATED_CLEAN_CUSTOMER_ID",
            message="El dataset limpio contiene IDs duplicados.",
            column="id_cliente",
            duplicated_count=int(ids.dropna().duplicated().sum()),
        )

    for columna in BASE_COLUMN_ORDER:
        serie = pd.to_numeric(df[columna], errors="coerce")

        cantidad_invalidos = int(serie.isna().sum())

        if cantidad_invalidos > 0:
            report.add_error(
                code="NULL_OR_NON_NUMERIC_AFTER_CLEANING",
                message=(
                    f"La columna {columna} conserva valores nulos "
                    "o no numéricos después de la limpieza."
                ),
                column=columna,
                invalid_count=cantidad_invalidos,
            )
            continue

        cantidad_infinitos = int(np.isinf(serie).sum())

        if cantidad_infinitos > 0:
            report.add_error(
                code="INFINITE_AFTER_CLEANING",
                message=(
                    f"La columna {columna} conserva valores infinitos "
                    "después de la limpieza."
                ),
                column=columna,
                infinite_count=cantidad_infinitos,
            )

        minimo, maximo = COLUMN_RANGES.get(
            columna,
            (None, None),
        )

        if minimo is not None:
            bajo_minimo = int((serie < minimo).sum())

            if bajo_minimo > 0:
                report.add_error(
                    code="VALUE_BELOW_MINIMUM",
                    message=(
                        f"La columna {columna} contiene valores "
                        f"menores que {minimo}."
                    ),
                    column=columna,
                    invalid_count=bajo_minimo,
                    minimum=minimo,
                )

        if maximo is not None:
            sobre_maximo = int((serie > maximo).sum())

            if sobre_maximo > 0:
                report.add_error(
                    code="VALUE_ABOVE_MAXIMUM",
                    message=(
                        f"La columna {columna} contiene valores "
                        f"mayores que {maximo}."
                    ),
                    column=columna,
                    invalid_count=sobre_maximo,
                    maximum=maximo,
                )

        if columna in INTEGER_COLUMNS:
            no_enteros = int(
                ((serie % 1) != 0).sum()
            )

            if no_enteros > 0:
                report.add_error(
                    code="NON_INTEGER_VALUE",
                    message=(
                        f"La columna {columna} debe contener "
                        "valores enteros."
                    ),
                    column=columna,
                    invalid_count=no_enteros,
                )

    return report


def main() -> None:
    """
    Prueba manual de extracción y validación con las fuentes reales.

    No sustituye todavía al orquestador final.
    """
    from etl.extract import extraer_fuentes, integrar

    logger.info("Iniciando validación estructural de fuentes.")

    df_streaming, df_perfil = extraer_fuentes()

    report = validar_fuentes(
        df_streaming,
        df_perfil,
    )

    report.raise_if_invalid()

    df_integrado = integrar(
        df_streaming,
        df_perfil,
    )

    logger.info(
        "Validación e integración completadas: %s filas, %s columnas.",
        df_integrado.shape[0],
        df_integrado.shape[1],
    )

    print(
        "Validación estructural correcta | "
        f"Errores: {len(report.errors)} | "
        f"Advertencias: {len(report.warnings)} | "
        f"Dataset integrado: {df_integrado.shape}"
    )


if __name__ == "__main__":
    main()