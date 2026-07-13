from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


STREAMING_COLUMN_ORDER = [
    "id_cliente",
    "horas_consumo_mensual",
    "gasto_mensual",
    "cantidad_contenidos_vistos",
    "sesiones_semana",
    "porcentaje_finalizacion",
    "tiempo_promedio_sesion_min",
    "cantidad_generos_consumidos",
    "porcentaje_uso_promociones",
    "antiguedad_cliente_meses",
]


PROFILE_COLUMN_ORDER = [
    "id_cliente",
    "edad",
    "dispositivos_registrados",
    "porcentaje_uso_app_movil",
    "cantidad_perfiles_creados",
    "interacciones_mensuales_soporte",
    "distancia_promedio_red_km",
]


BASE_COLUMN_ORDER = (
    STREAMING_COLUMN_ORDER
    + [
        columna
        for columna in PROFILE_COLUMN_ORDER
        if columna != "id_cliente"
    ]
)


# Orden oficial de las columnas de la fuente de streaming.
STREAMING_COLUMN_ORDER = [
    "id_cliente",
    "horas_consumo_mensual",
    "gasto_mensual",
    "cantidad_contenidos_vistos",
    "sesiones_semana",
    "porcentaje_finalizacion",
    "tiempo_promedio_sesion_min",
    "cantidad_generos_consumidos",
    "porcentaje_uso_promociones",
    "antiguedad_cliente_meses",
]


# Orden oficial de las columnas de la fuente de perfiles.
PROFILE_COLUMN_ORDER = [
    "id_cliente",
    "edad",
    "dispositivos_registrados",
    "porcentaje_uso_app_movil",
    "cantidad_perfiles_creados",
    "interacciones_mensuales_soporte",
    "distancia_promedio_red_km",
]


# Orden oficial del dataset después de integrar ambas fuentes.
# id_cliente aparece solo una vez.
BASE_COLUMN_ORDER = (
    STREAMING_COLUMN_ORDER
    + [
        columna
        for columna in PROFILE_COLUMN_ORDER
        if columna != "id_cliente"
    ]
)


# Conjuntos utilizados para validar presencia de columnas.
STREAMING_REQUIRED_COLUMNS = set(STREAMING_COLUMN_ORDER)

PROFILE_REQUIRED_COLUMNS = set(PROFILE_COLUMN_ORDER)

BASE_REQUIRED_COLUMNS = set(BASE_COLUMN_ORDER)


# Rangos de valores imposibles o inválidos.
# No representan todavía reglas estadísticas de outliers.
COLUMN_RANGES: dict[str, tuple[float | None, float | None]] = {
    "id_cliente": (1, None),
    "horas_consumo_mensual": (0, None),
    "gasto_mensual": (0, None),
    "cantidad_contenidos_vistos": (0, None),
    "sesiones_semana": (0, None),
    "porcentaje_finalizacion": (0, 100),
    "tiempo_promedio_sesion_min": (0, None),
    "cantidad_generos_consumidos": (0, None),
    "porcentaje_uso_promociones": (0, 1),
    "antiguedad_cliente_meses": (0, None),
    "edad": (13, 100),
    "dispositivos_registrados": (0, None),
    "porcentaje_uso_app_movil": (0, 1),
    "cantidad_perfiles_creados": (0, None),
    "interacciones_mensuales_soporte": (0, None),
    "distancia_promedio_red_km": (0, None),
}


INTEGER_COLUMNS = {
    "id_cliente",
    "cantidad_contenidos_vistos",
    "sesiones_semana",
    "cantidad_generos_consumidos",
    "antiguedad_cliente_meses",
    "edad",
    "dispositivos_registrados",
    "cantidad_perfiles_creados",
    "interacciones_mensuales_soporte",
}


NUMERIC_COLUMNS = BASE_REQUIRED_COLUMNS


Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class ValidationIssue:
    """Representa un problema encontrado durante una validación."""

    code: str
    message: str
    severity: Severity
    column: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Resultado estructurado de una validación."""

    stage: str
    issues: list[ValidationIssue] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [
            issue
            for issue in self.issues
            if issue.severity == "error"
        ]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [
            issue
            for issue in self.issues
            if issue.severity == "warning"
        ]

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def add_error(
        self,
        code: str,
        message: str,
        column: str | None = None,
        **details: Any,
    ) -> None:
        self.issues.append(
            ValidationIssue(
                code=code,
                message=message,
                severity="error",
                column=column,
                details=details,
            )
        )

    def add_warning(
        self,
        code: str,
        message: str,
        column: str | None = None,
        **details: Any,
    ) -> None:
        self.issues.append(
            ValidationIssue(
                code=code,
                message=message,
                severity="warning",
                column=column,
                details=details,
            )
        )

    def raise_if_invalid(self) -> None:
        """Detiene el pipeline cuando existen errores bloqueantes."""
        if self.is_valid:
            return

        messages = [
            f"[{issue.code}] {issue.message}"
            for issue in self.errors
        ]

        raise SchemaValidationError(
            f"Validación fallida en etapa '{self.stage}': "
            + " | ".join(messages)
        )


class PipelineError(RuntimeError):
    """Error base controlado del pipeline."""


class ConfigurationError(PipelineError):
    """Configuración incompleta o inválida."""


class SourceExtractionError(PipelineError):
    """Error al obtener una fuente de datos."""


class SchemaValidationError(PipelineError):
    """La estructura de los datos no cumple el contrato."""


class DataIntegrationError(PipelineError):
    """Las fuentes no pueden integrarse de manera válida."""


class DataQualityError(PipelineError):
    """El dataset limpio no cumple los requisitos de calidad."""


class ModelTrainingError(PipelineError):
    """Error controlado durante el entrenamiento."""


class ArtifactValidationError(PipelineError):
    """Los artefactos generados están incompletos o son inválidos."""


class ArtifactPublicationError(PipelineError):
    """No fue posible publicar una ejecución completa."""