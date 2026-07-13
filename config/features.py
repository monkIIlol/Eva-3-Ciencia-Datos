from __future__ import annotations


ID_COLUMN = "id_cliente"


KMEANS_FEATURES = [
    "horas_consumo_mensual",
    "gasto_mensual",
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


REGRESSION_TARGET = "gasto_mensual"

REGRESSION_FEATURES = [
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


CLASSIFICATION_TARGET = "riesgo_bajo_compromiso"

CLASSIFICATION_FEATURES = [
    "horas_consumo_mensual",
    "cantidad_contenidos_vistos",
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


# Variables que construyen directamente la etiqueta de clasificación.
CLASSIFICATION_TARGET_SOURCE_COLUMNS = [
    "sesiones_semana",
    "porcentaje_finalizacion",
]


# Variables derivadas que no deben entrar automáticamente a los modelos.
DERIVED_BUSINESS_COLUMNS = [
    "contenidos_por_sesion",
    "gasto_por_hora",
    "minutos_totales_estimados",
    "soporte_por_dispositivo",
    "generos_por_contenido",
    "engagement_score",
    "nivel_engagement",
    "cliente_antiguo",
    "uso_promociones_alto",
    "valor_cliente",
]