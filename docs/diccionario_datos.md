# Diccionario de datos - Pipeline analítico

## Objetivo

Este documento describe las variables principales utilizadas en el pipeline de datos del proyecto.

El proceso de preparación genera dos salidas principales:

| Archivo | Uso |
|---|---|
| `data/dataset_base_limpio.csv` | Dataset limpio con variables base oficiales. Se utiliza para entrenar KMeans, regresión y clasificación. |
| `data/dataset_analitico.csv` | Dataset enriquecido con variables derivadas para análisis, KPIs y visualización. |
| `data/dataset_modelo.csv` | Archivo conservado por compatibilidad con versiones anteriores del proyecto. |

## Variables originales

Estas variables provienen de las fuentes iniciales y forman parte de la base del proyecto.

| Variable | Descripción |
|---|---|
| `id_cliente` | Identificador único del cliente. No se imputa ni se modifica estadísticamente. |
| `horas_consumo_mensual` | Cantidad de horas que el usuario consume contenido durante el mes. |
| `gasto_mensual` | Gasto mensual del cliente en la plataforma. |
| `cantidad_contenidos_vistos` | Cantidad de contenidos vistos por el usuario durante el mes. |
| `sesiones_semana` | Número de sesiones semanales del usuario. |
| `porcentaje_finalizacion` | Porcentaje promedio de finalización de contenidos. |
| `tiempo_promedio_sesion_min` | Duración promedio de una sesión en minutos. |
| `cantidad_generos_consumidos` | Cantidad de géneros distintos consumidos por el usuario. |
| `porcentaje_uso_promociones` | Proporción de uso de promociones por parte del cliente. |
| `antiguedad_cliente_meses` | Antigüedad del cliente medida en meses. |
| `edad` | Edad del cliente. |
| `dispositivos_registrados` | Cantidad de dispositivos asociados a la cuenta. |
| `porcentaje_uso_app_movil` | Proporción de uso desde aplicación móvil. |
| `cantidad_perfiles_creados` | Cantidad de perfiles creados dentro de la cuenta. |
| `interacciones_mensuales_soporte` | Cantidad de interacciones mensuales con soporte. |
| `distancia_promedio_red_km` | Distancia promedio estimada a la red o nodo de servicio. |

## Variables derivadas del dataset analítico

Estas variables se generan en `etl/prepare_dataset.py` para enriquecer el análisis de negocio.

| Variable | Descripción |
|---|---|
| `contenidos_por_sesion` | Promedio de contenidos vistos por sesión mensual estimada. |
| `gasto_por_hora` | Relación entre gasto mensual y horas de consumo. |
| `minutos_totales_estimados` | Conversión de horas de consumo mensual a minutos. |
| `soporte_por_dispositivo` | Relación entre interacciones de soporte y dispositivos registrados. |
| `generos_por_contenido` | Relación entre géneros consumidos y cantidad de contenidos vistos. |
| `engagement_score` | Indicador relativo de compromiso del usuario, construido con consumo, sesiones, finalización y antigüedad. |
| `nivel_engagement` | Clasificación del usuario según su engagement: bajo, medio o alto. |
| `cliente_antiguo` | Indicador binario para clientes con antigüedad relevante. |
| `uso_promociones_alto` | Indicador binario para clientes con alto uso de promociones. |
| `valor_cliente` | Segmentación descriptiva del cliente: alto valor, valor medio o valor en riesgo. |

## Cálculo intermedio

Para calcular `contenidos_por_sesion`, se utiliza una estimación mensual de sesiones:

`sesiones_mes_estimadas = sesiones_semana * 4`

Luego se calcula:

`contenidos_por_sesion = cantidad_contenidos_vistos / sesiones_mes_estimadas`

Esta estimación evita mezclar directamente una variable mensual con una variable semanal.

`sesiones_mes_estimadas` funciona como cálculo intermedio y no necesariamente queda persistida como variable final del dataset.

## Variables oficiales de modelado

Las variables usadas por los modelos se centralizan en:

`config/features.py`

Esto evita que cada modelo seleccione columnas de forma independiente.

| Modelo | Variables configuradas |
|---|---|
| KMeans | `KMEANS_FEATURES` |
| Regresión | `REGRESSION_FEATURES` |
| Clasificación | `CLASSIFICATION_FEATURES` |

Esta separación ayuda a controlar qué columnas entran a cada modelo y reduce el riesgo de fuga de datos.

## Reglas de limpieza aplicadas

Durante la preparación del dataset se aplican controles de calidad.

| Regla | Descripción |
|---|---|
| Tratamiento de `id_cliente` | No se imputa. Si es nulo, no numérico, duplicado o menor/igual a cero, el pipeline debe detenerse. |
| Duplicados | Se valida que no existan duplicados por `id_cliente`. |
| Tipos numéricos | Las columnas esperadas se convierten a formato numérico cuando corresponde. |
| Nulos | Los nulos en variables analíticas pueden imputarse con mediana cuando es metodológicamente válido. |
| Infinitos | Los valores infinitos se reemplazan antes de continuar el procesamiento. |
| Rangos de negocio | Se controlan valores imposibles, como cantidades negativas, porcentajes fuera de rango o edades inválidas. |
| Outliers | Se realiza diagnóstico mediante IQR para identificar valores extremos. |
| Optimización | Se reducen tipos de datos cuando corresponde para mejorar uso de memoria. |

## Archivos generados

| Archivo | Descripción |
|---|---|
| `data/dataset_base_limpio.csv` | Dataset base limpio utilizado por los modelos. |
| `data/dataset_analitico.csv` | Dataset enriquecido para análisis, KPIs y visualización. |
| `data/dataset_modelo.csv` | Archivo mantenido por compatibilidad con versiones anteriores. |
| `data/kpis_negocio.csv` | KPIs agrupados por nivel de engagement y valor de cliente. |
| `data/reporte_calidad.json` | Reporte automático con evidencia de calidad, nulos, duplicados, outliers, memoria y variables derivadas. |

## Nota metodológica

Las variables derivadas se construyen para apoyar análisis, segmentación y toma de decisiones.

Algunas variables deben usarse con cuidado en modelos supervisados. Por ejemplo, si un modelo intenta predecir `gasto_mensual`, no debería usar `gasto_por_hora` como variable predictora, porque esa variable contiene información directa del gasto y podría generar fuga de datos.

Además, `valor_cliente` es una clasificación descriptiva creada con reglas internas del proyecto. No debe interpretarse como una etiqueta real entregada por la empresa.

Por esta razón, el pipeline separa el `dataset_base_limpio.csv`, orientado a modelado, del `dataset_analitico.csv`, orientado a análisis, KPIs y visualización.