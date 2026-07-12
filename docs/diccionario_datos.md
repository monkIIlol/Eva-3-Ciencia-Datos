# Diccionario de datos - Dataset analítico

## Objetivo

Este documento describe las variables principales utilizadas en el dataset preparado para modelado.

El archivo generado por el proceso de preparación es:

`data/dataset_modelo.csv`

## Variables originales

| Variable | Descripción |
|---|---|
| id_cliente | Identificador único del cliente. No se imputa, porque no corresponde inventar un ID estadísticamente. |
| horas_consumo_mensual | Cantidad de horas que el usuario consume contenido durante el mes. |
| gasto_mensual | Gasto mensual del cliente en la plataforma. |
| cantidad_contenidos_vistos | Cantidad de contenidos vistos por el usuario durante el mes. |
| sesiones_semana | Número de sesiones semanales del usuario. |
| porcentaje_finalizacion | Porcentaje promedio de finalización de contenidos. |
| tiempo_promedio_sesion_min | Duración promedio de una sesión en minutos. |
| cantidad_generos_consumidos | Cantidad de géneros distintos consumidos por el usuario. |
| porcentaje_uso_promociones | Proporción de uso de promociones por parte del cliente. |
| antiguedad_cliente_meses | Antigüedad del cliente medida en meses. |
| edad | Edad del cliente. |
| dispositivos_registrados | Cantidad de dispositivos asociados a la cuenta. |
| porcentaje_uso_app_movil | Proporción de uso desde aplicación móvil. |
| cantidad_perfiles_creados | Cantidad de perfiles creados dentro de la cuenta. |
| interacciones_mensuales_soporte | Cantidad de interacciones mensuales con soporte. |
| distancia_promedio_red_km | Distancia promedio estimada a la red o nodo de servicio. |

## Variables derivadas

| Variable | Descripción |
|---|---|
| sesiones_mes_estimadas | Estimación de sesiones mensuales. Se calcula como `sesiones_semana * 4`. |
| contenidos_por_sesion | Promedio de contenidos vistos por sesión mensual estimada. |
| gasto_por_hora | Relación entre gasto mensual y horas de consumo. |
| minutos_totales_estimados | Conversión de horas de consumo mensual a minutos. |
| soporte_por_dispositivo | Relación entre interacciones de soporte y dispositivos registrados. |
| generos_por_contenido | Relación entre géneros consumidos y cantidad de contenidos vistos. |
| engagement_score | Indicador relativo de compromiso del usuario, construido con consumo, sesiones, finalización y antigüedad. |
| nivel_engagement | Clasificación del usuario según su engagement: bajo, medio o alto. |
| cliente_antiguo | Indicador binario que vale 1 si el cliente tiene al menos 36 meses de antigüedad. |
| uso_promociones_alto | Indicador binario que vale 1 si el uso de promociones es igual o superior a 50%. |
| valor_cliente | Segmentación descriptiva del cliente: alto valor, valor medio o valor en riesgo. |

## Reglas de limpieza aplicadas

Durante la preparación del dataset se aplican las siguientes reglas:

| Regla | Descripción |
|---|---|
| Tratamiento de `id_cliente` | No se imputa. Si viene nulo, se elimina el registro. |
| Duplicados | Se eliminan duplicados por `id_cliente`. |
| Tipos numéricos | Las columnas esperadas se convierten a formato numérico. |
| Nulos | Los nulos numéricos se imputan con la mediana, excepto `id_cliente`. |
| Infinitos | Los valores infinitos se reemplazan antes de imputar. |
| Rangos de negocio | Se corrigen valores imposibles, como edad negativa, porcentajes fuera de rango o cantidades menores a cero. |
| Outliers | Se tratan valores extremos mediante IQR y winsorización. |
| Optimización | Se reducen tipos de datos con `downcast` para mejorar uso de memoria. |

## Archivos generados

| Archivo | Descripción |
|---|---|
| data/dataset_modelo.csv | Dataset final preparado para análisis y modelado. |
| data/kpis_negocio.csv | KPIs agrupados por nivel de engagement y valor de cliente. |
| data/reporte_calidad.json | Reporte de calidad con nulos, duplicados, outliers, memoria y variables derivadas. |

## Nota metodológica

Las variables derivadas se construyen para apoyar análisis, segmentación y futuras tareas de modelado.

Algunas variables deben usarse con cuidado en modelos supervisados. Por ejemplo, si un modelo intenta predecir `gasto_mensual`, no debería usar `gasto_por_hora` como variable predictora, porque esa variable contiene información directa del gasto y podría generar fuga de datos.

Además, `valor_cliente` es una clasificación descriptiva creada con reglas internas del proyecto. No debe interpretarse como una etiqueta real entregada por la empresa.