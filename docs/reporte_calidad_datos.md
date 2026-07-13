# Reporte de calidad de datos

## Objetivo

Este documento resume las reglas de limpieza, validación y transformación aplicadas en la etapa de preparación del dataset analítico.

El proceso se implementa en:

`etl/prepare_dataset.py`

Y genera los siguientes archivos:

| Archivo | Descripción |
|---|---|
| `data/dataset_modelo.csv` | Dataset final preparado para análisis y modelos. |
| `data/kpis_negocio.csv` | KPIs agrupados por nivel de engagement y valor de cliente. |
| `data/reporte_calidad.json` | Reporte automático con nulos, duplicados, outliers, memoria y variables derivadas. |

## Flujo aplicado

```text
data/data_consolidada.csv
↓
etl/prepare_dataset.py
↓
data/dataset_modelo.csv
data/kpis_negocio.csv
data/reporte_calidad.json
```

## Tratamiento de identificadores

La columna `id_cliente` se trata como identificador único.

Por esta razón:

- no se imputa con mediana;
- no se inventa un valor estadístico;
- si viene nulo, el registro se elimina;
- después de la limpieza se valida que no existan duplicados.

Esta decisión evita crear clientes artificiales o duplicar identificadores existentes.

## Tratamiento de nulos

Las variables numéricas, excepto `id_cliente`, se convierten a formato numérico.

Si aparecen valores nulos, se imputan con la mediana.

Se usa mediana porque es menos sensible a valores extremos que el promedio.

Ejemplo:

```text
Si una columna tiene valores muy altos o muy bajos,
la media puede distorsionarse.
La mediana representa mejor el centro de los datos.
```

## Tratamiento de infinitos

Los valores infinitos (`inf` o `-inf`) se reemplazan antes de imputar.

Esto es importante porque pueden aparecer por errores de cálculo o divisiones inválidas.

## Reglas de rangos de negocio

Se aplican límites para evitar valores imposibles o inconsistentes.

| Variable | Regla aplicada |
|---|---|
| `sesiones_semana` | mínimo 1 |
| `horas_consumo_mensual` | mínimo 0.1 |
| `gasto_mensual` | mínimo 0 |
| `cantidad_contenidos_vistos` | mínimo 0 |
| `tiempo_promedio_sesion_min` | mínimo 0 |
| `cantidad_generos_consumidos` | mínimo 0 |
| `antiguedad_cliente_meses` | mínimo 0 |
| `edad` | entre 13 y 100 |
| `dispositivos_registrados` | mínimo 1 |
| `cantidad_perfiles_creados` | mínimo 1 |
| `interacciones_mensuales_soporte` | mínimo 0 |
| `distancia_promedio_red_km` | mínimo 0 |
| `porcentaje_finalizacion` | entre 0 y 100 |
| `porcentaje_uso_promociones` | entre 0 y 1 |
| `porcentaje_uso_app_movil` | entre 0 y 1 |

## Tratamiento de outliers

Se aplica el método IQR para tratar valores extremos.

El IQR permite detectar valores atípicos usando el rango intercuartílico:

```text
IQR = Q3 - Q1
Límite inferior = Q1 - 1.5 * IQR
Límite superior = Q3 + 1.5 * IQR
```

En vez de eliminar registros, se usa winsorización con `clip`.

Esto significa que los valores extremos se ajustan al límite permitido, conservando la fila del cliente.

Esta decisión es útil porque el dataset tiene solo 300 registros y eliminar demasiadas filas podría reducir la información disponible para los modelos.

## Variables derivadas creadas

Se crearon variables nuevas para enriquecer el análisis:

| Variable | Justificación |
|---|---|
| `sesiones_mes_estimadas` | Permite comparar sesiones con variables mensuales. |
| `contenidos_por_sesion` | Mide intensidad de consumo por sesión mensual estimada. |
| `gasto_por_hora` | Relaciona gasto con consumo. |
| `minutos_totales_estimados` | Facilita interpretación del consumo en minutos. |
| `soporte_por_dispositivo` | Mide carga de soporte relativa a dispositivos. |
| `generos_por_contenido` | Representa diversidad de consumo. |
| `engagement_score` | Resume compromiso del usuario. |
| `nivel_engagement` | Clasifica usuarios en bajo, medio y alto engagement. |
| `cliente_antiguo` | Identifica clientes con antigüedad relevante. |
| `uso_promociones_alto` | Marca usuarios sensibles a promociones. |
| `valor_cliente` | Segmenta usuarios por valor descriptivo. |

## Corrección conceptual aplicada

La variable `contenidos_por_sesion` se calcula usando sesiones mensuales estimadas:

```text
sesiones_mes_estimadas = sesiones_semana * 4
contenidos_por_sesion = cantidad_contenidos_vistos / sesiones_mes_estimadas
```

Esto evita mezclar directamente una variable mensual con una variable semanal.

## Optimización de memoria

El dataset se optimiza usando `downcast` en columnas numéricas.

Esto reduce el uso de memoria y mejora la escalabilidad del pipeline para datasets más grandes.

## Pruebas automatizadas

Las pruebas se encuentran en:

`tests/test_prepare_dataset.py`

Se validan casos normales y casos problemáticos.

## Pruebas negativas agregadas

Se incorporaron pruebas con datos artificialmente incorrectos para comprobar robustez:

| Caso probado | Objetivo |
|---|---|
| `id_cliente` nulo | Verificar que se elimine y no se impute. |
| `id_cliente` duplicado | Verificar que se eliminen duplicados. |
| Columna faltante | Verificar que el sistema lance error. |
| Valores negativos | Verificar corrección de rangos. |
| Porcentajes fuera de rango | Verificar límites válidos. |
| Strings en columnas numéricas | Verificar conversión e imputación. |
| Valores infinitos | Verificar reemplazo de infinitos. |
| `contenidos_por_sesion` | Verificar que use sesiones mensuales estimadas. |

## Resultado de pruebas

La suite completa de pruebas fue ejecutada correctamente con:

```bash
py -3.11 -m unittest discover tests
```

Resultado esperado:

```text
Ran 25 tests
OK
```

## Importancia para el proyecto

Esta etapa permite que el proyecto no dependa directamente de datos crudos.

El flujo mejora porque:

- controla errores antes del modelado;
- genera variables útiles para análisis;
- produce KPIs de negocio;
- deja evidencia de calidad de datos;
- permite defender técnicamente las decisiones de limpieza;
- prepara el camino para que los modelos usen `dataset_modelo.csv`.

## Limitación actual

Actualmente `dataset_modelo.csv` todavía debe integrarse completamente con los modelos, API y dashboard.

La integración final debería hacer que:

```text
model/train.py
model/train_supervisado.py
API
dashboard
Docker
CI
```

usen o muestren los resultados generados por esta etapa.

Esto se debe realizar en una rama de integración grupal para evitar conflictos con los cambios de otros integrantes.

## Frase para defensa

Mi aporte fue construir una etapa de preparación del dataset analítico. Esta etapa toma los datos consolidados, aplica limpieza robusta, controla identificadores, nulos, infinitos, rangos y outliers, crea variables derivadas, genera KPIs de negocio y produce un reporte de calidad. Además, agregué pruebas automatizadas con casos normales y negativos para demostrar que el proceso responde correctamente frente a datos problemáticos.