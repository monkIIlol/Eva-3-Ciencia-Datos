# Pipeline ETL y preparación del dataset analítico

## Objetivo

El objetivo del pipeline es integrar datos de usuarios, validar su calidad y preparar un dataset analítico listo para análisis, segmentación y modelos de Machine Learning.

Esta etapa permite que el proyecto no dependa directamente de datos crudos, sino de un dataset preparado, documentado y validado.

## Flujo general del pipeline

```text
usuarios_streaming.csv + perfil_usuarios
↓
etl/extract.py
↓
data/data_consolidada.csv
↓
etl/prepare_dataset.py
↓
data/dataset_modelo.csv
data/kpis_negocio.csv
data/reporte_calidad.json
↓
modelos ML + API + dashboard
```

## Fuentes de datos

El proyecto integra dos fuentes principales:

| Fuente | Descripción |
|---|---|
| `data/usuarios_streaming.csv` | Datos de consumo y comportamiento de usuarios en la plataforma. |
| `perfil_usuarios` | Datos de perfil de usuarios provenientes de PostgreSQL o archivo de respaldo local. |

La integración se realiza mediante la columna `id_cliente`.

## Extracción e integración

El archivo `etl/extract.py` se encarga de integrar las fuentes de datos.

La salida principal de esta etapa es:

```text
data/data_consolidada.csv
```

Este archivo contiene los datos unidos antes de aplicar la preparación analítica.

## Validación previa

El proyecto también cuenta con validaciones en:

```text
etl/validate.py
```

Estas validaciones revisan aspectos como:

- columnas esperadas;
- valores nulos;
- duplicados por `id_cliente`;
- tipos numéricos;
- coincidencia de identificadores entre fuentes;
- dataset integrado final.

## Preparación del dataset analítico

El archivo principal de esta etapa es:

```text
etl/prepare_dataset.py
```

Este módulo toma `data/data_consolidada.csv` y genera un dataset preparado para análisis y modelos.

La salida principal es:

```text
data/dataset_modelo.csv
```

Además genera:

```text
data/kpis_negocio.csv
data/reporte_calidad.json
```

## Limpieza aplicada

La etapa de preparación aplica reglas de limpieza para mejorar la calidad del dataset.

### Tratamiento de `id_cliente`

`id_cliente` se considera un identificador único.

Por eso:

- no se imputa con mediana;
- no se inventa estadísticamente;
- si viene nulo, el registro se elimina;
- después de la limpieza se valida que no queden duplicados.

Esta decisión evita crear clientes artificiales o duplicar IDs existentes.

### Tratamiento de nulos

Las columnas numéricas, excepto `id_cliente`, se convierten a formato numérico.

Si aparecen nulos, se imputan con la mediana.

Se usa mediana porque es menos sensible a valores extremos que el promedio.

### Tratamiento de infinitos

Los valores infinitos se reemplazan antes de imputar.

Esto evita errores por divisiones inválidas o datos mal formados.

### Rangos de negocio

Se aplican reglas para evitar valores imposibles:

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

Se aplica el método IQR para detectar valores extremos.

```text
IQR = Q3 - Q1
Límite inferior = Q1 - 1.5 * IQR
Límite superior = Q3 + 1.5 * IQR
```

En vez de eliminar registros, se usa winsorización mediante `clip`.

Esto permite conservar la cantidad de usuarios, pero reduce el impacto de valores extremos.

## Variables derivadas creadas

El proceso de preparación genera variables nuevas para enriquecer el análisis.

| Variable | Descripción |
|---|---|
| `sesiones_mes_estimadas` | Estimación mensual de sesiones a partir de `sesiones_semana * 4`. |
| `contenidos_por_sesion` | Promedio de contenidos vistos por sesión mensual estimada. |
| `gasto_por_hora` | Relación entre gasto mensual y horas de consumo. |
| `minutos_totales_estimados` | Conversión de horas mensuales a minutos. |
| `soporte_por_dispositivo` | Interacciones de soporte por dispositivo registrado. |
| `generos_por_contenido` | Relación entre géneros consumidos y contenidos vistos. |
| `engagement_score` | Indicador relativo de compromiso del usuario. |
| `nivel_engagement` | Clasificación del usuario en bajo, medio o alto engagement. |
| `cliente_antiguo` | Indicador de clientes con al menos 36 meses de antigüedad. |
| `uso_promociones_alto` | Indicador de usuarios con uso de promociones igual o superior a 50%. |
| `valor_cliente` | Segmentación descriptiva: alto valor, valor medio o valor en riesgo. |

## Corrección conceptual importante

La variable `contenidos_por_sesion` no se calcula directamente como:

```text
cantidad_contenidos_vistos / sesiones_semana
```

porque una variable es mensual y la otra semanal.

Por eso primero se calcula:

```text
sesiones_mes_estimadas = sesiones_semana * 4
```

y luego:

```text
contenidos_por_sesion = cantidad_contenidos_vistos / sesiones_mes_estimadas
```

Esto hace que la interpretación sea más coherente.

## KPIs de negocio generados

El archivo:

```text
data/kpis_negocio.csv
```

resume indicadores agrupados por `nivel_engagement` y `valor_cliente`.

Incluye:

- cantidad de usuarios;
- gasto promedio;
- consumo promedio en horas;
- finalización promedio;
- uso promedio de promociones;
- antigüedad promedio;
- porcentaje de usuarios.

Estos KPIs permiten interpretar el comportamiento de los clientes desde una mirada de negocio.

## Reporte de calidad

El archivo:

```text
data/reporte_calidad.json
```

registra evidencia automática del proceso de preparación.

Incluye:

- filas y columnas originales;
- filas y columnas finales;
- IDs nulos eliminados;
- duplicados eliminados;
- nulos antes y después de limpieza;
- outliers detectados mediante IQR;
- memoria utilizada;
- variables derivadas creadas;
- nota metodológica.

## Optimización de memoria

El dataset preparado aplica reducción de tipos numéricos mediante `downcast`.

Esto ayuda a reducir el consumo de memoria y mejora la escalabilidad del pipeline para datasets más grandes.

## Pruebas automatizadas

Las pruebas se encuentran en:

```text
tests/test_prepare_dataset.py
```

Se validan casos normales y casos negativos.

Las pruebas comprueban que:

- se generen los archivos de salida;
- existan las variables derivadas;
- no existan nulos en variables críticas;
- no existan infinitos;
- `engagement_score` esté entre 0 y 1;
- las categorías de `nivel_engagement` y `valor_cliente` sean válidas;
- los KPIs tengan contenido;
- el reporte de calidad tenga información relevante.

## Pruebas negativas agregadas

También se agregaron pruebas con datos problemáticos artificiales.

| Caso probado | Objetivo |
|---|---|
| `id_cliente` nulo | Verificar que se elimine y no se impute. |
| `id_cliente` duplicado | Verificar que se eliminen duplicados. |
| Columna faltante | Verificar que se lance error. |
| Valores negativos | Verificar corrección de rangos. |
| Porcentajes fuera de rango | Verificar límites válidos. |
| Strings en columnas numéricas | Verificar conversión e imputación. |
| Valores infinitos | Verificar reemplazo correcto. |
| `contenidos_por_sesion` | Verificar que use sesiones mensuales estimadas. |

## Comandos principales

Ejecutar preparación del dataset:

```bash
py -3.11 etl/prepare_dataset.py
```

Ejecutar pruebas de preparación:

```bash
py -3.11 -m unittest tests/test_prepare_dataset.py
```

Ejecutar toda la suite de pruebas:

```bash
py -3.11 -m unittest discover tests
```

Resultado esperado:

```text
Ran 25 tests
OK
```

## Valor para el proyecto

Esta etapa fortalece el proyecto porque:

- transforma datos consolidados en un dataset analítico;
- evita trabajar directamente con datos crudos;
- aplica limpieza robusta;
- controla errores comunes;
- genera variables útiles para análisis y modelos;
- crea KPIs de negocio;
- deja evidencia de calidad de datos;
- incorpora pruebas automatizadas y casos negativos;
- prepara el camino para que modelos, API y dashboard usen un dataset más confiable.

## Limitación actual

Actualmente `dataset_modelo.csv` queda preparado para ser utilizado por los modelos.

La integración final debe hacer que:

```text
model/train.py
model/train_supervisado.py
API
dashboard
Docker
CI
```

usen o muestren los resultados generados por esta etapa.

Esta integración debe realizarse en una rama grupal para evitar conflictos con los cambios de otros integrantes.

## Documentos relacionados

| Documento | Descripción |
|---|---|
| `docs/diccionario_datos.md` | Explica las variables originales y derivadas. |
| `docs/reporte_calidad_datos.md` | Explica reglas de limpieza, outliers, pruebas y decisiones metodológicas. |
| `data/reporte_calidad.json` | Evidencia automática generada por el script. |