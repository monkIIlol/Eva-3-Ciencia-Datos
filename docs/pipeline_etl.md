# Pipeline ETL y preparación del dataset analítico

## Objetivo

El objetivo del pipeline es integrar datos de usuarios, validar su calidad, preparar datasets confiables y entrenar modelos de Machine Learning de forma ordenada y reproducible.

El pipeline permite que el proyecto no entrene modelos directamente sobre datos crudos, sino sobre una salida limpia, validada y controlada.

El archivo principal del flujo integrado es:

`pipeline/run.py`

## Flujo general del pipeline

```text
usuarios_streaming.csv + perfil_usuarios
↓
extracción de fuentes
↓
validación de datos
↓
integración por id_cliente
↓
preparación del dataset
↓
dataset_base_limpio.csv + dataset_analitico.csv
↓
KMeans + modelos supervisados
↓
API + dashboard
```

## Fuentes de datos

El proyecto integra dos fuentes principales:

| Fuente | Descripción |
|---|---|
| `data/usuarios_streaming.csv` | Datos de consumo y comportamiento de usuarios en la plataforma. |
| `perfil_usuarios` | Datos de perfil de usuarios provenientes de PostgreSQL o archivo de respaldo local. |

La integración se realiza mediante la columna `id_cliente`.

## Extracción, validación e integración

El proceso ETL contempla tres responsabilidades principales.

| Etapa | Archivo | Descripción |
|---|---|---|
| Extracción | `etl/extract.py` | Obtiene los datos desde las fuentes disponibles. |
| Validación | `etl/validate.py` | Revisa columnas esperadas, nulos, duplicados, tipos numéricos y compatibilidad entre fuentes. |
| Integración | `etl/integrate.py` | Une las fuentes mediante `id_cliente` y genera el dataset consolidado. |

La salida principal de esta etapa es:

```text
data/data_consolidada.csv
```

Este archivo contiene los datos integrados antes de la preparación analítica.

## Preparación del dataset

La preparación se realiza en:

```text
etl/prepare_dataset.py
```

Esta etapa toma el dataset integrado y genera salidas separadas según su propósito.

| Archivo | Propósito |
|---|---|
| `data/dataset_base_limpio.csv` | Dataset limpio con variables base oficiales. Es la entrada principal de los modelos. |
| `data/dataset_analitico.csv` | Dataset enriquecido con variables derivadas para análisis, KPIs y visualización. |
| `data/dataset_modelo.csv` | Archivo conservado por compatibilidad con versiones anteriores del proyecto. |
| `data/kpis_negocio.csv` | KPIs agrupados por nivel de engagement y valor de cliente. |
| `data/reporte_calidad.json` | Evidencia automática de calidad, nulos, rangos, outliers y variables derivadas. |

## Separación entre dataset base y dataset analítico

Una decisión importante del pipeline es separar el dataset limpio de modelado y el dataset enriquecido de análisis.

### `dataset_base_limpio.csv`

Contiene las variables originales limpias, validadas y listas para modelado.

Este dataset se usa como entrada para:

- KMeans;
- regresión;
- clasificación.

### `dataset_analitico.csv`

Contiene variables derivadas útiles para análisis de negocio, KPIs y visualización.

Ejemplos:

- `contenidos_por_sesion`;
- `gasto_por_hora`;
- `engagement_score`;
- `nivel_engagement`;
- `valor_cliente`.

Estas variables no se incorporan automáticamente a los modelos para evitar fuga de datos.

Por ejemplo, si el objetivo de regresión es predecir `gasto_mensual`, no corresponde usar `gasto_por_hora`, porque esa variable contiene información directa del gasto.

## Limpieza aplicada

La etapa de preparación aplica reglas de limpieza para mejorar la calidad del dataset y evitar errores posteriores en los modelos.

### Tratamiento de `id_cliente`

`id_cliente` se considera un identificador único y crítico.

Por eso:

- no se imputa;
- no se inventa estadísticamente;
- no puede ser nulo;
- no puede ser no numérico;
- no puede estar duplicado;
- no puede ser menor o igual a cero.

Si `id_cliente` no cumple estas reglas, el pipeline detiene la ejecución. Esta decisión evita crear clientes artificiales o alterar identificadores reales.

### Tratamiento de nulos

Las columnas numéricas se convierten a formato numérico cuando corresponde.

Para variables analíticas, los nulos pueden imputarse con mediana cuando es metodológicamente válido.

Se usa mediana porque es menos sensible a valores extremos que el promedio.

### Tratamiento de infinitos

Los valores infinitos se reemplazan antes de imputar o procesar.

Esto evita errores por divisiones inválidas o datos mal formados.

### Rangos de negocio

Se aplican reglas para evitar valores imposibles o inconsistentes.

| Variable | Regla general |
|---|---|
| `sesiones_semana` | valor positivo |
| `horas_consumo_mensual` | valor positivo |
| `gasto_mensual` | mínimo 0 |
| `cantidad_contenidos_vistos` | mínimo 0 |
| `tiempo_promedio_sesion_min` | mínimo 0 |
| `cantidad_generos_consumidos` | mínimo 0 |
| `antiguedad_cliente_meses` | mínimo 0 |
| `edad` | rango válido de edad |
| `dispositivos_registrados` | mínimo 1 |
| `cantidad_perfiles_creados` | mínimo 1 |
| `interacciones_mensuales_soporte` | mínimo 0 |
| `distancia_promedio_red_km` | mínimo 0 |
| `porcentaje_finalizacion` | rango porcentual válido |
| `porcentaje_uso_promociones` | rango entre 0 y 1 |
| `porcentaje_uso_app_movil` | rango entre 0 y 1 |

## Tratamiento de outliers

El pipeline utiliza diagnóstico de outliers mediante IQR.

```text
IQR = Q3 - Q1
Límite inferior = Q1 - 1.5 * IQR
Límite superior = Q3 + 1.5 * IQR
```

Este análisis permite detectar valores extremos y registrar evidencia en el reporte de calidad.

## Variables derivadas

El dataset analítico incorpora variables derivadas para enriquecer el análisis de comportamiento de usuarios.

| Variable | Descripción |
|---|---|
| `contenidos_por_sesion` | Promedio de contenidos vistos por sesión mensual estimada. |
| `gasto_por_hora` | Relación entre gasto mensual y horas de consumo. |
| `minutos_totales_estimados` | Conversión de horas mensuales a minutos. |
| `soporte_por_dispositivo` | Interacciones de soporte por dispositivo registrado. |
| `generos_por_contenido` | Relación entre géneros consumidos y contenidos vistos. |
| `engagement_score` | Indicador relativo de compromiso del usuario. |
| `nivel_engagement` | Clasificación del usuario en bajo, medio o alto engagement. |
| `cliente_antiguo` | Indicador de clientes con antigüedad relevante. |
| `uso_promociones_alto` | Indicador de usuarios con alto uso de promociones. |
| `valor_cliente` | Segmentación descriptiva: alto valor, valor medio o valor en riesgo. |

## Corrección conceptual de `contenidos_por_sesion`

La variable `contenidos_por_sesion` no se calcula directamente dividiendo por `sesiones_semana`, porque una variable es mensual y la otra semanal.

Primero se estima una cantidad mensual de sesiones:

```text
sesiones_mes_estimadas = sesiones_semana * 4
```

Luego se calcula:

```text
contenidos_por_sesion = cantidad_contenidos_vistos / sesiones_mes_estimadas
```

`sessions_mes_estimadas` funciona como cálculo intermedio para construir una variable más coherente.

## KPIs de negocio

El archivo:

```text
data/kpis_negocio.csv
```

resume indicadores agrupados por `nivel_engagement` y `valor_cliente`.

Incluye:

- cantidad de usuarios;
- gasto promedio;
- consumo promedio;
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

Puede incluir información como:

- filas y columnas originales;
- filas y columnas finales;
- validación de identificadores;
- duplicados detectados;
- nulos antes y después de limpieza;
- diagnóstico de outliers;
- memoria utilizada;
- variables derivadas creadas;
- notas metodológicas.

## Optimización de memoria

El dataset preparado aplica reducción de tipos numéricos cuando corresponde.

Esto ayuda a reducir el consumo de memoria y mejora la escalabilidad del pipeline para datasets más grandes.

## Variables oficiales de modelado

Las variables usadas por los modelos están centralizadas en:

```text
config/features.py
```

Esto evita que cada modelo seleccione columnas de forma independiente.

| Modelo | Configuración |
|---|---|
| KMeans | `KMEANS_FEATURES` |
| Regresión | `REGRESSION_FEATURES` |
| Clasificación | `CLASSIFICATION_FEATURES` |

Esta separación también ayuda a evitar fuga de datos.

## Modelos entrenados

El pipeline entrena modelos no supervisados y supervisados.

| Tipo | Archivo | Propósito |
|---|---|---|
| KMeans | `model/train.py` | Segmentación de usuarios. |
| Regresión | `model/train_supervisado.py` | Predicción de `gasto_mensual`. |
| Clasificación | `model/train_supervisado.py` | Predicción de `riesgo_bajo_compromiso`. |

## API y dashboard

Los artefactos generados por el pipeline son utilizados posteriormente por:

| Componente | Archivo |
|---|---|
| API | `api/main.py` |
| Dashboard | `dashboards/app.py` |

La API expone resultados de segmentación, métricas y predicciones.

El dashboard permite visualizar la información desde una perspectiva ejecutiva, técnica, operativa y predictiva.

## Docker Compose

El flujo integrado considera servicios separados:

```text
postgres
↓
pipeline
↓
api
↓
dashboard
```

Esta estructura permite que:

- PostgreSQL esté disponible antes de ejecutar el pipeline;
- la API espere los artefactos generados;
- el dashboard espere a que la API esté saludable.

## Pruebas automatizadas

Las pruebas se encuentran en la carpeta:

```text
tests/
```

Incluyen pruebas de:

- validación de datos;
- preparación del dataset;
- modelos supervisados;
- API;
- pipeline integrado.

La cantidad exacta de pruebas puede variar según la versión del repositorio, pero el resultado esperado es que la suite complete correctamente con `OK`.

## Comandos principales

Ejecutar pipeline completo:

```bash
py -3.11 -m pipeline.run
```

Ejecutar preparación del dataset:

```bash
py -3.11 etl/prepare_dataset.py
```

Ejecutar toda la suite de pruebas:

```bash
py -3.11 -m unittest discover tests
```

Ejecutar Docker Compose:

```bash
docker compose up --build
```

## Valor para el proyecto

Esta etapa fortalece el proyecto porque:

- evita entrenar modelos directamente sobre datos crudos;
- incorpora validación y preparación antes del modelado;
- separa dataset base limpio y dataset analítico;
- reduce el riesgo de fuga de datos;
- genera variables útiles para análisis;
- crea KPIs de negocio;
- deja evidencia de calidad de datos;
- centraliza variables oficiales de modelado;
- permite una ejecución reproducible mediante pipeline integrado;
- entrega una historia técnica coherente para la defensa del EFT.

## Limitaciones y mejoras futuras

Aunque el pipeline integrado mejora la solución, todavía se pueden realizar mejoras:

- ampliar pruebas negativas con más casos problemáticos;
- exponer más KPIs desde la API;
- mostrar el reporte de calidad directamente en el dashboard;
- agregar monitoreo de tiempos por etapa;
- publicar imágenes Docker en un registro externo;
- automatizar un despliegue continuo real.

Estas limitaciones no invalidan el flujo actual, sino que representan mejoras para una versión más productiva.

## Documentos relacionados

| Documento | Descripción |
|---|---|
| `docs/diccionario_datos.md` | Explica variables originales y derivadas. |
| `docs/reporte_calidad_datos.md` | Explica reglas de limpieza, outliers, pruebas y decisiones metodológicas. |
| `docs/diseno_pipeline_integrado.md` | Describe la arquitectura del pipeline integrado. |
| `data/reporte_calidad.json` | Evidencia automática generada por el pipeline. |