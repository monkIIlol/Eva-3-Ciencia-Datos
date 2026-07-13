# Diseño del pipeline integrado

## Objetivo

Este documento describe el diseño del pipeline integrado del proyecto de segmentación y predicción de usuarios de una plataforma de streaming.

El objetivo del pipeline es ejecutar de forma ordenada y reproducible todas las etapas principales del sistema:

1. extracción de fuentes;
2. validación de datos;
3. integración de fuentes;
4. preparación del dataset;
5. entrenamiento de KMeans;
6. entrenamiento de modelos supervisados;
7. validación de productos;
8. publicación de artefactos.

El archivo principal del pipeline es:

`pipeline/run.py`

## Flujo general

```text
Fuentes de datos
↓
Extracción
↓
Validación
↓
Integración
↓
Preparación del dataset
↓
Entrenamiento KMeans
↓
Entrenamiento supervisado
↓
Validación final
↓
Publicación de artefactos
```

## Etapas del pipeline

| Etapa | Archivo principal | Descripción |
|---|---|---|
| Extracción | `etl/extract.py` | Obtiene los datos de consumo y perfil de usuarios. |
| Validación | `etl/validate.py` | Verifica columnas, tipos, nulos, duplicados y compatibilidad entre fuentes. |
| Integración | `etl/integrate.py` | Une las fuentes mediante `id_cliente` y valida la relación entre registros. |
| Preparación | `etl/prepare_dataset.py` | Limpia datos, genera dataset base limpio, dataset analítico, KPIs y reporte de calidad. |
| KMeans | `model/train.py` | Entrena el modelo de segmentación no supervisada. |
| Supervisado | `model/train_supervisado.py` | Entrena modelos de regresión y clasificación. |
| Validación final | `pipeline/run.py` | Comprueba que las salidas sean compatibles antes de publicar artefactos. |
| Publicación | `pipeline/run.py` | Guarda datasets, modelos, métricas y manifiesto final. |

## Datasets generados

El pipeline genera distintos datasets con propósitos separados.

| Archivo | Propósito |
|---|---|
| `data/data_consolidada.csv` | Dataset integrado desde las fuentes originales. |
| `data/dataset_base_limpio.csv` | Dataset limpio con las variables base oficiales. Es la entrada de los modelos. |
| `data/dataset_analitico.csv` | Dataset enriquecido con variables derivadas para análisis, KPIs y visualización. |
| `data/dataset_modelo.csv` | Archivo conservado por compatibilidad con versiones anteriores del proyecto. |
| `data/kpis_negocio.csv` | Indicadores agrupados por nivel de engagement y valor de cliente. |
| `data/reporte_calidad.json` | Evidencia automática de limpieza, nulos, rangos, outliers y variables derivadas. |

## Separación entre dataset base y dataset analítico

Una decisión importante del diseño fue separar `dataset_base_limpio.csv` de `dataset_analitico.csv`.

El `dataset_base_limpio.csv` contiene las variables originales limpias y validadas.  
Este dataset se utiliza para entrenar KMeans, regresión y clasificación.

El `dataset_analitico.csv` contiene variables derivadas como:

- `contenidos_por_sesion`;
- `gasto_por_hora`;
- `engagement_score`;
- `nivel_engagement`;
- `valor_cliente`.

Estas variables son útiles para análisis y negocio, pero no se incorporan automáticamente a los modelos para evitar fuga de datos.

Por ejemplo, si el objetivo de regresión es predecir `gasto_mensual`, no corresponde usar `gasto_por_hora`, porque esa variable contiene información directa del gasto.

## Preparación del dataset

La preparación se realiza en:

`etl/prepare_dataset.py`

Esta etapa aplica:

- validación de columnas requeridas;
- control estricto de `id_cliente`;
- conversión de variables numéricas;
- reemplazo de infinitos;
- detección de valores fuera de rango;
- imputación con mediana en variables analíticas;
- diagnóstico de outliers mediante IQR;
- optimización de tipos de datos;
- generación de variables derivadas;
- generación de KPIs;
- generación de reporte de calidad.

## Tratamiento de `id_cliente`

`id_cliente` se trata como identificador único.

Por eso:

- no se imputa;
- no se corrige con estadísticas;
- no puede ser nulo;
- no puede ser no numérico;
- no puede ser duplicado;
- no puede ser menor o igual a cero.

Si el identificador no cumple estas reglas, el pipeline detiene la ejecución porque no es seguro inventar o modificar IDs de clientes.

## Variables oficiales de modelado

Las variables utilizadas por cada modelo están centralizadas en:

`config/features.py`

Esto evita que cada archivo elija columnas distintas.

### KMeans

KMeans utiliza las variables definidas en:

`KMEANS_FEATURES`

### Regresión

La regresión predice:

`gasto_mensual`

y utiliza las variables definidas en:

`REGRESSION_FEATURES`

### Clasificación

La clasificación predice:

`riesgo_bajo_compromiso`

y utiliza las variables definidas en:

`CLASSIFICATION_FEATURES`

Las variables que construyen directamente la etiqueta de clasificación quedan separadas para evitar fuga de datos.

## Validación antes de publicar

El pipeline no guarda los resultados finales inmediatamente.

Primero valida que:

- la cantidad de usuarios se conserve entre integración, preparación y modelos;
- no existan `id_cliente` duplicados;
- KMeans haya sido entrenado con las variables oficiales;
- los centroides usen las columnas correctas;
- los modelos supervisados respeten las variables configuradas;
- las métricas sean numéricas y válidas;
- las métricas supervisadas correspondan al dataset base limpio.

Solo si estas validaciones pasan, el pipeline publica los artefactos.

## Manifiesto del pipeline

Al finalizar correctamente, se genera:

`models/pipeline_manifest.json`

Este archivo funciona como evidencia de ejecución completa.

Incluye información como:

- estado de la ejecución;
- fecha y duración;
- validación;
- integración;
- preparación;
- métricas de KMeans;
- métricas de regresión;
- métricas de clasificación;
- variables usadas por cada modelo;
- estrategia de publicación.

## Docker Compose

El proyecto incorpora un servicio específico para el pipeline.

En `docker-compose.yml`, el flujo queda separado en servicios:

```text
postgres
↓
pipeline
↓
api
↓
dashboard
```

Esto significa que:

- PostgreSQL debe estar saludable antes de ejecutar el pipeline;
- la API espera que el pipeline termine correctamente;
- el dashboard espera que la API esté saludable.

Esta estructura evita que la API se levante sin modelos o artefactos generados.

## Pruebas del pipeline

Las pruebas de integración se encuentran en:

`tests/test_pipeline.py`

Estas pruebas verifican que:

- el pipeline conserve los usuarios;
- la validación e integración sean correctas;
- los modelos supervisados consuman `dataset_base_limpio.csv`;
- KMeans sea compatible con inferencia desde la API;
- el manifiesto indique ejecución completa;
- las columnas provenientes desde SQLAlchemy se normalicen correctamente.

## Comandos útiles

Ejecutar pipeline completo:

```bash
py -3.11 -m pipeline.run
```

Ejecutar pruebas:

```bash
py -3.11 -m unittest discover tests
```

Ejecutar Docker Compose:

```bash
docker compose up --build
```

## Valor del diseño integrado

El diseño integrado mejora el proyecto porque:

- evita entrenar modelos directamente sobre datos crudos;
- centraliza las variables oficiales de modelado;
- separa dataset limpio de dataset analítico;
- reduce el riesgo de fuga de datos;
- valida los productos antes de publicarlos;
- genera un manifiesto de ejecución;
- permite levantar el sistema completo con Docker Compose;
- entrega una historia técnica coherente para la defensa del EFT.

## Relación con el aporte de preparación de datos

La preparación del dataset es una etapa central dentro del pipeline integrado.

Antes, el proyecto podía entrenar modelos directamente desde el dataset consolidado.  
Con el nuevo diseño, el flujo incorpora una etapa intermedia de calidad y preparación de datos.

Esta etapa genera:

- un dataset base limpio para modelado;
- un dataset analítico enriquecido;
- KPIs de negocio;
- un reporte de calidad;
- evidencia de control de nulos, duplicados, rangos y outliers.

Esto permite defender que los modelos no se entrenan directamente sobre datos crudos, sino sobre una salida preparada y validada.

## Limitaciones y mejoras futuras

Aunque el pipeline integrado mejora la reproducibilidad, todavía se pueden realizar mejoras:

- ampliar las pruebas de datos problemáticos;
- agregar monitoreo de tiempos por etapa;
- publicar imágenes Docker en un registro externo;
- automatizar despliegue continuo real;
- exponer más KPIs desde la API;
- visualizar el reporte de calidad directamente en el dashboard.

Estas mejoras no invalidan el pipeline actual, pero representan oportunidades para avanzar hacia un flujo más productivo y profesional.