# Analítica y Segmentación de Usuarios de Streaming

Proyecto de Ciencia de Datos — SCY1101 Programación para la Ciencia de Datos

Solución de analítica de clientes para una plataforma de streaming. El proyecto integra datos de consumo y perfil de usuarios, aplica validación y preparación de datos, entrena modelos no supervisados y supervisados, expone resultados mediante una API REST y los presenta en un dashboard interactivo.

El repositorio fue desarrollado originalmente para la Evaluación 3 y actualmente constituye la base técnica para la Evaluación Final Transversal.

## Integrantes

* Verónica Cereceda
* Diego Torres
* Cristian Urcullú

## Contexto del problema

El negocio utiliza actualmente estrategias generales para todos sus usuarios, sin diferenciar sus comportamientos, niveles de consumo o características de perfil.

La solución busca apoyar la personalización de campañas, recomendaciones y estrategias de retención mediante tres componentes analíticos:

* segmentación de usuarios mediante KMeans;
* estimación del gasto mensual mediante modelos de regresión;
* identificación de perfiles de bajo compromiso mediante modelos de clasificación.

El detalle del caso de negocio y las fuentes de datos se encuentra en [`docs/contexto_negocio.md`](docs/contexto_negocio.md).

## Fuentes de datos

El proyecto utiliza dos fuentes principales:

1. `data/usuarios_streaming.csv`: información de consumo y comportamiento en la plataforma.
2. PostgreSQL: información de perfil de los usuarios, cargada desde `database/perfil_usuarios.csv`.

Ambas fuentes se integran mediante `id_cliente`.

## Arquitectura actual

```text
usuarios_streaming.csv ───────────────┐
                                      │
perfil_usuarios.csv → PostgreSQL ─────┤
                                      ▼
                              etl/extract.py
                                      │
                                      ▼
                         data/data_consolidada.csv
                                      │
              ┌───────────────────────┼────────────────────────┐
              │                       │                        │
              ▼                       ▼                        ▼
       etl/validate.py       etl/prepare_dataset.py       model/train.py
              │                       │                        │
              │                       ├─ dataset_modelo.csv    ├─ KMeans
              │                       ├─ kpis_negocio.csv      ├─ PCA
              │                       └─ reporte_calidad.json  └─ métricas
              │
              └───────────────────────┐
                                      │
                                      ▼
                         model/train_supervisado.py
                                      │
                       ┌──────────────┴──────────────┐
                       ▼                             ▼
                   Regresión                     Clasificación
                de gasto mensual             de bajo compromiso
                       │                             │
                       └──────────────┬──────────────┘
                                      ▼
                                 API FastAPI
                                      │
                                      ▼
                              Dashboard Streamlit
```

La validación, preparación y entrenamiento se encuentran implementados como módulos separados. La integración de todas las etapas dentro de un único orquestador end-to-end se encuentra en desarrollo.

El diagrama técnico se encuentra en [`docs/arquitectura.md`](docs/arquitectura.md).

## Componentes analíticos

### Segmentación no supervisada

El modelo KMeans agrupa usuarios con comportamientos similares utilizando variables de consumo y perfil.

La cantidad de clusters se evalúa mediante:

* método del codo;
* coeficiente Silhouette;
* interpretación de negocio;
* visualización con PCA.

### Regresión

Se comparan modelos para estimar el gasto mensual de un usuario:

* regresión lineal;
* Random Forest Regressor.

La evaluación considera:

* MAE;
* RMSE;
* R².

### Clasificación

Se comparan modelos para identificar perfiles de bajo compromiso:

* regresión logística;
* Random Forest Classifier.

La evaluación considera:

* accuracy;
* precision;
* recall;
* F1-score;
* ROC-AUC.

La variable de clasificación representa una etiqueta proxy de bajo compromiso construida a partir del comportamiento de los usuarios. No corresponde a una variable observada de abandono o cancelación real.

## Preparación de datos

El módulo `etl/prepare_dataset.py` realiza:

* eliminación de duplicados;
* conversión de columnas numéricas;
* tratamiento de valores nulos e infinitos;
* validación y corrección de algunos rangos;
* creación de variables derivadas;
* optimización de tipos de datos;
* generación de KPIs;
* generación de un reporte de calidad.

Artefactos generados:

```text
data/dataset_modelo.csv
data/kpis_negocio.csv
data/reporte_calidad.json
```

## Estructura del repositorio

```text
.
├── .github/
│   └── workflows/           # Integración continua con GitHub Actions
├── api/                     # API REST desarrollada con FastAPI
├── dashboards/              # Dashboard interactivo desarrollado con Streamlit
├── database/                # Inicialización y carga de PostgreSQL
├── data/                    # Fuentes, datasets procesados y resultados
├── docker/                  # Dockerfiles de la API y el dashboard
├── docs/                    # Documentación técnica y de negocio
├── etl/                     # Extracción, validación y preparación de datos
├── model/                   # Modelos no supervisados y supervisados
├── models/                  # Artefactos serializados de los modelos
├── tests/                   # Pruebas automatizadas
├── docker-compose.yml
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

## Requisitos

### Ejecución mediante Docker

* Docker Desktop
* Docker Compose

### Ejecución local

* Python 3.11
* pip

Python 3.11 es la versión de referencia utilizada por Docker y GitHub Actions.

## Cómo ejecutar el proyecto

### Opción A: Docker

Docker es la opción recomendada para levantar PostgreSQL, la API y el dashboard.

#### 1. Crear el archivo de variables de entorno

En Linux o macOS:

```bash
cp .env.example .env
```

En PowerShell:

```powershell
Copy-Item .env.example .env
```

También puede copiarse manualmente desde el explorador de archivos.

#### 2. Construir y levantar los servicios

```bash
docker compose up --build
```

Una vez levantados los contenedores:

* API: `http://localhost:8000`
* Swagger: `http://localhost:8000/docs`
* Dashboard: `http://localhost:8501`

Para detener los servicios:

```bash
docker compose down
```

Para eliminar además los volúmenes creados:

```bash
docker compose down -v
```

La guía detallada se encuentra en [`docs/guia_despliegue.md`](docs/guia_despliegue.md).

## Ejecución local para desarrollo y pruebas

### 1. Crear un entorno virtual

#### Windows

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### Linux o macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

La carpeta `.venv` es local y no debe subirse al repositorio.

### 2. Instalar dependencias

Actualizar pip:

```bash
python -m pip install --upgrade pip
```

Instalar las dependencias necesarias para desarrollo y pruebas:

```bash
python -m pip install -r requirements-dev.txt
```

`requirements-dev.txt` incluye las dependencias principales del proyecto y las dependencias adicionales utilizadas por las pruebas automatizadas.

## Validación de datos

Para ejecutar la validación de las fuentes:

```bash
python etl/validate.py
```

La validación comprueba:

* existencia de las columnas esperadas;
* presencia de valores nulos;
* duplicados en `id_cliente`;
* tipos de datos numéricos;
* coincidencia de usuarios entre las fuentes;
* consistencia del dataset integrado.

## Preparación del dataset

Para ejecutar la limpieza, transformación y generación de KPIs:

```bash
python etl/prepare_dataset.py
```

Este comando requiere que exista previamente:

```text
data/data_consolidada.csv
```

## Entrenamiento de modelos

### Modelo KMeans

```bash
python model/train.py
```

### Modelos supervisados

```bash
python model/train_supervisado.py
```

## Ejecución de pruebas

Para ejecutar todas las pruebas automatizadas:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

Las pruebas cubren actualmente:

* validación de fuentes;
* preparación del dataset;
* generación de KPIs;
* entrenamiento de modelos supervisados;
* predicciones;
* endpoints de la API.

También pueden ejecutarse archivos de prueba específicos:

```bash
python -m unittest tests/test_validacion_datos.py -v
```

```bash
python -m unittest tests/test_prepare_dataset.py -v
```

```bash
python -m unittest tests/test_modelo_supervisado.py -v
```

```bash
python -m unittest tests/test_api.py -v
```

## API REST

La API se encuentra implementada con FastAPI.

Principales endpoints:

```text
GET  /
GET  /dashboard-data
GET  /metricas-supervisado
POST /predict
POST /predict-gasto
POST /predict-riesgo
```

La documentación interactiva se encuentra disponible en:

```text
http://localhost:8000/docs
```

## Dashboard

El dashboard contiene vistas orientadas a diferentes usuarios:

* vista ejecutiva;
* vista técnica;
* vista operativa;
* modelos supervisados y predicciones.

Permite visualizar:

* distribución de clusters;
* características promedio de los segmentos;
* métricas del modelo KMeans;
* representación PCA;
* métricas de clasificación y regresión;
* predicciones para nuevos usuarios.

## Integración continua

El repositorio utiliza GitHub Actions para automatizar:

* instalación de dependencias;
* validación de datos;
* entrenamiento de modelos;
* ejecución de pruebas;
* construcción de imágenes Docker.

Las dependencias de desarrollo y testing se encuentran declaradas en:

```text
requirements-dev.txt
```

El workflow se encuentra en:

```text
.github/workflows/ci.yml
```

## Variables de entorno

La plantilla `.env.example` contiene las variables necesarias para la conexión con PostgreSQL:

```text
POSTGRES_USER
POSTGRES_PASSWORD
POSTGRES_DB
POSTGRES_HOST
POSTGRES_PORT
```

El archivo `.env` contiene la configuración local y no debe versionarse.

Antes de utilizar el proyecto en un ambiente distinto, se deben revisar también las configuraciones de conexión utilizadas por los módulos ETL y asegurar que no existan credenciales escritas directamente en el código.

## Flujo de trabajo con Git

Cada integrante desarrolla sus cambios en una rama independiente.

El flujo esperado es:

```text
main
  └── rama individual
          └── commit
                  └── push
                          └── Pull Request
                                  └── revisión
                                          └── merge
```

Los cambios deben incorporarse a `main` mediante Pull Request y revisión de al menos otro integrante.

Antes de crear un Pull Request se recomienda ejecutar:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

## Limitaciones actuales

* La variable de bajo compromiso es una etiqueta proxy y no representa churn real.
* La validación, preparación y entrenamiento todavía se ejecutan mediante comandos separados.
* Los modelos aún no consumen necesariamente todas las variables generadas por `prepare_dataset.py`.
* El pipeline de integración continua no reproduce completamente PostgreSQL como fuente de entrada.
* El dataset contiene solo 300 usuarios, por lo que los resultados deben interpretarse considerando el tamaño de la muestra.
* Los resultados representan apoyo para decisiones y no demuestran por sí solos una reducción real del abandono.

## Estado del proyecto

Actualmente el repositorio contiene los principales componentes técnicos exigidos:

* integración de fuentes;
* validación;
* preparación de datos;
* modelos supervisados y no supervisados;
* API;
* dashboard;
* Docker;
* pruebas;
* integración continua.

La siguiente etapa consiste en integrar estos componentes dentro de un único pipeline reproducible y coherente desde la extracción hasta la visualización.
