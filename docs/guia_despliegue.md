# Guía de Despliegue del Sistema

Este documento explica cómo ejecutar el proyecto de segmentación de usuarios de streaming utilizando Docker y también cómo probar sus componentes principales de forma local.

El sistema integra dos fuentes de datos: un archivo CSV con información de consumo de usuarios y una base de datos PostgreSQL con información de perfil. Luego se ejecuta un pipeline ETL, se entrena un modelo KMeans y los resultados se visualizan mediante una API y un dashboard interactivo.

## 1. Requisitos previos

Para ejecutar el proyecto se necesita tener instalado:

* Docker Desktop.
* Docker Compose.
* Git.
* Python 3.11, solo si se desea ejecutar validaciones o pruebas de forma local.

## 2. Servicios del proyecto

El archivo `docker-compose.yml` levanta los siguientes servicios:

| Servicio    | Descripción                                                      | Puerto |
| ----------- | ---------------------------------------------------------------- | ------ |
| `postgres`  | Base de datos PostgreSQL con los datos de perfil de usuarios     | 5432   |
| `api`       | Servicio FastAPI que ejecuta el pipeline y expone los resultados | 8000   |
| `dashboard` | Dashboard interactivo desarrollado con Streamlit                 | 8501   |

## 3. Variables de entorno

El proyecto utiliza un archivo `.env` con las variables necesarias para conectar los servicios:

```env
POSTGRES_USER=admin
POSTGRES_PASSWORD=admin
POSTGRES_DB=streaming_db
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
DATABASE_URL=postgresql://admin:admin@postgres:5432/streaming_db
```

Estas variables permiten que la API se conecte correctamente a PostgreSQL dentro del entorno Docker.

## 4. Ejecución con Docker

Desde la raíz del proyecto, ejecutar:

```bash
docker compose up --build
```

Este comando construye y levanta los contenedores necesarios para ejecutar el sistema completo.

Durante la ejecución se realiza el siguiente flujo:

```text
CSV + PostgreSQL
↓
Pipeline ETL
↓
Dataset consolidado
↓
Modelo KMeans
↓
API FastAPI
↓
Dashboard Streamlit
```

## 5. Acceso a los servicios

Cuando los contenedores estén levantados, se puede acceder a:

| Servicio  | URL                     |
| --------- | ----------------------- |
| API       | `http://localhost:8000` |
| Dashboard | `http://localhost:8501` |

El dashboard permite revisar los segmentos de usuarios, comparar clusters y analizar indicadores como consumo mensual, gasto, antigüedad, promociones y dispositivos registrados.

## 6. Validación de datos

El proyecto incluye un script de validación del pipeline ETL:

```bash
python etl/validate.py
```

En Windows, usando Python Launcher:

```bash
py -3.11 etl/validate.py
```

Este script valida:

* columnas esperadas en cada fuente;
* valores nulos;
* duplicados en `id_cliente`;
* tipos de datos numéricos;
* coincidencia de usuarios entre las fuentes;
* correcta integración del dataset final.

Una salida esperada es:

```text
[OK] Dataset integrado validado correctamente.
```

## 7. Pruebas automatizadas

Para ejecutar las pruebas automatizadas:

```bash
python -m unittest tests/test_validacion_datos.py
```

En Windows:

```bash
py -3.11 -m unittest tests/test_validacion_datos.py
```

Una salida correcta debería mostrar:

```text
Ran 8 tests

OK
```

Estas pruebas permiten comprobar que las fuentes de datos existen, tienen la estructura esperada y pueden integrarse correctamente antes de aplicar el modelo KMeans.

## 8. Ejecución local sin Docker

Para ejecutar scripts de forma local, primero se deben instalar las dependencias:

```bash
pip install -r requirements.txt
```

Luego se pueden ejecutar las validaciones:

```bash
python etl/validate.py
```

Y las pruebas:

```bash
python -m unittest tests/test_validacion_datos.py
```

Para ejecutar el pipeline completo de forma local, se debe tener una base PostgreSQL disponible y configurada con los datos de perfil de usuarios.

## 9. Archivos generados

Durante la ejecución del pipeline y el entrenamiento del modelo se pueden generar archivos como:

```text
data/data_consolidada.csv
data/usuarios_segmentados.csv
data/evaluacion_k.csv
models/modelo_kmeans.pkl
models/scaler.pkl
models/pca.pkl
models/metricas.json
```

Estos archivos representan el dataset integrado, los usuarios segmentados, la evaluación de distintos valores de K y los objetos entrenados del modelo.

## 10. Solución de problemas comunes

Si Docker no levanta correctamente, se recomienda detener los contenedores y reconstruir:

```bash
docker compose down
docker compose up --build
```

Si la API no puede conectarse a PostgreSQL, revisar que el archivo `.env` tenga correctamente definidas las variables `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, `POSTGRES_HOST` y `POSTGRES_PORT`.

Si el dashboard no muestra datos, verificar primero que la API esté funcionando en `http://localhost:8000`.

## 11. Conclusión

El uso de Docker permite ejecutar el sistema de forma reproducible, integrando base de datos, pipeline ETL, modelo KMeans, API y dashboard en un flujo end-to-end.
