# Arquitectura del sistema

## Visión general

El proyecto es un pipeline end-to-end que integra dos fuentes de datos,
entrena un modelo de segmentación (KMeans) y expone los resultados en un
dashboard interactivo. Todo el sistema se orquesta con Docker Compose y se
levanta con un solo comando.

## Diagrama de flujo de datos

```
┌──────────────────────────┐                       ┌────────────────────────────┐
│  usuarios_streaming.csv  │                       │   perfil_usuarios.csv      │
│  (archivo plano)         │                       │   → cargado en Postgres    │
└────────────┬─────────────┘                       └─────────────┬──────────────┘
             │                                                   │
             └──────────────────────┬────────────────────────────┘
                                     ▼
                       ┌──────────────────────────────────────────┐
                       │           etl/extract.py                 │
                       │  extraer_csv() → extraer_postgres()      │
                       │  → integrar()                            │
                       └─────────────────────┬────────────────────┘
                                             ▼
                       ┌──────────────────────────────────────────┐
                       │           etl/validate.py                │
                       │  Valida columnas, nulos, duplicados,     │
                       │  tipos numéricos y consistencia de       │
                       │  id_cliente entre fuentes                │
                       └─────────────────────┬────────────────────┘
                                             ▼
                       ┌──────────────────────────────────────────┐
                       │           model/train.py                 │
                       │  StandardScaler → KMeans                 │
                       │  (KneeLocator + Silhouette para k)       │
                       │  → PCA (solo visualización)              │
                       │  Genera: usuarios_segmentados.csv,       │
                       │  centroides.csv, evaluacion_k.csv,       │
                       │  metricas.json, modelo/scaler/pca .pkl   │
                       └─────────────────────┬────────────────────┘
                                             ▼
                       ┌──────────────────────────────────────────┐
                       │             api/main.py                  │
                       │  FastAPI. Endpoints:                     │
                       │  GET /  ·  GET /dashboard-data           │
                       │  POST /predict                           │
                       └─────────────────────┬────────────────────┘
                                             ▼
                       ┌──────────────────────────────────────────┐
                       │          dashboards/app.py               │
                       │  Streamlit. Consume la API por HTTP.     │
                       │  Vistas: Ejecutiva, Técnica, Operativa,  │
                       │  con filtro global de segmentos          │
                       └──────────────────────────────────────────┘
```


## Componentes del pipeline

| Etapa | Archivo | Responsabilidad |
|---|---|---|
| Extracción e integración | `etl/extract.py` | Lee el CSV y la tabla de Postgres, los une por `id_cliente` y guarda `data/data_consolidada.csv`. |
| Validación | `etl/validate.py` | Comprueba columnas esperadas, nulos, `id_cliente` duplicados, tipos numéricos y consistencia entre fuentes antes del modelo. |
| Entrenamiento | `model/train.py` | Escala las variables, determina el k óptimo (codo + Silhouette), entrena KMeans, aplica PCA para visualización y persiste resultados y modelo. |
| Servicio | `api/main.py` | Expone los resultados del modelo vía API REST (FastAPI). No es una fuente de datos: sirve lo ya calculado. |
| Visualización | `dashboards/app.py` | Dashboard Streamlit que consume la API por HTTP y presenta las tres vistas por audiencia. |
| Pruebas | `tests/test_validacion_datos.py` | Pruebas automatizadas (unittest) sobre la validación de datos. |


## Contenedores (Docker Compose)

| Servicio | Imagen base | Puerto | Rol |
|---|---|---|---|
| `postgres` | `postgres:16` | 5432 | Almacena la tabla `perfil_usuarios`. Se inicializa automáticamente con `database/init.sql`. |
| `api` | `python:3.11` + FastAPI | 8000 | Ejecuta en secuencia extracción + entrenamiento y luego levanta la API. |
| `dashboard` | `python:3.11` + Streamlit | 8501 | Consume la API y renderiza el dashboard. |

El servicio `etl` depende de que `postgres` esté saludable (`healthcheck`)
y de que `api` esté disponible antes de ejecutarse. El `dashboard` depende
de que `etl` haya corrido al menos una vez (para tener datos en
`data/processed/`).


### Orden de arranque y healthchecks

El arranque está orquestado para evitar condiciones de carrera:

- `postgres` expone un `healthcheck` con `pg_isready`. No se considera
  disponible hasta que acepta conexiones reales.
- `api` declara `depends_on: postgres` con `condition: service_healthy`,
  por lo que no arranca hasta que Postgres está realmente listo (no solo
  creado). Internamente ejecuta `extract.py → train.py → uvicorn`. A su vez
  expone su propio `healthcheck` contra el endpoint raíz.
- `dashboard` declara `depends_on: api` con `condition: service_healthy`,
  por lo que no arranca hasta que la API responde correctamente.

Esto reemplaza la dependencia simple por orden de creación, que no
garantizaba que el servicio dependido estuviera operativo.


## Configuración por variables de entorno

Las credenciales y parámetros de conexión a Postgres no están escritos en
el código ni en el `docker-compose.yml`. Se definen en un archivo `.env`
(no versionado) y se inyectan a los contenedores en tiempo de ejecución.
El repositorio incluye `.env.example` como plantilla de las variables
necesarias (`POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`,
`POSTGRES_HOST`, `POSTGRES_PORT`).


## Persistencia y reproducibilidad

Los archivos generados por el pipeline (`data_consolidada.csv`,
`usuarios_segmentados.csv`, `centroides.csv`, `evaluacion_k.csv`) no se
versionan: se regeneran cada vez que se ejecuta el sistema, lo que
garantiza reproducibilidad. El modelo entrenado y sus artefactos se
guardan en un volumen Docker (`modelos`) compartido dentro del servicio
`api`.


## Decisiones de diseño relevantes

Ver [`decisiones_diseno.md`](decisiones_diseno.md) para la justificación
detallada de cada elección (estrategia de integración, escalamiento,
selección de k, uso de PCA, interpretación de segmentos y dashboard por
audiencias).