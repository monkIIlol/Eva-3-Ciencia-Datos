# Segmentación de Usuarios de Streaming

Proyecto E3 — SCY1101 Programación para la Ciencia de Datos

Pipeline end-to-end que integra datos de consumo y perfil de usuarios de una
plataforma de streaming, construye un modelo de segmentación con KMeans y
expone los resultados en un dashboard interactivo.

## Integrantes

- Verónica Cereceda
- Diego Torres
- Cristian Urcullú

## Contexto del problema

Ver [`docs/contexto_negocio.md`](docs/contexto_negocio.md) para el detalle completo
del caso de negocio y las fuentes de datos.

En resumen: se busca identificar segmentos de usuarios con comportamientos
similares (consumo, gasto, antigüedad, uso de promociones, etc.) para diseñar
estrategias diferenciadas de retención y recomendación.

## Arquitectura

```
usuarios_streaming.csv ─┐
perfil_usuarios.csv → Postgres ─┼─→ etl/extract.py (extraer + integrar)
                                                     │
                                                     ▼
                                       dataset consolidado de usuarios
                                                     │
                                                     ▼
                              escalamiento → KMeans (k óptimo vía codo + silhouette)
                                                     │
                                                     ▼
                                    dashboard interactivo (Streamlit)
```

Diagrama completo en [`docs/arquitectura.md`](docs/arquitectura.md).

## Estructura del repositorio

```
.
├── etl/            # Pipeline de extracción, validación e integración de fuentes
├── model/          # Entrenamiento de KMeans, selección de k, perfilamiento de clusters
├── api/            # API REST que expone el modelo entrenado (FastAPI)
├── dashboards/     # Dashboard interactivo
├── tests/          # Pruebas automatizadas
├── docker/         # Dockerfiles
├── database/       # init.sql y carga de perfil_usuarios en Postgres
├── docs/           # Documentación técnica y de negocio
└── data/
```

## Cómo correr el proyecto

### Opción A: Docker (recomendado, levanta el sistema completo)

Requiere tener Docker Desktop instalado y corriendo.

1. Crea el archivo `.env` a partir de la plantilla incluida:

   ```bash
   cp .env.example .env
   ```

   (En Windows, también puedes copiarlo manualmente desde el explorador de archivos.)

2. Levanta todos los servicios:

   ```bash
   docker compose up --build
   ```

Esto inicia Postgres, ejecuta el pipeline (extracción + validación + entrenamiento)
y levanta la API y el dashboard. Una vez arriba:

- API: http://localhost:8000 (documentación interactiva en http://localhost:8000/docs)
- Dashboard: http://localhost:8501

Ver [`docs/guia_despliegue.md`](docs/guia_despliegue.md) para más detalle.

### Opción B: validación y pruebas en entorno local

La validación de datos y las pruebas automatizadas se pueden ejecutar
localmente sin Docker (no requieren la base de datos). El pipeline completo,
en cambio, está pensado para correr con Docker (Opción A).

Antes de ejecutar el modelo de segmentación, el proyecto incluye una etapa de
validación de datos para revisar que las fuentes utilizadas en el pipeline ETL
sean consistentes.

Para ejecutar la validación de datos:

```bash
python etl/validate.py
```

En Windows, si se usa Python Launcher:

```bash
py -3.11 etl/validate.py
```

Esta validación revisa:

* columnas esperadas en cada fuente;
* valores nulos;
* duplicados en `id_cliente`;
* tipos de datos numéricos;
* coincidencia de usuarios entre `usuarios_streaming.csv` y `perfil_usuarios.csv`;
* correcta integración del dataset final.

Además, el proyecto cuenta con pruebas automatizadas para validar las fuentes de datos:

```bash
python -m unittest tests/test_validacion_datos.py
```

En Windows, si se usa Python Launcher:

```bash
py -3.11 -m unittest tests/test_validacion_datos.py
```

Si las pruebas se ejecutan correctamente, se espera una salida similar a:

```text
Ran 8 tests in 0.035s

OK
```

Esta etapa ayuda a asegurar que los datos estén correctamente preparados antes
de aplicar KMeans y utilizar los resultados en el dashboard.

## Configuración por variables de entorno

Las credenciales de conexión a Postgres se gestionan mediante variables de
entorno, no están escritas en el código. El archivo `.env` (no versionado)
contiene los valores reales, y `.env.example` sirve como plantilla de las
variables necesarias: `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`,
`POSTGRES_HOST` y `POSTGRES_PORT`.

## Flujo de trabajo en Git

Cada integrante trabaja en su propia rama y todo cambio llega a `main`
mediante Pull Request con revisión de al menos otro integrante. La rama `main`
está protegida para requerir aprobación antes de fusionar.