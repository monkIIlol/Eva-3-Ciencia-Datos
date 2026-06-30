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

### Opción A: entorno local

```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Luego ver [`docs/guia_despliegue.md`](docs/guia_despliegue.md) para los pasos
de configuración de Postgres y variables de entorno.

### Validación de datos y pruebas automatizadas

Antes de ejecutar el modelo de segmentación, el proyecto incluye una etapa de validación de datos para revisar que las fuentes utilizadas en el pipeline ETL sean consistentes.

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

Esta etapa ayuda a asegurar que los datos estén correctamente preparados antes de aplicar KMeans y utilizar los resultados en el dashboard.


### Opción B: Docker (recomendado para la demo)

```bash
docker compose up --build
```

> Nota: la configuración por variables de entorno (`.env`) todavía no está
> implementada — es una mejora pendiente. Por ahora, las credenciales de
> Postgres están definidas directamente en `docker-compose.yml`.
