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
perfil_usuarios.csv → Postgres ─┼─→ ETL (extract → validate → transform → load)
                                                     │
                                                     ▼
                                       dataset consolidado de clientes
                                                     │
                                                     ▼
                              escalamiento → KMeans (k óptimo vía codo + silhouette)
                                                     │
                                                     ▼
                                    dashboard interactivo (Streamlit/Dash)
```

Diagrama completo en [`docs/arquitectura.md`](docs/arquitectura.md).

## Estructura del repositorio

```
.
├── etl/            # Pipeline de extracción, validación, transformación y carga
├── model/          # Entrenamiento de KMeans, selección de k, perfilamiento de clusters
├── api/            # API REST (fuente de datos #3 — pendiente de definir)
├── dashboard/       # Dashboard interactivo
├── tests/          # Pruebas automatizadas
├── docker/         # Dockerfiles y docker-compose
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

### Opción B: Docker (recomendado para la demo)

```bash
cp .env.example .env
docker compose -f docker/docker-compose.yml up --build
```

## Flujo de trabajo en Git

Cada integrante trabaja en su propia rama (`feature/<nombre-de-la-parte>`) y
todo cambio llega a `main` mediante Pull Request con revisión de al menos otro
integrante. Ver [`docs/guia_contribucion.md`](docs/guia_contribucion.md).


