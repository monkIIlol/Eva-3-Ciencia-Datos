# Arquitectura del sistema

## Diagrama de flujo de datos

```
┌─────────────────────────┐   ┌──────────────────────────┐   ┌─────────────────────┐
│  usuarios_streaming.csv │   │   perfil_usuarios.csv     │   │   API REST (TBD)     │
│  (archivo plano)         │   │   → cargado en Postgres   │   │   fuente 3            │
└────────────┬─────────────┘   └─────────────┬──────────────┘   └──────────┬───────────┘
             │                                │                              │
             └──────────────────┬─────────────┴───────────────┬──────────────┘
                                 ▼                              ▼
                       ┌───────────────────────────────────────────┐
                       │           etl/extract.py                  │
                       │  extract_usuarios_streaming()              │
                       │  extract_perfil_usuarios()                 │
                       │  extract_api_source()                      │
                       └─────────────────────┬───────────────────────┘
                                              ▼
                       ┌───────────────────────────────────────────┐
                       │           etl/validate.py                  │
                       │  Esquemas pandera: tipos, rangos,           │
                       │  nulabilidad, unicidad                      │
                       └─────────────────────┬───────────────────────┘
                                              ▼
                       ┌───────────────────────────────────────────┐
                       │           etl/transform.py                 │
                       │  merge_sources() → check_duplicates()       │
                       │  → select_model_features()                  │
                       └─────────────────────┬───────────────────────┘
                                              ▼
                       ┌───────────────────────────────────────────┐
                       │           etl/load.py                      │
                       │  Guarda dataset_consolidado.csv             │
                       │  en data/processed/                         │
                       └─────────────────────┬───────────────────────┘
                                              ▼
                       ┌───────────────────────────────────────────┐
                       │       model/train_kmeans.py                │
                       │  scale_features() → find_optimal_k()        │
                       │  (KneeLocator + silhouette) → train_kmeans()│
                       └─────────────────────┬───────────────────────┘
                                              ▼
                       ┌───────────────────────────────────────────┐
                       │      model/segment_profile.py               │
                       │  assign_clusters() → profile_segments()     │
                       │  + interpretación de negocio                │
                       └─────────────────────┬───────────────────────┘
                                              ▼
                       ┌───────────────────────────────────────────┐
                       │          dashboard/app.py                   │
                       │  Streamlit: visión general, perfilamiento,  │
                       │  comparación, filtros interactivos          │
                       └───────────────────────────────────────────┘
```

## Contenedores (Docker Compose)

| Servicio | Imagen base | Puerto | Rol |
|---|---|---|---|
| `postgres` | `postgres:16-alpine` | 5432 | Almacena `perfil_usuarios` |
| `api` | `python:3.11-slim` + FastAPI | 8000 | Sirve la fuente de datos #3 |
| `etl` | `python:3.11-slim` | — | Ejecuta el pipeline completo y termina |
| `dashboard` | `python:3.11-slim` + Streamlit | 8501 | Expone el dashboard interactivo |

El servicio `etl` depende de que `postgres` esté saludable (`healthcheck`)
y de que `api` esté disponible antes de ejecutarse. El `dashboard` depende
de que `etl` haya corrido al menos una vez (para tener datos en
`data/processed/`).

## Decisiones de diseño relevantes

Ver [`decisiones_diseno.md`](decisiones_diseno.md) para la justificación
detallada de cada elección (estructura de carpetas, estrategia de join,
selección de k, uso de PCA, herramienta de dashboard, etc.).
