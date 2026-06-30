# Arquitectura del Sistema

## Diagrama de Flujo de Datos e Integración
┌──────────────────────────┐                                    ┌────────────────────────────┐

│  usuarios_streaming.csv  │                                    │   perfil_usuarios (DB)     │

│  (Archivo plano local)   │                                    │   → Guardado en Postgres   │
└────────────┬─────────────┘                                    └─────────────┬──────────────┘

│                                                                │
└──────────────────┬─────────────────────────────┬───────────────┘
▼                             ▼
┌─────────────────────────────────────────────┐
│               etl/extract.py                │
│ - extract_csv()                             │
│ - extraer_postgres() (id_cliente join)      │
└─────────────────────┬───────────────────────└
▼
┌─────────────────────────────────────────────┐
│               etl/validate.py               │
│ - Validación de rangos, nulos y unicidad    │
└─────────────────────┬───────────────────────└
▼
┌─────────────────────────────────────────────┐
│               model/train.py                │
│ - StandardScaler / Pipeline                 │
│ - Búsqueda de K óptimo (Codo + Silhouette)  │
│ - Entrenamiento KMeans (k=3) y guardado     │
└─────────────────────┬───────────────────────└
▼
┌─────────────────────────────────────────────┐
│               api/main.py (FastAPI)         │
│ - Expone puerto 8000                        │
│ - Endpoint: /dashboard-data                 │
└─────────────────────┬───────────────────────└
▼ (Consumo vía HTTP)
┌─────────────────────────────────────────────┐
│             dashboards/app.py               │
│ - Streamlit (Puerto 8501)                   │
│ - Control de espera / Reintentos activos    │
└─────────────────────────────────────────────┘
## Orquestación de Contenedores (Docker Compose)

El entorno completo se levanta de forma aislada y segura utilizando las variables de entorno estandarizadas en el archivo `.env` (`POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, `POSTGRES_HOST`, `POSTGRES_PORT` y `DATABASE_URL`):

| Servicio | Imagen Base | Puerto Externo | Rol dentro del Sistema |
|---|---|---|---|
| `postgres` | `postgres:16-alpine` | `5432` | Base de datos relacional que aloja la tabla `perfil_usuarios` utilizando las credenciales seguras del entorno. |
| `api` | `python:3.11-slim` | `8000` | Contenedor crítico. Ejecuta secuencialmente el pipeline ETL (`extract.py`), entrena el modelo KMeans (`train.py`) y activa el servidor FastAPI para servir los datos estructurados. |
| `dashboard` | `python:3.11-slim` | `8501` | Aplicación web interactiva en Streamlit organizada por pestañas de audiencia (Ejecutiva, Técnica y Operativa). Consume los datos mediante peticiones HTTP internas. |

### Sincronización y Dependencias:
* `api` espera a que `postgres` esté completamente levantado y respondiendo peticiones antes de iniciar el procesamiento ETL.
* `dashboard` incorpora un bucle de reintento automatizado (`time.sleep`) para esperar activamente a que el contenedor `api` termine de procesar las operaciones de ciencia de datos antes de pintar los gráficos en pantalla.

## Decisiones de diseño relevantes

Ver [`decisiones_diseno.md`](decisiones_diseno.md) para la justificación detallada de cada elección técnica.