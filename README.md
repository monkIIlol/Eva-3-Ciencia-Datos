# Analítica de Clientes y Segmentación de Usuarios de Streaming

Proyecto de Ciencia de Datos — **SCY1101 Programación para la Ciencia de Datos**  
Base técnica desarrollada para la Evaluación 3 y extendida para la Evaluación Final Transversal.

## Resumen

Esta solución integra datos de consumo y perfil de usuarios de una plataforma de streaming, valida su calidad, prepara datasets analíticos, entrena modelos no supervisados y supervisados, publica artefactos solo cuando el flujo completo termina correctamente y expone los resultados mediante una API REST y un dashboard interactivo.

El proyecto incluye:

- segmentación de usuarios con **KMeans**;
- visualización de los segmentos mediante **PCA**;
- estimación del gasto mensual con modelos de **regresión**;
- clasificación de perfiles de **bajo compromiso** mediante una etiqueta proxy;
- ETL con contratos de validación, control de errores y unión `one-to-one`;
- API con **FastAPI** y validación de entradas con Pydantic;
- dashboard con **Streamlit**;
- ejecución reproducible mediante **Docker Compose**;
- pruebas automatizadas sobre datos, modelos, API y pipeline completo.

## Integrantes

- Verónica Cereceda
- Diego Torres
- Cristian Urcullú

## Problema de negocio

El negocio utiliza estrategias generales para todos sus usuarios, sin diferenciar niveles de consumo, gasto, antigüedad, uso de promociones o características de perfil. Esto puede producir campañas poco efectivas y decisiones poco personalizadas.

La solución entrega tres productos analíticos:

1. **Segmentación de usuarios:** identifica grupos con comportamientos similares para apoyar campañas y estrategias diferenciadas.
2. **Predicción de gasto mensual:** estima el gasto esperado de un usuario a partir de variables de comportamiento y perfil.
3. **Riesgo de bajo compromiso:** identifica perfiles compatibles con baja participación en la plataforma.

> La clasificación de bajo compromiso utiliza una **etiqueta proxy** construida a partir de sesiones semanales y porcentaje de finalización. No corresponde a churn, cancelación ni abandono observado.

El contexto ampliado se encuentra en [`docs/contexto_negocio.md`](docs/contexto_negocio.md).

## Fuentes de datos

El pipeline utiliza dos fuentes reales:

1. `data/usuarios_streaming.csv`: consumo y comportamiento en la plataforma.
2. PostgreSQL, tabla `perfil_usuarios`: variables de perfil cargadas inicialmente desde `database/perfil_usuarios.csv`.

Las fuentes se relacionan mediante `id_cliente`. La integración exige:

- identificadores válidos y únicos en cada fuente;
- el mismo universo de usuarios;
- relación estricta `one-to-one`;
- esquema mínimo requerido;
- nombres de columnas normalizados a `str` antes del modelado.

## Arquitectura end-to-end

```text
usuarios_streaming.csv ────────────────┐
                                       │
perfil_usuarios.csv → PostgreSQL ──────┤
                                       ▼
                              Extracción de fuentes
                               etl/extract.py
                                       │
                                       ▼
                     Validación estructural y compatibilidad
                               etl/validate.py
                                       │
                                       ▼
                         Integración one-to-one controlada
                              etl/integrate.py
                                       │
                                       ▼
                     Limpieza + validación posterior estricta
                          etl/prepare_dataset.py
                                       │
                         ┌─────────────┴─────────────┐
                         ▼                           ▼
              Dataset base limpio          Dataset analítico + KPIs
                         │
             ┌───────────┴──────────────────┐
             ▼                              ▼
       KMeans + PCA               Regresión + clasificación
       model/train.py             model/train_supervisado.py
             └───────────┬──────────────────┘
                         ▼
               Validación de productos finales
                         │
                         ▼
             Publicación diferida de artefactos
                models/pipeline_manifest.json
                         │
               ┌─────────┴─────────┐
               ▼                   ▼
          API FastAPI       Dashboard Streamlit
```

El orquestador principal se encuentra en:

```text
pipeline/run.py
```

Y se ejecuta con:

```bash
python -m pipeline.run
```

## Garantías del pipeline

El pipeline no se limita a ejecutar scripts en secuencia. Las salidas de cada etapa se entregan directamente como objetos a la etapa siguiente y deben cumplir contratos explícitos.

### Antes de limpiar

Se distinguen dos tipos de problemas:

- **Errores bloqueantes:** fuente vacía, columnas requeridas ausentes, identificadores inválidos o duplicados, incompatibilidad entre usuarios o fallo de conexión.
- **Anomalías corregibles:** nulos analíticos, infinitos, tipos convertibles o valores fuera de dominio.

### Después de limpiar

Los nulos, infinitos, tipos incorrectos, identificadores repetidos y valores fuera de los rangos definidos pasan a ser errores bloqueantes.

### Publicación diferida

Los datasets y modelos se publican solo después de que:

1. la extracción termine;
2. las fuentes sean válidas;
3. la integración sea consistente;
4. la preparación apruebe la validación posterior;
5. KMeans y los modelos supervisados terminen;
6. los productos finales sean compatibles.

Cada archivo se escribe de forma atómica y el manifiesto se genera al final:

```text
models/pipeline_manifest.json
```

La presencia de un manifiesto con `"status": "completed"` indica que la ejecución llegó al final de la publicación. Esta estrategia evita publicar resultados parciales, aunque no constituye una transacción atómica única para todos los archivos.

## Preparación de datos

`etl/prepare_dataset.py` realiza:

- eliminación de duplicados exactos;
- conversión controlada de variables numéricas;
- reemplazo de infinitos;
- tratamiento de valores fuera de dominio;
- imputación por mediana en variables analíticas;
- tratamiento especial de `id_cliente`, que nunca se imputa;
- diagnóstico de outliers mediante IQR;
- optimización de tipos;
- creación de variables derivadas;
- generación de KPIs y reporte de calidad.

Productos principales:

```text
data/data_consolidada.csv
data/dataset_base_limpio.csv
data/dataset_analitico.csv
data/dataset_modelo.csv
data/kpis_negocio.csv
data/reporte_calidad.json
```

`dataset_modelo.csv` se conserva temporalmente como salida de compatibilidad con componentes anteriores.

## Contratos de variables para los modelos

Las vistas de modelado se declaran en [`config/features.py`](config/features.py).

- **KMeans:** 15 variables originales de consumo y perfil.
- **Regresión:** predice `gasto_mensual` y excluye el target de sus variables de entrada.
- **Clasificación:** predice `riesgo_bajo_compromiso` y excluye `sesiones_semana` y `porcentaje_finalizacion`, porque esas variables construyen directamente la etiqueta proxy.
- Las variables derivadas de negocio no se incorporan automáticamente a los modelos.

Esta separación evita seleccionar columnas de forma implícita y reduce riesgos de fuga de información.

## Modelos y metodología

### Segmentación con KMeans

El modelo:

- escala las 15 variables mediante `StandardScaler`;
- evalúa valores de `k` entre 2 y 10;
- calcula inercia y coeficiente Silhouette;
- utiliza `KneeLocator` para detectar el codo;
- usa Silhouette como alternativa controlada si no existe un codo válido;
- entrena PCA con dos componentes únicamente para visualización.

PCA no determina los clusters ni mide por sí solo la calidad del modelo.

### Regresión

Candidatos:

- regresión lineal;
- Random Forest Regressor.

Métricas:

- R²;
- MAE;
- RMSE.

### Clasificación

Candidatos:

- regresión logística;
- Random Forest Classifier.

Métricas:

- accuracy;
- precision;
- recall;
- F1-score;
- ROC-AUC;
- matriz de confusión.

### Selección sin usar el conjunto de prueba

Los modelos supervisados siguen este procedimiento:

1. separación entrenamiento/prueba de 80/20;
2. cálculo de la etiqueta proxy y sus umbrales usando solo entrenamiento;
3. búsqueda de hiperparámetros y selección mediante validación cruzada en entrenamiento;
4. evaluación única del ganador y de los candidatos sobre el conjunto test;
5. reajuste del modelo seleccionado con todos los datos para uso productivo.

El conjunto test funciona como auditoría final y no como criterio de selección.

## Resultados de la ejecución verificada

Los siguientes valores corresponden al dataset actual, con 300 usuarios y `random_state=29`. Pueden variar si cambian los datos, las versiones de las dependencias o la configuración.

| Componente | Modelo seleccionado | Resultado principal |
|---|---|---|
| Segmentación | KMeans, `k=3` | Silhouette `0.2311` |
| Visualización | PCA, 2 componentes | Varianza explicada acumulada `44.88%` |
| Regresión | Random Forest Regressor | CV R² `0.7431`, test R² `0.7589` |
| Regresión | Random Forest Regressor | MAE `51.61`, RMSE `70.26` |
| Clasificación | Random Forest Classifier | CV F1 `0.7864`, test F1 `0.9091` |
| Clasificación | Random Forest Classifier | test ROC-AUC `0.9773`, accuracy `0.95` |

Los resultados deben interpretarse como evidencia técnica sobre este conjunto de datos, no como demostración de impacto comercial real.

## Estructura del repositorio

```text
.
├── .github/workflows/       # Workflow de integración continua
├── api/                     # API REST con FastAPI
├── config/                  # Configuración y contratos de variables
├── dashboards/              # Dashboard Streamlit
├── database/                # Inicialización y datos de PostgreSQL
├── data/                    # Fuentes, datasets procesados y resultados
├── docker/                  # Dockerfiles
├── docs/                    # Documentación técnica y de negocio
├── etl/                     # Extracción, validación, integración y preparación
├── model/                   # Entrenamiento no supervisado y supervisado
├── models/                  # Artefactos serializados y manifiesto local
├── pipeline/                # Orquestador end-to-end
├── tests/                   # Pruebas automatizadas
├── docker-compose.yml
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

## Requisitos

### Ejecución recomendada

- Docker Desktop;
- Docker Compose.

### Desarrollo local

- Python 3.11;
- pip;
- Docker para levantar PostgreSQL.

Python 3.11 es la versión de referencia utilizada por las imágenes Docker. Se recomienda usar la misma versión localmente.

## Inicio rápido con Docker

### 1. Crear el archivo de entorno

En PowerShell:

```powershell
Copy-Item .env.example .env
```

En Linux o macOS:

```bash
cp .env.example .env
```

Las credenciales incluidas son solo para desarrollo local.

### 2. Construir y levantar la solución

```bash
docker compose up --build -d
```

Docker Compose ejecuta el flujo en este orden:

```text
PostgreSQL healthy
        ↓
pipeline.run termina correctamente
        ↓
API healthy
        ↓
dashboard iniciado
```

### 3. Verificar los servicios

```bash
docker compose ps -a
```

Estado esperado:

```text
postgres    Up (healthy)
pipeline    Exited (0)
api         Up (healthy)
dashboard   Up
```

`pipeline` es un trabajo de inicialización y entrenamiento. Que termine con `Exited (0)` es correcto; no es un servidor permanente.

### 4. Acceder a la solución

- API: `http://localhost:8000`
- Health check: `http://localhost:8000/health`
- Swagger: `http://localhost:8000/docs`
- Dashboard: `http://localhost:8501`

### 5. Revisar logs

```bash
docker compose logs pipeline
docker compose logs api
docker compose logs dashboard
```

### 6. Inspeccionar el manifiesto dentro de Docker

Los modelos de Docker se almacenan en el volumen nombrado `modelos`:

```bash
docker compose exec api cat /app/models/pipeline_manifest.json
```

Los archivos de `data/` se escriben en la carpeta local mediante un bind mount. Los modelos generados dentro de Docker no son los mismos archivos físicos que aparecen en `./models` durante una ejecución local.

### 7. Detener la solución

```bash
docker compose down
```

Para borrar también el volumen de modelos:

```bash
docker compose down -v
```

> `docker compose down -v` elimina los modelos entrenados almacenados en el volumen de Docker, por lo que el pipeline deberá ejecutarse nuevamente.

## Ejecución local

### 1. Crear y activar un entorno virtual

PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux o macOS:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2. Instalar dependencias

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

### 3. Levantar PostgreSQL

```bash
docker compose up -d postgres
```

La ejecución local usa por defecto `localhost:5432`. El hostname `postgres` se utiliza dentro de la red de Docker Compose.

### 4. Ejecutar el pipeline completo

```bash
python -m pipeline.run
```

Esta ejecución publica datasets en `./data` y modelos en `./models`.

### 5. Levantar la API localmente

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

El dashboard está configurado para consumir `http://api:8000` dentro de Docker Compose, por lo que su ejecución recomendada es mediante contenedor.

## Ejecución individual de etapas

Los comandos individuales son útiles para diagnóstico, pero el camino oficial es `python -m pipeline.run`.

```bash
python -m etl.extract
python -m etl.validate
python -m etl.prepare_dataset
python -m model.train
python -m model.train_supervisado
```

Los entrenamientos individuales requieren que existan previamente sus datasets de entrada.

## Pruebas automatizadas

Para ejecutar toda la suite:

```bash
python -m unittest discover tests
```

En el estado verificado del proyecto:

```text
Ran 51 tests
OK
```

La suite cubre:

- validación estructural de fuentes;
- errores bloqueantes y advertencias corregibles;
- integración `one-to-one`;
- normalización de columnas provenientes de SQLAlchemy;
- limpieza y validación posterior;
- variables derivadas, KPIs y reportes;
- KMeans, PCA y consistencia de tipos numéricos;
- regresión y clasificación;
- prevención de fuga de variables del target proxy;
- selección por validación cruzada;
- pipeline completo y publicación diferida;
- esquemas y respuestas de la API;
- endpoints de inferencia.

La advertencia de deprecación entre Starlette y `httpx` puede aparecer en el entorno local actual, pero no implica un fallo de las pruebas.

## API REST

| Método | Endpoint | Descripción |
|---|---|---|
| GET | `/` | Estado general y artefactos disponibles |
| GET | `/health` | Estado técnico de la API y los modelos |
| GET | `/dashboard-data` | Datos de segmentos, centroides y evaluación de `k` |
| GET | `/metricas-supervisado` | Métricas y evidencia de modelos supervisados |
| POST | `/predict` | Asigna un cluster mediante KMeans |
| POST | `/predict-gasto` | Estima el gasto mensual |
| POST | `/predict-riesgo` | Estima la probabilidad del proxy de bajo compromiso |

La especificación interactiva y los esquemas de entrada están disponibles en:

```text
http://localhost:8000/docs
```

Las entradas son validadas por Pydantic. La API rechaza campos ausentes, tipos inválidos, valores fuera de rango y campos no definidos por el contrato.

## Artefactos generados

### Datos y evidencia

```text
data/data_consolidada.csv
data/dataset_base_limpio.csv
data/dataset_analitico.csv
data/dataset_modelo.csv
data/kpis_negocio.csv
data/reporte_calidad.json
data/usuarios_segmentados.csv
data/evaluacion_k.csv
data/centroides.csv
data/predicciones_regresion_test.csv
data/predicciones_clasificacion_test.csv
```

### Modelos y métricas

```text
models/modelo_kmeans.pkl
models/scaler.pkl
models/pca.pkl
models/metricas.json
models/modelo_regresion_gasto.pkl
models/modelo_clasificacion_riesgo.pkl
models/metricas_supervisado.json
models/pipeline_manifest.json
```

## Variables de entorno

La plantilla `.env.example` define:

```text
POSTGRES_USER
POSTGRES_PASSWORD
POSTGRES_DB
POSTGRES_HOST
POSTGRES_PORT
```

También se pueden configurar:

```text
RANDOM_STATE
N_JOBS
LOG_LEVEL
POSTGRES_TABLE
```

Docker Compose consume `.env` automáticamente. El código Python utiliza variables del entorno del proceso y valores por defecto definidos en [`config/settings.py`](config/settings.py); no carga `.env` por sí mismo.

## Integración continua

El repositorio incluye `.github/workflows/ci.yml`. Antes de presentar CI/CD como completamente cerrado, el workflow debe mantenerse alineado con la arquitectura actual:

- ejecutar el pipeline integrado o sus fixtures contractuales;
- disponer de PostgreSQL cuando se pruebe la extracción real;
- ejecutar la suite completa;
- construir las imágenes Docker;
- diferenciar integración continua de despliegue continuo.

En el estado actual, la ruta reproducible y verificada es Docker Compose más la suite local. No existe todavía un despliegue automático a un entorno productivo.

## Limitaciones

- El dataset contiene 300 usuarios; las métricas deben interpretarse considerando el tamaño de muestra.
- La etiqueta de clasificación representa bajo compromiso y no churn observado.
- El Silhouette de KMeans indica una separación moderada, no segmentos perfectamente aislados.
- PCA explica alrededor del 44.9% de la varianza con dos componentes y se utiliza solo para visualización.
- Los identificadores de cluster son arbitrarios. Las interpretaciones de negocio del dashboard deben revisarse después de reentrenamientos que alteren la asignación de etiquetas.
- La publicación es diferida y atómica por archivo, pero no una transacción única entre todos los archivos.
- El health check confirma disponibilidad de artefactos; las pruebas automatizadas y las solicitudes reales verifican además la inferencia.
- No existe evidencia causal de que las recomendaciones reduzcan abandono o aumenten ingresos.

## Documentación adicional

- [`docs/contexto_negocio.md`](docs/contexto_negocio.md)
- [`docs/arquitectura.md`](docs/arquitectura.md)
- [`docs/diseno_pipeline_integrado.md`](docs/diseno_pipeline_integrado.md)
- [`docs/pipeline_etl.md`](docs/pipeline_etl.md)
- [`docs/guia_despliegue.md`](docs/guia_despliegue.md)
- [`docs/guia_pruebas.md`](docs/guia_pruebas.md)
- [`docs/manual_usuario_dashboard.md`](docs/manual_usuario_dashboard.md)
- [`docs/decisiones_diseno.md`](docs/decisiones_diseno.md)

## Estado actual

La implementación principal se encuentra integrada y verificada de extremo a extremo:

```text
CSV + PostgreSQL
→ validación
→ integración
→ limpieza
→ feature engineering y KPIs
→ KMeans/PCA
→ regresión y clasificación
→ validación final
→ publicación
→ API
→ dashboard
```

La prioridad siguiente es mantener sincronizados el workflow de CI, la documentación secundaria y las evidencias de presentación con esta arquitectura final.