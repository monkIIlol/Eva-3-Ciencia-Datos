# Manual de Usuario del Dashboard

## 1. Objetivo

El dashboard presenta la solución completa de analítica de usuarios de una plataforma de streaming. No se limita a mostrar clusters: integra resultados de negocio, calidad de datos, trazabilidad del pipeline, segmentación, modelos supervisados y una demostración de inferencia mediante API.

Su propósito es permitir que una audiencia ejecutiva, técnica u operativa pueda responder preguntas como:

- ¿qué grupos de usuarios existen y qué acciones convienen para cada uno?;
- ¿con qué calidad y trazabilidad se generaron los resultados?;
- ¿por qué se eligieron los modelos actuales?;
- ¿qué significan sus métricas y limitaciones?;
- ¿cómo responde el sistema ante un usuario real o hipotético?

## 2. Acceso

Con Docker Compose:

```text
http://localhost:8501
```

El dashboard consume la URL definida en `API_BASE_URL`. Dentro de Docker se utiliza `http://api:8000`; en una ejecución local puede utilizarse `http://localhost:8000`.

La barra lateral permite:

- revisar el estado de la API;
- actualizar los datos almacenados en caché;
- filtrar uno o más segmentos.

## 3. Resumen ejecutivo

Esta vista comunica resultados y acciones de negocio:

- usuarios analizados;
- cantidad de segmentos;
- gasto promedio;
- usuarios de alto valor;
- usuarios clasificados como valor en riesgo;
- distribución y gasto promedio por segmento;
- matriz de engagement y valor del cliente;
- descripción y acción sugerida para cada perfil.

Las acciones son recomendaciones analíticas. El proyecto no demuestra todavía un efecto causal ni un retorno económico medido.

## 4. Datos y pipeline

Esta pestaña hace visible el recorrido end-to-end:

```text
CSV + PostgreSQL
→ validación estructural
→ integración one-to-one
→ limpieza y variables derivadas
→ KMeans y PCA
→ regresión y clasificación
→ API y dashboard
```

También muestra:

- identificador y duración de la última ejecución;
- filas e IDs integrados;
- errores de validación;
- comparación antes/después de la limpieza;
- nulos, duplicados y valores fuera de rango;
- outliers detectados mediante IQR;
- reducción de memoria;
- variables derivadas;
- garantías de publicación y trazabilidad.

Los outliers se detectan y reportan, pero no se eliminan automáticamente porque pueden representar clientes reales de alto consumo o gasto.

## 5. Segmentación

La pestaña de segmentación incluye:

- `k` seleccionado;
- inercia y método del codo;
- coeficiente Silhouette;
- varianza explicada por PCA;
- proyección de usuarios en dos componentes;
- heatmap de perfiles relativos;
- promedios por segmento.

El Silhouette cercano a `0.231` indica separación moderada-baja. Los segmentos son interpretables, pero existe solapamiento.

PCA se utiliza únicamente para visualización. Dos componentes conservan aproximadamente `44.9%` de la varianza, por lo que la proyección 2D no contiene toda la información de las 15 variables originales.

Los números de cluster son arbitrarios. El dashboard genera etiquetas descriptivas a partir del perfil relativo de los centroides.

## 6. Modelos predictivos

### Regresión

Predice gasto mensual y presenta:

- modelo ganador;
- R² de validación cruzada y test;
- MAE y RMSE;
- comparación entre algoritmos;
- gasto real frente a predicho;
- distribución de residuos;
- importancia de variables cuando el modelo la admite.

R² expresa proporción de variabilidad explicada; no equivale a porcentaje de precisión. MAE representa error promedio y RMSE penaliza con mayor intensidad los errores grandes.

### Clasificación

Predice el proxy de bajo compromiso y presenta:

- modelo ganador;
- precision, recall, F1 y ROC-AUC;
- matriz de confusión;
- probabilidades de riesgo en el holdout;
- importancia de variables;
- umbrales utilizados para construir la etiqueta.

`sesiones_semana` y `porcentaje_finalizacion` construyen la etiqueta histórica, por lo que se excluyen de las variables del clasificador para evitar fuga de datos.

La clase representa bajo compromiso, no churn o cancelación real.

## 7. Operación y demo

### Usuario existente

Permite seleccionar un `id_cliente` y revisar:

- segmento asignado;
- gasto y finalización;
- recomendación operativa;
- comparación frente al centroide del segmento;
- tabla de usuarios filtrados.

### Usuario nuevo

El simulador ejecuta tres endpoints:

1. `/predict-gasto` estima gasto mensual;
2. `/predict` asigna un segmento utilizando el gasto estimado;
3. `/predict-riesgo` calcula el proxy de bajo compromiso.

El resultado muestra segmento, gasto, riesgo, probabilidad y acción sugerida. Los mensajes de error provienen del detalle validado por FastAPI.

## 8. Actualización de datos

Los datos se almacenan en caché durante 60 segundos. El botón **Actualizar datos** limpia la caché y vuelve a consultar la API.

Después de reentrenar modelos o ejecutar nuevamente el pipeline, se recomienda actualizar el dashboard y comprobar el `run_id` de la pestaña de pipeline.

## 9. Interpretación responsable

- Las métricas se obtienen con un dataset de 300 usuarios.
- Las categorías de valor y engagement son construcciones analíticas.
- Bajo compromiso es una etiqueta proxy.
- Los clusters no representan grupos naturales perfectamente separados.
- Las recomendaciones deben validarse mediante experimentación de negocio antes de implementarse a gran escala.
