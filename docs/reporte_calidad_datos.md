# Reporte de calidad de datos

## Objetivo

Este documento resume las reglas de limpieza, validación y transformación aplicadas durante la preparación de datos del proyecto.

La etapa de preparación busca asegurar que los modelos no trabajen directamente sobre datos crudos, sino sobre datasets limpios, validados y documentados.

El proceso se implementa principalmente en:

`etl/prepare_dataset.py`

y se ejecuta dentro del pipeline integrado mediante:

`pipeline/run.py`

## Archivos generados

| Archivo | Descripción |
|---|---|
| `data/dataset_base_limpio.csv` | Dataset limpio con variables base oficiales. Es la entrada principal para KMeans, regresión y clasificación. |
| `data/dataset_analitico.csv` | Dataset enriquecido con variables derivadas para análisis, KPIs y visualización. |
| `data/dataset_modelo.csv` | Archivo conservado por compatibilidad con versiones anteriores del proyecto. |
| `data/kpis_negocio.csv` | KPIs agrupados por nivel de engagement y valor de cliente. |
| `data/reporte_calidad.json` | Reporte automático con evidencia de calidad de datos. |

## Flujo aplicado

```text
data/data_consolidada.csv
↓
etl/prepare_dataset.py
↓
data/dataset_base_limpio.csv
data/dataset_analitico.csv
data/kpis_negocio.csv
data/reporte_calidad.json
↓
modelos + API + dashboard
```

## Tratamiento de identificadores

La columna `id_cliente` se trata como identificador único y crítico.

Por esta razón:

- no se imputa con mediana;
- no se inventa un valor estadístico;
- no puede ser nulo;
- no puede ser no numérico;
- no puede estar duplicado;
- no puede ser menor o igual a cero.

Si `id_cliente` no cumple estas reglas, el pipeline debe detener la ejecución.

Esta decisión evita crear clientes artificiales, duplicar identificadores o entrenar modelos con registros mal asociados.

## Tratamiento de nulos

Las variables numéricas se convierten a formato numérico cuando corresponde.

Para variables analíticas, los nulos pueden imputarse con la mediana cuando es metodológicamente válido.

Se usa mediana porque es menos sensible a valores extremos que el promedio.

Ejemplo:

```text
Si una columna tiene valores muy altos o muy bajos,
la media puede distorsionarse.
La mediana representa mejor el centro de los datos.
```

## Tratamiento de infinitos

Los valores infinitos (`inf` o `-inf`) se reemplazan antes de continuar con el procesamiento.

Esto es importante porque pueden aparecer por errores de cálculo, divisiones inválidas o datos mal formados.

## Reglas de rangos de negocio

Se aplican controles para evitar valores imposibles o inconsistentes.

| Variable | Regla general |
|---|---|
| `sesiones_semana` | valor positivo |
| `horas_consumo_mensual` | valor positivo |
| `gasto_mensual` | mínimo 0 |
| `cantidad_contenidos_vistos` | mínimo 0 |
| `tiempo_promedio_sesion_min` | mínimo 0 |
| `cantidad_generos_consumidos` | mínimo 0 |
| `antiguedad_cliente_meses` | mínimo 0 |
| `edad` | rango válido de edad |
| `dispositivos_registrados` | mínimo 1 |
| `cantidad_perfiles_creados` | mínimo 1 |
| `interacciones_mensuales_soporte` | mínimo 0 |
| `distancia_promedio_red_km` | mínimo 0 |
| `porcentaje_finalizacion` | rango porcentual válido |
| `porcentaje_uso_promociones` | rango entre 0 y 1 |
| `porcentaje_uso_app_movil` | rango entre 0 y 1 |

## Tratamiento de outliers

Se utiliza diagnóstico de outliers mediante el método IQR.

El IQR permite detectar valores atípicos usando el rango intercuartílico:

```text
IQR = Q3 - Q1
Límite inferior = Q1 - 1.5 * IQR
Límite superior = Q3 + 1.5 * IQR
```

Este análisis deja evidencia en el reporte de calidad y permite identificar variables con valores extremos.

En datasets pequeños, como este caso de 300 usuarios, conviene evitar eliminar registros sin justificación fuerte, porque cada fila representa un cliente y puede aportar información útil al análisis.

## Separación entre dataset base y dataset analítico

Una decisión importante del proyecto es separar dos salidas:

### `dataset_base_limpio.csv`

Contiene las variables originales limpias y validadas.

Este dataset se usa para entrenar:

- KMeans;
- regresión;
- clasificación.

### `dataset_analitico.csv`

Contiene variables derivadas para análisis, KPIs y visualización.

Esta separación evita que variables derivadas con información del objetivo entren automáticamente a los modelos.

Por ejemplo, si el modelo predice `gasto_mensual`, no corresponde usar `gasto_por_hora`, porque esa variable contiene información directa del gasto.

## Variables derivadas creadas

Se crearon variables nuevas para enriquecer el análisis:

| Variable | Justificación |
|---|---|
| `contenidos_por_sesion` | Mide intensidad de consumo por sesión mensual estimada. |
| `gasto_por_hora` | Relaciona gasto con consumo. Debe usarse con cuidado para evitar fuga de datos. |
| `minutos_totales_estimados` | Facilita la interpretación del consumo en minutos. |
| `soporte_por_dispositivo` | Mide carga de soporte relativa a dispositivos. |
| `generos_por_contenido` | Representa diversidad de consumo. |
| `engagement_score` | Resume compromiso del usuario. |
| `nivel_engagement` | Clasifica usuarios en bajo, medio y alto engagement. |
| `cliente_antiguo` | Identifica clientes con antigüedad relevante. |
| `uso_promociones_alto` | Marca usuarios sensibles a promociones. |
| `valor_cliente` | Segmenta usuarios por valor descriptivo. |

## Corrección conceptual aplicada

La variable `contenidos_por_sesion` no debe mezclar directamente una variable mensual con una semanal.

Por eso se usa una estimación mensual de sesiones:

```text
sesiones_mes_estimadas = sesiones_semana * 4
contenidos_por_sesion = cantidad_contenidos_vistos / sesiones_mes_estimadas
```

`sesiones_mes_estimadas` funciona como cálculo intermedio para obtener una interpretación más coherente de `contenidos_por_sesion`.

## KPIs de negocio

El archivo:

`data/kpis_negocio.csv`

resume indicadores agrupados por `nivel_engagement` y `valor_cliente`.

Incluye:

- cantidad de usuarios;
- gasto promedio;
- consumo promedio;
- finalización promedio;
- uso promedio de promociones;
- antigüedad promedio;
- porcentaje de usuarios.

Estos KPIs permiten interpretar el comportamiento de los clientes desde una mirada de negocio.

## Reporte automático de calidad

El archivo:

`data/reporte_calidad.json`

registra evidencia automática del proceso.

Puede incluir información como:

- cantidad de filas originales;
- cantidad de filas finales;
- columnas originales y finales;
- validación de identificadores;
- duplicados detectados;
- nulos antes y después de limpieza;
- diagnóstico de outliers;
- memoria utilizada;
- variables derivadas generadas;
- notas metodológicas.

## Optimización de memoria

El dataset preparado aplica reducción de tipos numéricos cuando corresponde.

Esto ayuda a disminuir el consumo de memoria y mejora la escalabilidad del pipeline para datasets de mayor tamaño.

## Pruebas automatizadas

Las pruebas se encuentran en la carpeta:

`tests/`

Incluyen validaciones relacionadas con:

- calidad de datos;
- preparación del dataset;
- generación de artefactos;
- modelos supervisados;
- API;
- pipeline integrado.

La cantidad exacta de pruebas puede variar según la versión del repositorio, pero el resultado esperado es que la suite finalice correctamente con `OK`.

Comando utilizado:

```bash
py -3.11 -m unittest discover tests
```

## Importancia para el proyecto

Esta etapa permite que el proyecto no dependa directamente de datos crudos.

El flujo mejora porque:

- controla errores antes del modelado;
- genera un dataset base limpio para los modelos;
- genera un dataset analítico para análisis y KPIs;
- produce evidencia automática de calidad de datos;
- reduce el riesgo de fuga de datos;
- permite defender técnicamente las decisiones de limpieza;
- se integra dentro del pipeline ejecutado por `pipeline/run.py`.

## Relación con los modelos

El pipeline integrado usa `dataset_base_limpio.csv` como entrada principal de los modelos.

Esto permite que KMeans, regresión y clasificación trabajen sobre datos preparados y validados.

El `dataset_analitico.csv` queda orientado a análisis, KPIs, interpretación y visualización.

## Limitaciones y mejoras futuras

Aunque la etapa de calidad fortalece el proyecto, todavía se pueden realizar mejoras:

- ampliar pruebas negativas con más casos problemáticos;
- visualizar el reporte de calidad directamente en el dashboard;
- exponer más KPIs desde la API;
- registrar tiempos de ejecución por etapa;
- agregar monitoreo de drift o cambios en distribución de datos;
- automatizar despliegue continuo real.

Estas mejoras no invalidan el flujo actual, sino que representan oportunidades para una versión más productiva y profesional.

## Frase para defensa

Mi aporte fue construir y documentar una etapa de preparación del dataset analítico. Esta etapa toma los datos consolidados, controla identificadores, nulos, infinitos, rangos y outliers, genera un dataset base limpio para modelos, un dataset analítico enriquecido, KPIs de negocio y un reporte de calidad. Además, esta etapa quedó integrada dentro del pipeline general, por lo que los modelos ya no dependen directamente de datos crudos.