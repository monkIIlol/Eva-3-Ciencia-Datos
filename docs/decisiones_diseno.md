# Decisiones de Diseño del Proyecto

## 1. Enfoque general del proyecto

El proyecto se desarrolló como una solución end-to-end para segmentar usuarios de una plataforma de streaming digital. El objetivo principal fue integrar distintas fuentes de datos, preparar un conjunto analítico consolidado, aplicar un modelo de aprendizaje no supervisado y comunicar los resultados mediante un dashboard interactivo.

La solución considera un flujo completo desde la carga de datos hasta la visualización de los segmentos encontrados.

## 2. Integración de fuentes de datos

Se trabajó con dos fuentes principales:

* `usuarios_streaming.csv`: contiene información de consumo dentro de la plataforma.
* `perfil_usuarios.csv`: contiene información complementaria del usuario y se carga mediante PostgreSQL.

Ambas fuentes se integran utilizando la columna `id_cliente`, ya que esta permite relacionar el comportamiento de consumo con el perfil del usuario.

Esta integración es importante porque el modelo necesita una visión más completa de cada usuario. Si solo se usaran datos de consumo, se perdería información relevante del perfil. Si solo se usaran datos de perfil, no se reflejaría el comportamiento real dentro de la plataforma.

## 3. Validación del pipeline ETL

Antes de aplicar el modelo, se implementó una validación de datos para revisar que las fuentes fueran consistentes.

Las validaciones aplicadas fueron:

* revisión de columnas esperadas;
* revisión de valores nulos;
* revisión de `id_cliente` duplicados;
* revisión de tipos numéricos;
* revisión de coincidencia de usuarios entre ambas fuentes;
* validación del dataset integrado final.

Esto permite detectar errores antes del entrenamiento del modelo y evita trabajar con datos incompletos o mal integrados.

## 4. Selección de variables para el modelo

Para el modelo final se utilizaron solamente variables numéricas, ya que KMeans trabaja con distancias entre puntos. La columna `id_cliente` no se usó como variable predictora, porque solo identifica al usuario y no representa una característica de comportamiento.

Las variables utilizadas representan aspectos como:

* consumo mensual;
* gasto mensual;
* cantidad de contenidos vistos;
* sesiones semanales;
* porcentaje de finalización;
* uso de promociones;
* antigüedad;
* edad;
* dispositivos registrados;
* uso de aplicación móvil;
* interacciones con soporte.

Estas variables permiten construir una segmentación basada en comportamiento y perfil de usuario.

## 5. Uso de escalamiento

Se aplicó escalamiento antes de entrenar el modelo KMeans.

Esta decisión es importante porque las variables tienen unidades y rangos distintos. Por ejemplo, el gasto mensual puede tener valores mucho más altos que el porcentaje de uso de promociones o la cantidad de dispositivos registrados.

Si no se escalan los datos, las variables con valores más grandes pueden dominar el cálculo de distancia y afectar la formación de los clusters.

## 6. Uso de KMeans

Se utilizó KMeans porque el objetivo del proyecto es segmentar usuarios en grupos con características similares.

KMeans permite dividir los datos en K clusters, donde cada usuario se asigna al grupo cuyo centroide se encuentra más cerca según sus características.

Este algoritmo es adecuado para el caso porque permite encontrar patrones de comportamiento sin tener una etiqueta previa. Es decir, no se sabe de antemano qué tipo de usuario es cada persona, sino que el modelo ayuda a descubrir esos perfiles.

## 7. Selección del número de clusters

Para seleccionar el número de clusters se utilizaron dos criterios:

* método del codo;
* coeficiente Silhouette.

El método del codo permite observar cómo disminuye la inercia al aumentar el número de clusters. La idea es identificar el punto donde agregar más clusters ya no mejora de forma significativa la compactación de los grupos.

El coeficiente Silhouette permite evaluar qué tan bien separados están los clusters. Un valor más alto indica que los usuarios están mejor agrupados dentro de su cluster y más alejados de otros clusters.

Con base en estos análisis, se trabajó con 3 clusters, ya que entregan una segmentación interpretable y útil para el negocio.

## 8. Interpretación de segmentos

La interpretación de los clusters se realizó considerando los centroides y los promedios de las principales variables de cada grupo.

Los segmentos definidos fueron:

### Cluster 0: Usuarios habituales exploradores

Este grupo representa usuarios con sesiones frecuentes, pero de menor duración. Tienen un consumo moderado y exploran distintos contenidos dentro de la plataforma.

Desde el negocio, este segmento puede abordarse con recomendaciones personalizadas que aumenten la duración de las sesiones y fomenten la finalización de contenidos.

### Cluster 1: Usuarios nuevos sensibles a precio

Este grupo se caracteriza por tener menor antigüedad, menor consumo y mayor uso de promociones.

Desde el negocio, este segmento requiere estrategias de retención temprana, beneficios de bienvenida y campañas que reduzcan el abandono inicial.

### Cluster 2: Usuarios premium leales

Este grupo presenta mayor gasto, alto porcentaje de finalización y baja sensibilidad a promociones.

Desde el negocio, este segmento representa usuarios de alto valor. Se recomienda fidelizarlos mediante contenido exclusivo, beneficios premium y experiencias personalizadas, más que mediante descuentos.

## 9. Dashboard interactivo

El dashboard permite visualizar los resultados de la segmentación desde distintas perspectivas.

Se incorporan vistas orientadas a diferentes audiencias:

* vista ejecutiva: resume los segmentos y su valor de negocio;
* vista técnica: muestra métricas como inercia, Silhouette y evaluación de K;
* vista operativa: permite explorar usuarios y características de los clusters.

Esto facilita que los resultados no queden solo como análisis técnico, sino que puedan ser comprendidos por usuarios de negocio.

## 10. Aporte de la validación y testing

Se agregó una etapa de validación y pruebas automatizadas para asegurar que los datos utilizados por el modelo sean consistentes.

Esto mejora la robustez del pipeline, ya que permite comprobar automáticamente que las fuentes existan, tengan las columnas correctas, no tengan duplicados, no presenten nulos y puedan integrarse correctamente.

Con esto se busca disminuir errores antes de ejecutar el modelo y entregar una solución más confiable.
