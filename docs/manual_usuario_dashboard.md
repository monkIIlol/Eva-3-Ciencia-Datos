# Manual de Usuario del Dashboard

## 1. Objetivo del dashboard

El dashboard tiene como objetivo presentar los resultados del modelo de segmentación de usuarios de streaming de forma clara e interactiva.

Permite visualizar los segmentos encontrados por el modelo KMeans, comparar sus principales características y apoyar la toma de decisiones de negocio relacionadas con retención, fidelización y personalización de contenido.

## 2. Audiencia del dashboard

El dashboard está pensado para distintas audiencias:

* usuarios ejecutivos, que necesitan una visión general de los segmentos y su valor de negocio;
* usuarios técnicos, que necesitan revisar métricas del modelo como inercia, método del codo y coeficiente Silhouette;
* usuarios operativos, que necesitan explorar usuarios, clusters y características específicas.

## 3. Vista Ejecutiva

La vista ejecutiva resume los principales resultados de la segmentación.

En esta sección se puede observar:

* cantidad de usuarios por cluster;
* porcentaje de usuarios por segmento;
* descripción general de cada grupo;
* interpretación de negocio de los segmentos;
* recomendaciones generales para cada tipo de usuario.

Los segmentos definidos son:

### Cluster 0: Usuarios habituales exploradores

Corresponde a usuarios con sesiones frecuentes, pero de menor duración. Presentan un consumo moderado y exploran distintos contenidos dentro de la plataforma.

Acción recomendada:

* reforzar recomendaciones personalizadas;
* sugerir contenido relacionado;
* incentivar la finalización de contenidos;
* aumentar la duración de las sesiones.

### Cluster 1: Usuarios nuevos sensibles a precio

Corresponde a usuarios con menor antigüedad, menor consumo y mayor uso de promociones.

Acción recomendada:

* aplicar campañas de retención temprana;
* entregar beneficios de bienvenida;
* usar promociones controladas;
* fomentar el hábito de consumo durante los primeros meses.

### Cluster 2: Usuarios premium leales

Corresponde a usuarios de alto valor, con mayor gasto, alto porcentaje de finalización y menor sensibilidad a promociones.

Acción recomendada:

* entregar beneficios premium;
* ofrecer contenido exclusivo;
* priorizar estrategias de fidelización;
* evitar descuentos innecesarios, ya que son usuarios menos sensibles a promociones.

## 4. Vista Técnica

La vista técnica permite revisar la calidad y justificación del modelo KMeans.

En esta sección se pueden encontrar elementos como:

* método del codo;
* curva de inercia;
* coeficiente Silhouette;
* valor de K seleccionado;
* comparación entre diferentes cantidades de clusters.

El método del codo permite observar cómo disminuye la inercia a medida que aumenta la cantidad de clusters. El objetivo es identificar un punto donde agregar más clusters ya no entrega una mejora significativa.

El coeficiente Silhouette permite evaluar qué tan separados y compactos están los clusters. Un valor más alto indica una mejor separación entre grupos.

Estas métricas ayudan a justificar la selección de 3 clusters para la segmentación final.

## 5. Vista Operativa

La vista operativa permite explorar los datos de manera más detallada.

En esta sección se pueden revisar:

* usuarios segmentados;
* cluster asignado a cada usuario;
* variables asociadas al consumo;
* variables asociadas al perfil del usuario;
* filtros por segmento o cluster;
* comparación de características entre grupos.

Esta vista es útil para analizar casos específicos y comprender cómo se distribuyen los usuarios dentro de cada segmento.

## 6. Indicadores principales

El dashboard permite analizar indicadores relevantes para interpretar los segmentos, tales como:

* horas de consumo mensual promedio;
* gasto mensual promedio;
* cantidad promedio de contenidos vistos;
* sesiones por semana;
* porcentaje de finalización;
* uso de promociones;
* antigüedad promedio;
* dispositivos registrados;
* uso de aplicación móvil;
* interacciones con soporte.

Estos indicadores permiten comparar el comportamiento de los clusters y asignarles una interpretación de negocio.

## 7. Uso de filtros

Cuando el dashboard incluye filtros por cluster, el usuario puede seleccionar uno o más segmentos para revisar sus características específicas.

Esto permite responder preguntas como:

* ¿qué segmento tiene mayor gasto mensual?;
* ¿qué grupo usa más promociones?;
* ¿qué usuarios tienen mayor antigüedad?;
* ¿qué segmento muestra mayor consumo de contenido?;
* ¿qué grupo puede requerir campañas de retención?

## 8. Interpretación de resultados

Los clusters entregados por KMeans no tienen un significado automático. El modelo solo asigna números como Cluster 0, Cluster 1 y Cluster 2.

La interpretación se realiza analizando los promedios y características principales de cada grupo. Por eso, el dashboard combina visualizaciones con descripciones de negocio.

Esto permite transformar el resultado técnico del modelo en información útil para la empresa.

## 9. Recomendaciones de uso

Para usar correctamente el dashboard, se recomienda:

1. revisar primero la vista ejecutiva para entender los segmentos principales;
2. revisar la vista técnica para comprender por qué se eligió el número de clusters;
3. usar la vista operativa para explorar usuarios y características específicas;
4. comparar los indicadores promedio entre segmentos;
5. relacionar cada segmento con acciones de negocio concretas.

## 10. Conclusión

El dashboard permite comunicar los resultados del modelo de segmentación de forma clara y orientada al negocio.

Su principal valor es facilitar que la empresa comprenda mejor a sus usuarios y pueda tomar decisiones diferenciadas para cada segmento, en vez de aplicar una misma estrategia para todos.
