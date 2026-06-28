# Contexto de Negocio

## 1. Descripción del problema

Una empresa de streaming digital busca mejorar la experiencia de sus usuarios mediante estrategias más personalizadas. Actualmente, la plataforma entrega recomendaciones y beneficios de forma general, sin diferenciar claramente los distintos tipos de usuarios que existen dentro del servicio.

Esto genera una oportunidad de mejora, ya que no todos los usuarios tienen el mismo comportamiento. Algunos consumen muchas horas de contenido, otros utilizan promociones con mayor frecuencia, algunos tienen baja antigüedad y otros presentan un comportamiento más estable y leal.

Por esta razón, el objetivo del proyecto es identificar segmentos de usuarios con características similares, utilizando técnicas de aprendizaje no supervisado.

## 2. Necesidad del negocio

La empresa necesita comprender mejor a sus usuarios para tomar decisiones más específicas en áreas como:

* recomendaciones personalizadas de contenido;
* campañas de retención;
* beneficios para usuarios frecuentes;
* estrategias para aumentar el consumo dentro de la plataforma;
* identificación de usuarios sensibles a promociones;
* diferenciación entre usuarios nuevos, habituales y premium.

La segmentación permite pasar de una estrategia general para todos los usuarios a una estrategia diferenciada según el comportamiento de cada grupo.

## 3. Fuentes de datos utilizadas

El proyecto integra información desde dos fuentes principales.

### Fuente 1: `usuarios_streaming.csv`

Esta fuente contiene información relacionada con el consumo de los usuarios dentro de la plataforma.

Incluye variables como:

* horas de consumo mensual;
* gasto mensual;
* cantidad de contenidos vistos;
* sesiones por semana;
* porcentaje de finalización de contenidos;
* tiempo promedio de sesión;
* cantidad de géneros consumidos;
* porcentaje de uso de promociones;
* antigüedad del usuario.

### Fuente 2: `perfil_usuarios.csv`

Esta fuente contiene información complementaria del perfil del usuario. En el proyecto, esta información se considera como una fuente proveniente de una base de datos PostgreSQL.

Incluye variables como:

* edad;
* dispositivos registrados;
* porcentaje de uso desde aplicación móvil;
* cantidad de perfiles creados;
* interacciones mensuales con soporte;
* distancia promedio asociada a la red de conexión.

Ambas fuentes se integran mediante la columna `id_cliente`.

## 4. Importancia de integrar las fuentes

La integración de ambas fuentes permite construir una visión más completa de cada usuario.

Si solo se analiza el consumo, se pueden observar patrones de uso, pero no necesariamente características asociadas al perfil del usuario. Por otro lado, si solo se analiza el perfil, se pierde información importante sobre el comportamiento real dentro de la plataforma.

Al unir ambas fuentes, el modelo puede segmentar considerando tanto el comportamiento de consumo como las características complementarias del usuario.

## 5. Enfoque analítico

El proyecto utiliza aprendizaje no supervisado, específicamente el algoritmo KMeans, para identificar grupos de usuarios con comportamientos similares.

Este enfoque es adecuado porque no existe una etiqueta previa que indique a qué tipo de segmento pertenece cada usuario. El modelo busca patrones en los datos y agrupa usuarios según similitudes entre sus variables.

Antes de entrenar el modelo, se realiza una validación del pipeline ETL para revisar que los datos tengan columnas correctas, no presenten valores nulos, no tengan duplicados y puedan integrarse correctamente.

## 6. Segmentos identificados

A partir del análisis realizado, se trabaja con tres segmentos principales de usuarios.

### Cluster 0: Usuarios habituales exploradores

Este grupo está compuesto por usuarios con sesiones frecuentes, pero de menor duración. Presentan un consumo moderado y tienden a explorar distintos contenidos dentro de la plataforma.

Desde el punto de vista del negocio, este segmento puede abordarse con recomendaciones personalizadas que incentiven una mayor duración de sesión y una mayor finalización de contenidos.

### Cluster 1: Usuarios nuevos sensibles a precio

Este grupo se caracteriza por usuarios con menor antigüedad, menor consumo y mayor uso de promociones.

Desde el punto de vista del negocio, este segmento requiere estrategias de retención temprana. Algunas acciones posibles son beneficios de bienvenida, promociones controladas y recomendaciones iniciales para aumentar el hábito de consumo.

### Cluster 2: Usuarios premium leales

Este grupo presenta mayor gasto, alto porcentaje de finalización y menor sensibilidad a promociones.

Desde el punto de vista del negocio, este segmento representa usuarios de alto valor. Para este grupo se recomienda aplicar estrategias de fidelización, contenido exclusivo, beneficios premium y experiencias personalizadas.

## 7. Valor de negocio esperado

La segmentación permite que la empresa tome decisiones más enfocadas según el tipo de usuario.

Algunos beneficios esperados son:

* mejorar la personalización de recomendaciones;
* aumentar la retención de usuarios nuevos;
* identificar usuarios de alto valor;
* optimizar campañas promocionales;
* evitar entregar descuentos innecesarios a usuarios que ya presentan alta lealtad;
* apoyar decisiones de marketing, producto y experiencia de usuario.

## 8. Conclusión del contexto

El proyecto busca transformar datos dispersos de la plataforma en información útil para la toma de decisiones. Mediante la integración de fuentes, validación de datos, aplicación de KMeans y visualización en dashboard, la empresa puede comprender mejor a sus usuarios y diseñar estrategias diferenciadas para cada segmento.
