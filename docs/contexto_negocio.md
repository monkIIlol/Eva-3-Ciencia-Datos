# Contexto de Negocio y Segmentación de Usuarios

## 1. ¿Cuál es el problema que queremos resolver?
La empresa de streaming ha observado un aumento en la tasa de abandono de sus clientes, lo que afecta sus ingresos y eleva los costos de adquisición de nuevos usuarios. Para enfrentar este problema, busca comprender el comportamiento de sus clientes mediante técnicas de segmentación que permitan diseñar estrategias de retención, personalizar campañas y ofrecer recomendaciones más efectivas.
* Dar recomendaciones de contenido personalizadas según lo que realmente ven.
* Crear campañas de retención enfocadas en los usuarios que están pensando en dejar el servicio.
* Armar promociones o beneficios exclusivos para los usuarios más fieles y que más gastan.

El desafío técnico es que la información está repartida. Los datos de lo que consume el usuario están en un archivo, y los datos de su perfil están en otro sistema. Por eso, el primer paso antes de meter cualquier modelo es juntar y ordenar todo en un solo lugar.

---

## 2. Las fuentes de datos que usamos

Unimos toda la información usando el `id_cliente` como llave principal, cruzando estas dos fuentes:

### Datos de consumo (`usuarios_streaming.csv`)
Tiene el registro mensual de cómo usan la plataforma:
* **Horas de consumo y contenidos vistos:** Para saber qué tan activos son.
* **Gasto mensual:** Cuánto dinero le ingresa a la empresa por ese usuario.
* **Sesiones por semana y duración promedio:** Qué tan seguido entran y cuánto tiempo se quedan.
* **Porcentaje de finalización:** Si terminan las series y películas o las dejan a medias.
* **Géneros consumidos y uso de promociones:** Qué tan variados son sus gustos y si buscan siempre los descuentos.
* **Antigüedad:** Cuántos meses llevan siendo clientes.

### Perfil de usuario (`perfil_usuarios.csv`)
Son los datos que vienen desde la base de datos de Postgres y nos dan el contexto de quién es el usuario:
* **Edad:** Para segmentar por rango de edad.
* **Dispositivos y uso de la app móvil:** Para saber si ven streaming en el teléfono, la tele o el computador.
* **Perfiles creados:** Para identificar si la cuenta la usa una sola persona o una familia.
* **Interacciones con soporte:** Cuántas veces han pedido ayuda por problemas técnicos.
* **Distancia a la red:** Un dato técnico para evaluar si su conexión podría tener problemas de velocidad o latencia.

---

## 3. ¿Qué es lo que hace nuestra solución?
Para cumplir con lo que pide el negocio, el proyecto se divide en tres partes:
1. **Pipeline ETL:** Un script automático que lee Postgres y el CSV,valida la calidad de los datos detectando registros nulos y duplicados antes de consolidarlos, y los une de forma limpia en el archivo final `data/data_consolidada.csv`.
2. **Modelo KMeans:** Tomamos las variables, las escalamos para que los rangos no alteren los resultados y entrenamos el modelo usando el método del codo y Silhouette. Así definimos que el número ideal son 3 grupos de usuarios.
3. **Dashboard en Streamlit:** Una app interactiva con tres pestañas pensadas para distintas personas de la empresa: una vista rápida con gráficos de torta y negocio para los jefes (Vista Ejecutiva), una con las métricas del modelo (Vista Técnica) y otra para revisar las tablas a fondo (Vista Operativa).