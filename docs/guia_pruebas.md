# Guía de Validación y Pruebas

## 1. Objetivo

Esta guía describe cómo ejecutar la validación de datos y las pruebas automatizadas del proyecto de segmentación de usuarios de streaming.

El objetivo es comprobar que las fuentes de datos utilizadas por el pipeline ETL sean consistentes antes de aplicar el modelo de clustering KMeans.

## 2. Archivos relacionados

Los archivos principales asociados a esta etapa son:

* `etl/validate.py`: script encargado de validar las fuentes de datos.
* `tests/test_validacion_datos.py`: pruebas automatizadas para comprobar que la validación funcione correctamente.
* `data/usuarios_streaming.csv`: fuente de datos de consumo de usuarios.
* `database/perfil_usuarios.csv`: fuente de datos complementaria de perfil de usuarios.

## 3. Validación del pipeline ETL

Para ejecutar la validación de datos, se debe usar el siguiente comando desde la raíz del proyecto:

```bash
py -3.11 etl/validate.py
```

Este script revisa:

* existencia de las columnas esperadas;
* valores nulos;
* duplicados en `id_cliente`;
* tipos de datos numéricos;
* coincidencia de usuarios entre ambas fuentes;
* correcta integración del dataset final.

Si la validación es correcta, se espera una salida similar a:

```text
[OK] usuarios_streaming.csv: columnas esperadas encontradas.
[OK] perfil_usuarios.csv: columnas esperadas encontradas.
[OK] usuarios_streaming.csv: no contiene valores nulos.
[OK] perfil_usuarios.csv: no contiene valores nulos.
[OK] usuarios_streaming.csv: no tiene id_cliente duplicados.
[OK] perfil_usuarios.csv: no tiene id_cliente duplicados.
[OK] usuarios_streaming.csv: todas las columnas esperadas son numéricas.
[OK] perfil_usuarios.csv: todas las columnas esperadas son numéricas.
[OK] Integración: los id_cliente coinciden entre ambas fuentes.

[OK] Dataset integrado validado correctamente.
Filas finales: 300
Columnas finales: 16
```

## 4. Pruebas automatizadas

Para ejecutar las pruebas automatizadas, se debe usar el siguiente comando:

```bash
py -3.11 -m unittest tests/test_validacion_datos.py
```

Estas pruebas verifican que:

* los archivos de datos existan;
* las columnas esperadas estén presentes;
* no existan valores nulos;
* no existan `id_cliente` duplicados;
* las variables esperadas sean numéricas;
* los identificadores de usuarios coincidan entre ambas fuentes;
* el dataset integrado pueda construirse correctamente.

Si las pruebas son exitosas, se espera una salida similar a:

```text
Ran 8 tests in 0.035s

OK
```

## 5. Importancia dentro del proyecto

La validación y las pruebas automatizadas permiten detectar problemas antes de ejecutar el modelo KMeans.

Esto es importante porque el modelo de clustering trabaja con distancias entre variables numéricas. Si los datos tuvieran columnas faltantes, nulos, duplicados o errores de integración, los resultados de la segmentación podrían ser incorrectos.

Por esta razón, esta etapa ayuda a mejorar la robustez del pipeline ETL y aumenta la confiabilidad de los resultados utilizados en el dashboard.

## 6. Interpretación del resultado

Cuando la validación y las pruebas muestran `OK`, significa que las fuentes de datos cumplen con las condiciones mínimas para continuar con el entrenamiento del modelo.

Esto no significa que el modelo sea automáticamente perfecto, pero sí indica que los datos de entrada fueron revisados correctamente antes de ser utilizados en la segmentación.
