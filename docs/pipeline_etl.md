# Pipeline ETL y preparación del dataset analítico

## Objetivo

El objetivo del pipeline es integrar datos de consumo y perfil de usuarios, validar su calidad y preparar un dataset final para modelos de Machine Learning.

## Flujo del pipeline

```text
usuarios_streaming.csv + perfil_usuarios
↓
etl/extract.py
↓
data/data_consolidada.csv
↓
etl/prepare_dataset.py
↓
data/dataset_modelo.csv
data/kpis_negocio.csv
data/reporte_calidad.json
↓
modelos ML + API + dashboard