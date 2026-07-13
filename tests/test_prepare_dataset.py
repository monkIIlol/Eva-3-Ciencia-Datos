"""
Pruebas automatizadas para la etapa de preparación del dataset analítico.
Incluye pruebas normales y pruebas negativas/controladas.
"""

import unittest
from pathlib import Path
import json

import numpy as np
import pandas as pd

from etl.prepare_dataset import (
    preparar_dataset,
    limpiar_dataset,
    crear_variables_derivadas,
    validar_columnas_base,
    RUTA_DATASET_MODELO,
    RUTA_KPIS,
    RUTA_REPORTE,
)


def crear_dataframe_prueba():
    """Crea un dataset sintético pequeño con las columnas esperadas."""
    return pd.DataFrame({
        "id_cliente": [1, 2, 3, 4, 5],
        "horas_consumo_mensual": [20, 25, 30, 35, 40],
        "gasto_mensual": [100, 120, 140, 160, 180],
        "cantidad_contenidos_vistos": [10, 12, 14, 16, 18],
        "sesiones_semana": [2, 3, 4, 5, 6],
        "porcentaje_finalizacion": [60, 65, 70, 75, 80],
        "tiempo_promedio_sesion_min": [30, 35, 40, 45, 50],
        "cantidad_generos_consumidos": [2, 3, 4, 5, 6],
        "porcentaje_uso_promociones": [0.1, 0.2, 0.3, 0.4, 0.5],
        "antiguedad_cliente_meses": [12, 24, 36, 48, 60],
        "edad": [20, 25, 30, 35, 40],
        "dispositivos_registrados": [1, 2, 3, 4, 5],
        "porcentaje_uso_app_movil": [0.2, 0.3, 0.4, 0.5, 0.6],
        "cantidad_perfiles_creados": [1, 2, 3, 4, 5],
        "interacciones_mensuales_soporte": [0, 1, 2, 3, 4],
        "distancia_promedio_red_km": [1, 2, 3, 4, 5],
    })


class TestPrepareDataset(unittest.TestCase):
    """Pruebas para validar la preparación del dataset analítico."""

    @classmethod
    def setUpClass(cls):
        cls.df = preparar_dataset()

    def test_archivos_generados(self):
        self.assertTrue(RUTA_DATASET_MODELO.exists())
        self.assertTrue(RUTA_KPIS.exists())
        self.assertTrue(RUTA_REPORTE.exists())

    def test_columnas_derivadas_existen(self):
        columnas_esperadas = [
            "sesiones_mes_estimadas",
            "contenidos_por_sesion",
            "gasto_por_hora",
            "minutos_totales_estimados",
            "soporte_por_dispositivo",
            "generos_por_contenido",
            "engagement_score",
            "nivel_engagement",
            "cliente_antiguo",
            "uso_promociones_alto",
            "valor_cliente",
        ]

        for columna in columnas_esperadas:
            self.assertIn(columna, self.df.columns)

    def test_sin_nulos_en_columnas_derivadas(self):
        columnas_derivadas = [
            "sesiones_mes_estimadas",
            "contenidos_por_sesion",
            "gasto_por_hora",
            "minutos_totales_estimados",
            "soporte_por_dispositivo",
            "generos_por_contenido",
            "engagement_score",
        ]

        self.assertEqual(self.df[columnas_derivadas].isna().sum().sum(), 0)

    def test_sin_infinitos_en_columnas_derivadas(self):
        columnas_derivadas = [
            "contenidos_por_sesion",
            "gasto_por_hora",
            "soporte_por_dispositivo",
            "generos_por_contenido",
        ]

        valores = self.df[columnas_derivadas].to_numpy()
        self.assertFalse(np.isinf(valores).any())

    def test_engagement_score_rango_valido(self):
        self.assertGreaterEqual(self.df["engagement_score"].min(), 0)
        self.assertLessEqual(self.df["engagement_score"].max(), 1)

    def test_nivel_engagement_valores_validos(self):
        valores_validos = {"bajo", "medio", "alto"}
        valores_obtenidos = set(self.df["nivel_engagement"].dropna().unique())
        self.assertTrue(valores_obtenidos.issubset(valores_validos))

    def test_valor_cliente_valores_validos(self):
        valores_validos = {"alto_valor", "valor_medio", "valor_en_riesgo"}
        valores_obtenidos = set(self.df["valor_cliente"].dropna().unique())
        self.assertTrue(valores_obtenidos.issubset(valores_validos))

    def test_kpis_negocio_no_vacio(self):
        kpis = pd.read_csv(RUTA_KPIS)

        self.assertGreater(len(kpis), 0)
        self.assertIn("usuarios", kpis.columns)
        self.assertIn("gasto_promedio", kpis.columns)
        self.assertIn("porcentaje_usuarios", kpis.columns)

    def test_reporte_calidad_contenido(self):
        with open(RUTA_REPORTE, "r", encoding="utf-8") as archivo:
            reporte = json.load(archivo)

        self.assertIn("filas_originales", reporte)
        self.assertIn("filas_finales", reporte)
        self.assertIn("variables_derivadas", reporte)
        self.assertIn("outliers_iqr", reporte)
        self.assertIn("nota_metodologica", reporte)


class TestPrepareDatasetCasosNegativos(unittest.TestCase):
    """Pruebas con datos problemáticos para validar robustez."""

    def test_id_cliente_nulo_se_elimina_no_se_imputa(self):
        df = crear_dataframe_prueba()
        df.loc[2, "id_cliente"] = np.nan

        df_limpio = limpiar_dataset(df)

        self.assertFalse(df_limpio["id_cliente"].isna().any())
        self.assertNotIn(3, df_limpio["id_cliente"].tolist())
        self.assertEqual(len(df_limpio), 4)

    def test_id_cliente_duplicado_se_elimina(self):
        df = crear_dataframe_prueba()
        df.loc[4, "id_cliente"] = 2

        df_limpio = limpiar_dataset(df)

        self.assertFalse(df_limpio["id_cliente"].duplicated().any())
        self.assertEqual(len(df_limpio), 4)

    def test_columna_faltante_lanza_error(self):
        df = crear_dataframe_prueba()
        df = df.drop(columns=["gasto_mensual"])

        with self.assertRaises(ValueError):
            validar_columnas_base(df)

    def test_valores_negativos_se_corrigen(self):
        df = crear_dataframe_prueba()
        df.loc[0, "gasto_mensual"] = -100
        df.loc[1, "antiguedad_cliente_meses"] = -5
        df.loc[2, "distancia_promedio_red_km"] = -20
        df.loc[3, "interacciones_mensuales_soporte"] = -3

        df_limpio = limpiar_dataset(df)

        self.assertGreaterEqual(df_limpio["gasto_mensual"].min(), 0)
        self.assertGreaterEqual(df_limpio["antiguedad_cliente_meses"].min(), 0)
        self.assertGreaterEqual(df_limpio["distancia_promedio_red_km"].min(), 0)
        self.assertGreaterEqual(df_limpio["interacciones_mensuales_soporte"].min(), 0)

    def test_porcentajes_fuera_de_rango_se_corrigen(self):
        df = crear_dataframe_prueba()
        df.loc[0, "porcentaje_finalizacion"] = 150
        df.loc[1, "porcentaje_uso_promociones"] = 2
        df.loc[2, "porcentaje_uso_app_movil"] = -1

        df_limpio = limpiar_dataset(df)

        self.assertLessEqual(df_limpio["porcentaje_finalizacion"].max(), 100)
        self.assertLessEqual(df_limpio["porcentaje_uso_promociones"].max(), 1)
        self.assertGreaterEqual(df_limpio["porcentaje_uso_app_movil"].min(), 0)

    def test_strings_en_columnas_numericas_se_imputan(self):
        df = crear_dataframe_prueba()
        df["gasto_mensual"] = df["gasto_mensual"].astype("object")
        df.loc[0, "gasto_mensual"] = "error"

        df_limpio = limpiar_dataset(df)

        self.assertFalse(df_limpio["gasto_mensual"].isna().any())
        self.assertTrue(pd.api.types.is_numeric_dtype(df_limpio["gasto_mensual"]))

    def test_infinitos_se_reemplazan(self):
        df = crear_dataframe_prueba()
        df["horas_consumo_mensual"] = df["horas_consumo_mensual"].astype("float64")
        df.loc[0, "horas_consumo_mensual"] = np.inf

        df_limpio = limpiar_dataset(df)

        self.assertFalse(np.isinf(df_limpio["horas_consumo_mensual"]).any())

    def test_contenidos_por_sesion_usa_sesiones_mensuales(self):
        df = crear_dataframe_prueba()
        df_limpio = limpiar_dataset(df)
        df_preparado = crear_variables_derivadas(df_limpio)

        esperado = (
            df_preparado["cantidad_contenidos_vistos"]
            / (df_preparado["sesiones_semana"] * 4)
        ).round(3)

        pd.testing.assert_series_equal(
            df_preparado["contenidos_por_sesion"],
            esperado,
            check_names=False,
        )


if __name__ == "__main__":
    unittest.main()