"""
Pruebas automatizadas para la etapa de preparación del dataset analítico.

Estas pruebas verifican que el módulo etl/prepare_dataset.py genere correctamente
el dataset de modelado, los KPIs de negocio y el reporte de calidad.
"""

import unittest
from pathlib import Path
import json

import numpy as np
import pandas as pd

from etl.prepare_dataset import (
    preparar_dataset,
    RUTA_DATASET_MODELO,
    RUTA_KPIS,
    RUTA_REPORTE,
)


class TestPrepareDataset(unittest.TestCase):
    """Pruebas para validar la preparación del dataset analítico."""

    @classmethod
    def setUpClass(cls):
        from pathlib import Path
        import unittest

        ruta_consolidada = Path("data/data_consolidada.csv")

        if not ruta_consolidada.exists():
            raise unittest.SkipTest(
                "No existe data/data_consolidada.csv en el entorno de CI. "
                "Se omiten estas pruebas porque dependen de un artefacto generado por el pipeline."
            )

        cls.df = preparar_dataset()

    def test_archivos_generados(self):
        """Verifica que los archivos de salida se generen correctamente."""
        self.assertTrue(RUTA_DATASET_MODELO.exists())
        self.assertTrue(RUTA_KPIS.exists())
        self.assertTrue(RUTA_REPORTE.exists())

    def test_columnas_derivadas_existen(self):
        """Verifica que las variables creadas por feature engineering existan."""
        columnas_esperadas = [
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
        """Verifica que las variables derivadas no tengan valores nulos."""
        columnas_derivadas = [
            "contenidos_por_sesion",
            "gasto_por_hora",
            "minutos_totales_estimados",
            "soporte_por_dispositivo",
            "generos_por_contenido",
            "engagement_score",
        ]

        total_nulos = self.df[columnas_derivadas].isna().sum().sum()
        self.assertEqual(total_nulos, 0)

    def test_sin_infinitos_en_columnas_derivadas(self):
        """Verifica que no existan valores infinitos por divisiones."""
        columnas_derivadas = [
            "contenidos_por_sesion",
            "gasto_por_hora",
            "soporte_por_dispositivo",
            "generos_por_contenido",
        ]

        valores = self.df[columnas_derivadas].to_numpy()
        self.assertFalse(np.isinf(valores).any())

    def test_engagement_score_rango_valido(self):
        """Verifica que engagement_score esté entre 0 y 1."""
        self.assertGreaterEqual(self.df["engagement_score"].min(), 0)
        self.assertLessEqual(self.df["engagement_score"].max(), 1)

    def test_nivel_engagement_valores_validos(self):
        """Verifica que nivel_engagement solo tenga categorías esperadas."""
        valores_validos = {"bajo", "medio", "alto"}
        valores_obtenidos = set(self.df["nivel_engagement"].dropna().unique())

        self.assertTrue(valores_obtenidos.issubset(valores_validos))

    def test_valor_cliente_valores_validos(self):
        """Verifica que valor_cliente solo tenga categorías esperadas."""
        valores_validos = {"alto_valor", "valor_medio", "valor_en_riesgo"}
        valores_obtenidos = set(self.df["valor_cliente"].dropna().unique())

        self.assertTrue(valores_obtenidos.issubset(valores_validos))

    def test_kpis_negocio_no_vacio(self):
        """Verifica que el archivo de KPIs tenga datos."""
        kpis = pd.read_csv(RUTA_KPIS)

        self.assertGreater(len(kpis), 0)
        self.assertIn("usuarios", kpis.columns)
        self.assertIn("gasto_promedio", kpis.columns)
        self.assertIn("porcentaje_usuarios", kpis.columns)

    def test_reporte_calidad_contenido(self):
        """Verifica que el reporte de calidad tenga información relevante."""
        with open(RUTA_REPORTE, "r", encoding="utf-8") as archivo:
            reporte = json.load(archivo)

        self.assertIn("filas_originales", reporte)
        self.assertIn("filas_finales", reporte)
        self.assertIn("variables_derivadas", reporte)
        self.assertGreater(len(reporte["variables_derivadas"]), 0)


if __name__ == "__main__":
    unittest.main()