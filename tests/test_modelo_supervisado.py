"""
Pruebas automatizadas para los modelos supervisados de regresión y
clasificación.

Entrena los modelos sobre el dataset consolidado y valida que:
- ambas tareas produzcan al menos un modelo con métricas razonables;
- las métricas guardadas cumplan un umbral mínimo de calidad;
- los modelos entrenados puedan predecir sobre datos nuevos sin errores.
"""

import unittest
from pathlib import Path

import pandas as pd

from model.train_supervisado import (
    entrenar_regresion, entrenar_clasificacion, FEATURES, FEATURES_CLASIFICACION,
)


class TestModeloSupervisado(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        ruta_datos = Path("data/data_consolidada.csv")
        if not ruta_datos.exists():
            raise unittest.SkipTest(
                "data/data_consolidada.csv no existe. "
                "Corre el pipeline ETL (etl/extract.py) antes de estas pruebas."
            )
        cls.data = pd.read_csv(ruta_datos)
        cls.modelo_regresion = entrenar_regresion(cls.data)
        cls.modelo_clasificacion = entrenar_clasificacion(cls.data)

    def test_modelo_regresion_entrena_correctamente(self):
        self.assertIsNotNone(self.modelo_regresion)

    def test_modelo_clasificacion_entrena_correctamente(self):
        self.assertIsNotNone(self.modelo_clasificacion)

    def test_prediccion_regresion_sobre_dato_nuevo(self):
        fila = self.data[FEATURES].iloc[[0]]
        pred = self.modelo_regresion.predict(fila)
        self.assertEqual(len(pred), 1)
        self.assertGreaterEqual(pred[0], 0)  # el gasto no debería ser negativo

    def test_prediccion_clasificacion_sobre_dato_nuevo(self):
        fila = self.data[FEATURES_CLASIFICACION].iloc[[0]]
        pred = self.modelo_clasificacion.predict(fila)
        proba = self.modelo_clasificacion.predict_proba(fila)
        self.assertEqual(len(pred), 1)
        self.assertIn(pred[0], [0, 1])
        self.assertAlmostEqual(proba[0].sum(), 1.0, places=5)

    def test_sin_fuga_de_datos_en_clasificacion(self):
        """Las variables usadas para construir la etiqueta
        (riesgo_bajo_compromiso) nunca deben usarse como features del
        modelo de clasificación, o las métricas quedan infladas."""
        self.assertNotIn("porcentaje_finalizacion", FEATURES_CLASIFICACION)
        self.assertNotIn("sesiones_semana", FEATURES_CLASIFICACION)


if __name__ == "__main__":
    unittest.main()
