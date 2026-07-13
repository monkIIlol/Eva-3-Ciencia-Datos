import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from config.features import (
    CLASSIFICATION_FEATURES,
    CLASSIFICATION_TARGET_SOURCE_COLUMNS,
    REGRESSION_FEATURES,
)
from etl.contracts import ModelTrainingError
from model.train_supervisado import entrenar_modelos_supervisados


class TestModeloSupervisado(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        ruta = Path("data/dataset_base_limpio.csv")
        if not ruta.exists():
            raise unittest.SkipTest(
                "data/dataset_base_limpio.csv no existe. Ejecuta la preparación."
            )
        cls.data = pd.read_csv(ruta)
        cls.resultado = entrenar_modelos_supervisados(
            cls.data,
            persistir=False,
        )

    def test_ambos_modelos_entrenan(self):
        self.assertIsNotNone(self.resultado.regresion.modelo)
        self.assertIsNotNone(self.resultado.clasificacion.modelo)

    def test_regresion_usa_features_oficiales(self):
        self.assertEqual(
            self.resultado.regresion.modelo.feature_names_in_.tolist(),
            REGRESSION_FEATURES,
        )

    def test_clasificacion_usa_features_oficiales(self):
        self.assertEqual(
            self.resultado.clasificacion.modelo.feature_names_in_.tolist(),
            CLASSIFICATION_FEATURES,
        )

    def test_clasificacion_no_usa_columnas_del_target(self):
        interseccion = set(CLASSIFICATION_FEATURES) & set(
            CLASSIFICATION_TARGET_SOURCE_COLUMNS
        )
        self.assertEqual(interseccion, set())

    def test_ganador_regresion_se_elige_por_cv(self):
        metricas = self.resultado.metricas["regresion"]
        candidatos = {
            nombre: valores
            for nombre, valores in metricas.items()
            if isinstance(valores, dict) and "cv_r2_mean" in valores
        }
        esperado = max(candidatos, key=lambda n: candidatos[n]["cv_r2_mean"])
        self.assertEqual(metricas["mejor_modelo"], esperado)

    def test_ganador_clasificacion_se_elige_por_cv(self):
        metricas = self.resultado.metricas["clasificacion"]
        candidatos = {
            nombre: valores
            for nombre, valores in metricas.items()
            if isinstance(valores, dict) and "cv_f1_mean" in valores
        }
        esperado = max(candidatos, key=lambda n: candidatos[n]["cv_f1_mean"])
        self.assertEqual(metricas["mejor_modelo"], esperado)

    def test_umbrales_se_calculan_solo_con_train(self):
        metricas = self.resultado.metricas["clasificacion"]
        self.assertEqual(metricas["origen_umbrales"], "solo_conjunto_train")

    def test_predicciones_regresion_son_finitas(self):
        predicciones = self.resultado.regresion.predicciones_test["valor_predicho"]
        self.assertTrue(np.isfinite(predicciones).all())
        self.assertEqual(len(predicciones), 60)

    def test_predicciones_clasificacion_son_validas(self):
        pred = self.resultado.clasificacion.predicciones_test
        self.assertTrue(set(pred["valor_predicho"]).issubset({0, 1}))
        self.assertTrue(pred["probabilidad_riesgo"].between(0, 1).all())
        self.assertEqual(len(pred), 60)

    def test_rechaza_feature_faltante(self):
        invalido = self.data.drop(columns=[REGRESSION_FEATURES[0]])
        with self.assertRaises(ModelTrainingError):
            entrenar_modelos_supervisados(invalido, persistir=False)


if __name__ == "__main__":
    unittest.main()