import unittest

import pandas as pd

from config.features import KMEANS_FEATURES
from etl.contracts import ModelTrainingError
from model.train import entrenar_kmeans


class TestModeloKMeans(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataset = pd.read_csv(
            "data/dataset_base_limpio.csv"
        )

        cls.resultado = entrenar_kmeans(
            cls.dataset,
            persistir=False,
        )

    def test_entrenamiento_genera_trescientos_usuarios(self):
        self.assertEqual(
            len(self.resultado.usuarios_segmentados),
            300,
        )

    def test_k_optimo_es_valido(self):
        self.assertIn(
            self.resultado.metricas["k_optimo"],
            range(2, 11),
        )

    def test_columnas_cluster_y_pca(self):
        columnas = (
            self.resultado.usuarios_segmentados.columns
        )

        self.assertIn("cluster", columnas)
        self.assertIn("pc1", columnas)
        self.assertIn("pc2", columnas)

    def test_centroides_usan_features_oficiales(self):
        self.assertEqual(
            self.resultado.centroides.columns.tolist(),
            KMEANS_FEATURES,
        )

    def test_scaler_usa_features_oficiales(self):
        self.assertEqual(
            self.resultado.scaler.feature_names_in_.tolist(),
            KMEANS_FEATURES,
        )

    def test_no_acepta_feature_faltante(self):
        dataset_invalido = self.dataset.drop(
            columns=[KMEANS_FEATURES[0]]
        )

        with self.assertRaises(ModelTrainingError):
            entrenar_kmeans(
                dataset_invalido,
                persistir=False,
            )


if __name__ == "__main__":
    unittest.main()