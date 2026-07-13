"""Pruebas de integración real del pipeline sin depender de PostgreSQL."""
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sqlalchemy.sql.elements import quoted_name

from config.features import KMEANS_FEATURES
from etl.integrate import integrar_fuentes
from etl.validate import validar_fuentes
from pipeline.run import ejecutar_pipeline


class TestPipelineIntegrado(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.streaming = pd.read_csv(
            Path("data/usuarios_streaming.csv")
        )
        cls.perfil = pd.read_csv(
            Path("database/perfil_usuarios.csv")
        )

        with patch(
            "pipeline.run.extraer_fuentes",
            return_value=(cls.streaming, cls.perfil),
        ):
            cls.resultado = ejecutar_pipeline(
                persistir=False
            )

    def test_conserva_los_trescientos_usuarios(self):
        self.assertEqual(
            self.resultado
            .preparation_result
            .dataset_base_limpio
            .shape,
            (300, 16),
        )
        self.assertEqual(
            len(
                self.resultado
                .kmeans_result
                .usuarios_segmentados
            ),
            300,
        )

    def test_validacion_e_integracion_aprobadas(self):
        self.assertTrue(
            self.resultado.validation_report.is_valid
        )
        self.assertEqual(
            self.resultado
            .integration_result
            .metadata["cardinalidad"],
            "one_to_one",
        )

    def test_modelos_supervisados_consumen_dataset_base(self):
        metadata = (
            self.resultado
            .supervised_result
            .metricas["metadata"]
        )

        self.assertEqual(
            metadata["dataset_entrada"],
            "dataset_base_limpio",
        )
        self.assertEqual(
            metadata["n_filas"],
            300,
        )

    def test_kmeans_admite_inferencia_tipo_api(self):
        """
        Garantiza compatibilidad entre el dtype usado al entrenar y el
        dtype generado por una solicitud JSON de la API.
        """
        base = (
            self.resultado
            .preparation_result
            .dataset_base_limpio
        )

        fila = pd.DataFrame(
            [
                {
                    feature: float(base.iloc[0][feature])
                    for feature in KMEANS_FEATURES
                }
            ],
            columns=KMEANS_FEATURES,
        )

        kmeans_result = self.resultado.kmeans_result
        escalada = kmeans_result.scaler.transform(fila)
        escalada = np.asarray(
            escalada,
            dtype=kmeans_result.modelo.cluster_centers_.dtype,
        )
        prediccion = kmeans_result.modelo.predict(escalada)

        self.assertEqual(
            kmeans_result.modelo.cluster_centers_.dtype,
            np.dtype("float64"),
        )
        self.assertEqual(prediccion.shape, (1,))

    def test_manifest_indica_ejecucion_completa(self):
        manifiesto = self.resultado.manifest

        self.assertEqual(
            manifiesto["status"],
            "completed",
        )
        self.assertEqual(
            manifiesto["publication"]["strategy"],
            "deferred_until_all_stages_pass",
        )

    def test_normaliza_columnas_quoted_name_de_sqlalchemy(self):
        """
        Reproduce el tipo de columna entregado por pd.read_sql_table.

        Este caso no aparecía en los fixtures CSV y fue detectado al
        ejecutar el pipeline real contra PostgreSQL.
        """
        perfil_sqlalchemy = self.perfil.copy()
        perfil_sqlalchemy.columns = [
            quoted_name(
                str(columna),
                quote=False,
            )
            for columna in perfil_sqlalchemy.columns
        ]

        reporte = validar_fuentes(
            self.streaming,
            perfil_sqlalchemy,
        )

        resultado = integrar_fuentes(
            df_streaming=self.streaming,
            df_perfil=perfil_sqlalchemy,
            validation_report=reporte,
        )

        self.assertTrue(
            all(
                type(columna) is str
                for columna
                in resultado.dataframe.columns
            )
        )

        # Comprueba exactamente la operación que antes fallaba.
        StandardScaler().fit(
            resultado.dataframe[KMEANS_FEATURES]
        )


if __name__ == "__main__":
    unittest.main()
