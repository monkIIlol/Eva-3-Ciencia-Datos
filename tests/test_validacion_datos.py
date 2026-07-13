"""
Pruebas del contrato de validación e integración.

Estas pruebas utilizan archivos locales como fixtures controladas.
El flujo productivo continúa extrayendo los perfiles desde PostgreSQL.
"""

import unittest
from pathlib import Path

import pandas as pd

from etl.contracts import (
    BASE_COLUMN_ORDER,
    DataIntegrationError,
)
from etl.integrate import integrar_fuentes
from etl.validate import validar_fuentes


class TestValidacionDatos(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ruta_streaming = Path(
            "data/usuarios_streaming.csv"
        )
        cls.ruta_perfil = Path(
            "database/perfil_usuarios.csv"
        )

        cls.df_streaming = pd.read_csv(
            cls.ruta_streaming
        )
        cls.df_perfil = pd.read_csv(
            cls.ruta_perfil
        )

    def test_archivos_fixture_existen(self):
        self.assertTrue(
            self.ruta_streaming.exists()
        )
        self.assertTrue(
            self.ruta_perfil.exists()
        )

    def test_fuentes_validas_no_generan_errores(self):
        reporte = validar_fuentes(
            self.df_streaming,
            self.df_perfil,
        )

        self.assertTrue(reporte.is_valid)
        self.assertEqual(len(reporte.errors), 0)

    def test_columna_requerida_faltante_es_error(self):
        streaming_invalido = (
            self.df_streaming.drop(
                columns=["gasto_mensual"]
            )
        )

        reporte = validar_fuentes(
            streaming_invalido,
            self.df_perfil,
        )

        codigos = [
            error.code
            for error in reporte.errors
        ]

        self.assertFalse(reporte.is_valid)
        self.assertIn(
            "MISSING_REQUIRED_COLUMNS",
            codigos,
        )

    def test_nulo_analitico_es_advertencia(self):
        streaming_con_nulo = (
            self.df_streaming.copy()
        )

        streaming_con_nulo.loc[
            0,
            "gasto_mensual",
        ] = None

        reporte = validar_fuentes(
            streaming_con_nulo,
            self.df_perfil,
        )

        codigos_warning = [
            warning.code
            for warning in reporte.warnings
        ]

        self.assertTrue(reporte.is_valid)
        self.assertIn(
            "NULL_VALUES",
            codigos_warning,
        )

    def test_id_duplicado_es_error(self):
        perfil_duplicado = pd.concat(
            [
                self.df_perfil,
                self.df_perfil.iloc[[0]],
            ],
            ignore_index=True,
        )

        reporte = validar_fuentes(
            self.df_streaming,
            perfil_duplicado,
        )

        codigos = [
            error.code
            for error in reporte.errors
        ]

        self.assertFalse(reporte.is_valid)
        self.assertIn(
            "DUPLICATED_CUSTOMER_ID",
            codigos,
        )

    def test_ids_incompatibles_generan_error(self):
        perfil_incompatible = (
            self.df_perfil.copy()
        )

        perfil_incompatible.loc[
            0,
            "id_cliente",
        ] = 999999

        reporte = validar_fuentes(
            self.df_streaming,
            perfil_incompatible,
        )

        codigos = [
            error.code
            for error in reporte.errors
        ]

        self.assertFalse(reporte.is_valid)
        self.assertIn(
            "CUSTOMER_ID_MISMATCH",
            codigos,
        )

    def test_integracion_valida(self):
        reporte = validar_fuentes(
            self.df_streaming,
            self.df_perfil,
        )

        resultado = integrar_fuentes(
            df_streaming=self.df_streaming,
            df_perfil=self.df_perfil,
            validation_report=reporte,
        )

        self.assertEqual(
            resultado.dataframe.shape,
            (300, 16),
        )

        self.assertEqual(
            resultado.dataframe.columns.tolist(),
            BASE_COLUMN_ORDER,
        )

        self.assertEqual(
            resultado.dataframe[
                "id_cliente"
            ].duplicated().sum(),
            0,
        )

        self.assertEqual(
            resultado.metadata["cardinalidad"],
            "one_to_one",
        )

    def test_integracion_rechaza_reporte_invalido(self):
        streaming_invalido = (
            self.df_streaming.drop(
                columns=["gasto_mensual"]
            )
        )

        reporte = validar_fuentes(
            streaming_invalido,
            self.df_perfil,
        )

        with self.assertRaises(
            DataIntegrationError
        ):
            integrar_fuentes(
                df_streaming=streaming_invalido,
                df_perfil=self.df_perfil,
                validation_report=reporte,
            )


if __name__ == "__main__":
    unittest.main()