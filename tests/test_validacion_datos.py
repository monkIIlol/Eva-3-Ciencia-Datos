"""
Pruebas automatizadas para validar las fuentes de datos del proyecto.

Estas pruebas revisan que:
- los archivos existan;
- tengan las columnas esperadas;
- no tengan valores nulos;
- no tengan id_cliente duplicados;
- los id_cliente coincidan entre ambas fuentes;
- el dataset integrado se pueda construir correctamente.
"""

import unittest
from pathlib import Path

import pandas as pd

from etl.validate import (
    COLUMNAS_STREAMING,
    COLUMNAS_PERFIL,
    cargar_csv,
    validar_columnas,
    validar_nulos,
    validar_duplicados_id,
    validar_tipos_numericos,
    validar_integracion,
    ejecutar_validaciones,
)


class TestValidacionDatos(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ruta_streaming = Path("data/usuarios_streaming.csv")
        cls.ruta_perfil = Path("database/perfil_usuarios.csv")

        cls.df_streaming = pd.read_csv(cls.ruta_streaming)
        cls.df_perfil = pd.read_csv(cls.ruta_perfil)

    def test_archivos_existen(self):
        self.assertTrue(self.ruta_streaming.exists())
        self.assertTrue(self.ruta_perfil.exists())

    def test_columnas_streaming(self):
        validar_columnas(
            self.df_streaming,
            COLUMNAS_STREAMING,
            "usuarios_streaming.csv"
        )

    def test_columnas_perfil(self):
        validar_columnas(
            self.df_perfil,
            COLUMNAS_PERFIL,
            "perfil_usuarios.csv"
        )

    def test_no_hay_nulos(self):
        validar_nulos(self.df_streaming, "usuarios_streaming.csv")
        validar_nulos(self.df_perfil, "perfil_usuarios.csv")

    def test_no_hay_id_cliente_duplicados(self):
        validar_duplicados_id(self.df_streaming, "usuarios_streaming.csv")
        validar_duplicados_id(self.df_perfil, "perfil_usuarios.csv")

    def test_tipos_numericos(self):
        validar_tipos_numericos(
            self.df_streaming,
            COLUMNAS_STREAMING,
            "usuarios_streaming.csv"
        )

        validar_tipos_numericos(
            self.df_perfil,
            COLUMNAS_PERFIL,
            "perfil_usuarios.csv"
        )

    def test_integracion_ids(self):
        validar_integracion(self.df_streaming, self.df_perfil)

    def test_dataset_integrado(self):
        df_integrado = ejecutar_validaciones()

        self.assertFalse(df_integrado.empty)
        self.assertIn("id_cliente", df_integrado.columns)
        self.assertEqual(df_integrado["id_cliente"].duplicated().sum(), 0)
        self.assertEqual(len(df_integrado), len(self.df_streaming))


if __name__ == "__main__":
    unittest.main()