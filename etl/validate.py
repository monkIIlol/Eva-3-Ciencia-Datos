"""
Validación de fuentes de datos para el pipeline ETL.

Este script revisa:
- columnas esperadas;
- valores nulos;
- id_cliente duplicados;
- tipos numéricos;
- consistencia de id_cliente entre fuentes;
- correcta integración entre usuarios_streaming y perfil_usuarios.
"""

import pandas as pd
from pathlib import Path


COLUMNAS_STREAMING = [
    "id_cliente",
    "horas_consumo_mensual",
    "gasto_mensual",
    "cantidad_contenidos_vistos",
    "sesiones_semana",
    "porcentaje_finalizacion",
    "tiempo_promedio_sesion_min",
    "cantidad_generos_consumidos",
    "porcentaje_uso_promociones",
    "antiguedad_cliente_meses",
]

COLUMNAS_PERFIL = [
    "id_cliente",
    "edad",
    "dispositivos_registrados",
    "porcentaje_uso_app_movil",
    "cantidad_perfiles_creados",
    "interacciones_mensuales_soporte",
    "distancia_promedio_red_km",
]


def cargar_csv(ruta):
    ruta = Path(ruta)

    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {ruta}")

    return pd.read_csv(ruta)


def validar_columnas(df, columnas_esperadas, nombre_fuente):
    columnas_faltantes = [col for col in columnas_esperadas if col not in df.columns]

    if columnas_faltantes:
        raise ValueError(
            f"La fuente {nombre_fuente} no tiene estas columnas: {columnas_faltantes}"
        )

    print(f"[OK] {nombre_fuente}: columnas esperadas encontradas.")


def validar_nulos(df, nombre_fuente):
    nulos = df.isnull().sum()
    columnas_con_nulos = nulos[nulos > 0]

    if not columnas_con_nulos.empty:
        raise ValueError(
            f"La fuente {nombre_fuente} tiene valores nulos:\n{columnas_con_nulos}"
        )

    print(f"[OK] {nombre_fuente}: no contiene valores nulos.")


def validar_duplicados_id(df, nombre_fuente):
    duplicados = df["id_cliente"].duplicated().sum()

    if duplicados > 0:
        raise ValueError(
            f"La fuente {nombre_fuente} tiene {duplicados} id_cliente duplicados."
        )

    print(f"[OK] {nombre_fuente}: no tiene id_cliente duplicados.")


def validar_tipos_numericos(df, columnas_esperadas, nombre_fuente):
    for columna in columnas_esperadas:
        if not pd.api.types.is_numeric_dtype(df[columna]):
            raise TypeError(
                f"La columna {columna} de {nombre_fuente} no es numérica."
            )

    print(f"[OK] {nombre_fuente}: todas las columnas esperadas son numéricas.")


def validar_integracion(df_streaming, df_perfil):
    ids_streaming = set(df_streaming["id_cliente"])
    ids_perfil = set(df_perfil["id_cliente"])

    ids_solo_streaming = ids_streaming - ids_perfil
    ids_solo_perfil = ids_perfil - ids_streaming

    if ids_solo_streaming:
        raise ValueError(
            "Existen id_cliente en usuarios_streaming.csv que no están en perfil_usuarios.csv."
        )

    if ids_solo_perfil:
        raise ValueError(
            "Existen id_cliente en perfil_usuarios.csv que no están en usuarios_streaming.csv."
        )

    print("[OK] Integración: los id_cliente coinciden entre ambas fuentes.")


def ejecutar_validaciones():
    print("Iniciando validación del pipeline ETL...\n")

    df_streaming = cargar_csv("data/usuarios_streaming.csv")
    df_perfil = cargar_csv("database/perfil_usuarios.csv")

    validar_columnas(df_streaming, COLUMNAS_STREAMING, "usuarios_streaming.csv")
    validar_columnas(df_perfil, COLUMNAS_PERFIL, "perfil_usuarios.csv")

    validar_nulos(df_streaming, "usuarios_streaming.csv")
    validar_nulos(df_perfil, "perfil_usuarios.csv")

    validar_duplicados_id(df_streaming, "usuarios_streaming.csv")
    validar_duplicados_id(df_perfil, "perfil_usuarios.csv")

    validar_tipos_numericos(df_streaming, COLUMNAS_STREAMING, "usuarios_streaming.csv")
    validar_tipos_numericos(df_perfil, COLUMNAS_PERFIL, "perfil_usuarios.csv")

    validar_integracion(df_streaming, df_perfil)

    df_integrado = df_streaming.merge(df_perfil, on="id_cliente", how="inner")

    print("\n[OK] Dataset integrado validado correctamente.")
    print(f"Filas finales: {df_integrado.shape[0]}")
    print(f"Columnas finales: {df_integrado.shape[1]}")

    return df_integrado


if __name__ == "__main__":
    ejecutar_validaciones()