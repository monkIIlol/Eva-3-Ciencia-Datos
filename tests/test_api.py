import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from config.features import (
    CLASSIFICATION_FEATURES,
    CLASSIFICATION_TARGET_SOURCE_COLUMNS,
    KMEANS_FEATURES,
    REGRESSION_FEATURES,
)


class TestAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not Path("models/modelo_kmeans.pkl").exists():
            raise unittest.SkipTest(
                "models/modelo_kmeans.pkl no existe. Ejecuta model/train.py."
            )

        from api.main import app

        cls.client = TestClient(app)
        cls.muestra = {
            "horas_consumo_mensual": 40,
            "gasto_mensual": 150,
            "cantidad_contenidos_vistos": 20,
            "sesiones_semana": 5,
            "porcentaje_finalizacion": 55,
            "tiempo_promedio_sesion_min": 100,
            "cantidad_generos_consumidos": 5,
            "porcentaje_uso_promociones": 0.3,
            "antiguedad_cliente_meses": 30,
            "edad": 40,
            "dispositivos_registrados": 2,
            "porcentaje_uso_app_movil": 0.5,
            "cantidad_perfiles_creados": 3,
            "interacciones_mensuales_soporte": 2,
            "distancia_promedio_red_km": 20,
        }

        cls.payload_kmeans = {
            campo: cls.muestra[campo]
            for campo in KMEANS_FEATURES
        }
        cls.payload_regresion = {
            campo: cls.muestra[campo]
            for campo in REGRESSION_FEATURES
        }
        cls.payload_clasificacion = {
            campo: cls.muestra[campo]
            for campo in CLASSIFICATION_FEATURES
        }

    def test_endpoint_raiz_informa_estado(self):
        respuesta = self.client.get("/")
        self.assertEqual(respuesta.status_code, 200)
        body = respuesta.json()
        self.assertIn("mensaje", body)
        self.assertIn("artefactos", body)
        self.assertTrue(body["artefactos"]["kmeans"])

    def test_health(self):
        respuesta = self.client.get("/health")
        self.assertEqual(respuesta.status_code, 200)
        self.assertIn(respuesta.json()["status"], {"ok", "degraded"})

    def test_dashboard_data(self):
        respuesta = self.client.get("/dashboard-data")
        self.assertEqual(respuesta.status_code, 200)
        body = respuesta.json()
        self.assertIn("usuarios", body)
        self.assertIn("centroides", body)
        self.assertIn("evaluacion_k", body)
        self.assertIn("metricas", body)

    def test_business_kpis(self):
        respuesta = self.client.get("/business-kpis")
        self.assertEqual(respuesta.status_code, 200)
        body = respuesta.json()
        self.assertIn("kpis", body)
        self.assertIn("resumen", body)
        self.assertGreater(len(body["kpis"]), 0)

    def test_data_quality(self):
        respuesta = self.client.get("/data-quality")
        self.assertEqual(respuesta.status_code, 200)
        body = respuesta.json()
        self.assertEqual(body["filas_originales"], 300)
        self.assertIn("limpieza", body)

    def test_pipeline_status(self):
        respuesta = self.client.get("/pipeline-status")
        self.assertEqual(respuesta.status_code, 200)
        body = respuesta.json()
        self.assertEqual(body["manifest"]["status"], "completed")
        self.assertIn("runtime", body)

    def test_model_evidence(self):
        respuesta = self.client.get("/model-evidence")
        if respuesta.status_code == 503:
            self.skipTest("Evidencia supervisada no disponible.")
        self.assertEqual(respuesta.status_code, 200)
        body = respuesta.json()
        self.assertIn("regression_predictions", body)
        self.assertIn("classification_predictions", body)
        self.assertIn("regression_feature_importance", body)
        self.assertIn("classification_feature_importance", body)

    def test_predict_cluster(self):
        respuesta = self.client.post(
            "/predict",
            json=self.payload_kmeans,
        )
        self.assertEqual(respuesta.status_code, 200)
        self.assertIn("cluster", respuesta.json())

    def test_predict_cluster_rechaza_campo_faltante(self):
        payload = self.payload_kmeans.copy()
        payload.pop(KMEANS_FEATURES[0])
        respuesta = self.client.post("/predict", json=payload)
        self.assertEqual(respuesta.status_code, 422)

    def test_predict_cluster_rechaza_campo_extra(self):
        payload = {**self.payload_kmeans, "campo_inventado": 1}
        respuesta = self.client.post("/predict", json=payload)
        self.assertEqual(respuesta.status_code, 422)

    def test_predict_rechaza_valor_fuera_de_rango(self):
        payload = self.payload_kmeans.copy()
        payload["porcentaje_uso_app_movil"] = 3.0
        respuesta = self.client.post("/predict", json=payload)
        self.assertEqual(respuesta.status_code, 422)

    def test_predict_gasto(self):
        respuesta = self.client.post(
            "/predict-gasto",
            json=self.payload_regresion,
        )
        if respuesta.status_code == 503:
            self.skipTest("Modelo de regresión no entrenado.")
        self.assertEqual(respuesta.status_code, 200)
        self.assertGreaterEqual(
            respuesta.json()["gasto_mensual_predicho"],
            0,
        )

    def test_predict_gasto_rechaza_target_como_entrada(self):
        payload = {
            **self.payload_regresion,
            "gasto_mensual": 999,
        }
        respuesta = self.client.post("/predict-gasto", json=payload)
        self.assertEqual(respuesta.status_code, 422)

    def test_predict_riesgo_con_features_minimas(self):
        respuesta = self.client.post(
            "/predict-riesgo",
            json=self.payload_clasificacion,
        )
        if respuesta.status_code == 503:
            self.skipTest("Modelo de clasificación no entrenado.")
        self.assertEqual(respuesta.status_code, 200)
        body = respuesta.json()
        self.assertIn(body["riesgo_bajo_compromiso"], [0, 1])
        self.assertGreaterEqual(body["probabilidad"], 0)
        self.assertLessEqual(body["probabilidad"], 1)

    def test_predict_riesgo_acepta_payload_dashboard(self):
        payload = {
            **self.payload_clasificacion,
            **{
                campo: self.muestra[campo]
                for campo in CLASSIFICATION_TARGET_SOURCE_COLUMNS
            },
        }
        respuesta = self.client.post("/predict-riesgo", json=payload)
        if respuesta.status_code == 503:
            self.skipTest("Modelo de clasificación no entrenado.")
        self.assertEqual(respuesta.status_code, 200)

    def test_metricas_supervisadas(self):
        respuesta = self.client.get("/metricas-supervisado")
        if respuesta.status_code == 503:
            self.skipTest("Métricas supervisadas no disponibles.")
        self.assertEqual(respuesta.status_code, 200)
        self.assertIn("regresion", respuesta.json())
        self.assertIn("clasificacion", respuesta.json())


if __name__ == "__main__":
    unittest.main()