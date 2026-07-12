import unittest
from pathlib import Path

from fastapi.testclient import TestClient


class TestAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not Path("models/modelo_kmeans.pkl").exists():
            raise unittest.SkipTest(
                "models/modelo_kmeans.pkl no existe. Corre model/train.py primero."
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

    def test_endpoint_raiz(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("mensaje", r.json())

    def test_predict_cluster(self):
        r = self.client.post("/predict", json=self.muestra)
        self.assertEqual(r.status_code, 200)
        self.assertIn("cluster", r.json())

    def test_predict_gasto(self):
        r = self.client.post("/predict-gasto", json=self.muestra)
        if r.status_code == 503:
            self.skipTest("Modelo de regresión no entrenado todavía.")
        self.assertEqual(r.status_code, 200)
        self.assertIn("gasto_mensual_predicho", r.json())
        self.assertGreaterEqual(r.json()["gasto_mensual_predicho"], 0)

    def test_predict_riesgo(self):
        r = self.client.post("/predict-riesgo", json=self.muestra)
        if r.status_code == 503:
            self.skipTest("Modelo de clasificación no entrenado todavía.")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIn(body["riesgo_bajo_compromiso"], [0, 1])
        self.assertGreaterEqual(body["probabilidad"], 0)
        self.assertLessEqual(body["probabilidad"], 1)

    def test_predict_gasto_datos_incompletos(self):
        r = self.client.post("/predict-gasto", json={"edad": 30})
        self.assertIn(r.status_code, [422, 503])


if __name__ == "__main__":
    unittest.main()
