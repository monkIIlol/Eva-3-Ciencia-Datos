import pandas as pd
import json
import pickle
import os
import logging
import sys

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
from sklearn.decomposition import PCA

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def ejecutar_entrenamiento():
    try:
        logging.info("--- Iniciando Proceso de Modelamiento Analítico ---")
        os.makedirs("models", exist_ok=True)

        ruta_entrada = "data/data_consolidada.csv"
        if not os.path.exists(ruta_entrada):
            raise FileNotFoundError(f"Falta el insumo analítico: {ruta_entrada}. Ejecute extract.py primero.")

        data = pd.read_csv(ruta_entrada)

        X = data.drop(columns=["id_cliente"], errors="ignore")
        X = X.select_dtypes(include=["number"])
        logging.info(f"Variables numéricas seleccionadas para el modelo: {list(X.columns)}")

        #Escalamiento obligatorio
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        logging.info("Escalamiento estándar aplicado con éxito.")

        #Probar k de 2 a 10, guardando inercia y silhouette
        inertias = []
        silhouettes = []
        rango_k = range(2, 11)

        logging.info("Evaluando curvas de optimización (Inercia y Coeficiente de Silueta)...")
        for k in rango_k:
            modelo = KMeans(n_clusters=k, random_state=29, n_init=10)
            clusters_temp = modelo.fit_predict(X_scaled)
            inertias.append(modelo.inertia_)
            silhouettes.append(silhouette_score(X_scaled, clusters_temp))

        #Elegir el k óptimo con el método del codo (KneeLocator)
        kl = KneeLocator(rango_k, inertias, curve="convex", direction="decreasing")
        k_optimo = kl.elbow

        #Si KneeLocator no encuentra codo, evita el colapso
        if k_optimo is None:
            k_optimo = 3  # Valor base por defecto para segmentación de streaming
            logging.warning("KneeLocator no detectó un codo matemático claro. Se asigna k=3 por resguardo analítico.")
        else:
            logging.info(f"🎯 K óptimo detectado automáticamente vía Método del Codo: k={k_optimo}")

        #Entrenar el modelo final con el k óptimo corregido
        kmeans = KMeans(n_clusters=k_optimo, random_state=29, n_init=10)
        data["cluster"] = kmeans.fit_predict(X_scaled)
        
        # Validación de la calidad de separación de los grupos
        score_silueta_final = float(silhouette_score(X_scaled, data["cluster"]))
        logging.info(f"Modelo entrenado exitosamente. Coeficiente Silhouette: {score_silueta_final:.4f}")

        #componentes para soporte de mapas de dispersión interactivos
        pca = PCA(n_components=2)
        componentes = pca.fit_transform(X_scaled)
        data["pc1"] = componentes[:, 0]
        data["pc2"] = componentes[:, 1]

        #Guardar el dataset con clusters
        data.to_csv("data/usuarios_segmentados.csv", index=False)

        metricas = {
            "k_optimo": int(k_optimo),
            "silhouette_score": score_silueta_final,
            "n_usuarios": int(len(data)),
            "n_clusters": int(k_optimo),
            "varianza_pca": float(pca.explained_variance_ratio_.sum()),
        }
        with open("models/metricas.json", "w") as f:
            json.dump(metricas, f, indent=4)

        #Guardar centroides en la escala original 
        centroides_original = scaler.inverse_transform(kmeans.cluster_centers_)
        centroides_df = pd.DataFrame(centroides_original, columns=X.columns)
        centroides_df.to_csv("data/centroides.csv", index=False)

        with open("models/modelo_kmeans.pkl", "wb") as f:
            pickle.dump(kmeans, f)
        with open("models/scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        with open("models/pca.pkl", "wb") as f:
            pickle.dump(pca, f)

        logging.info("Todos los artefactos del modelo y métricas han sido exportados correctamente.")

    except Exception as e:
        logging.error(f"Error crítico en la fase de modelamiento: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    ejecutar_entrenamiento()