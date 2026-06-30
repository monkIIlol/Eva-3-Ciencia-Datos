# Guía de Despliegue del Sistema

Este documento describe los pasos necesarios para instalar y ejecutar la solución integral de segmentación de usuarios de streaming mediante contenedores Docker.

## 1. Requisitos Previos
Para poner en marcha la solución se requiere contar con las siguientes herramientas en el sistema:
* Docker Desktop y Docker Compose instalados.
* Conexión a Internet para la descarga inicial de las imágenes base de Python y Postgres.

## 2. Configuración Externa (Variables de Entorno)
El proyecto utiliza un archivo `.env` en la raíz del repositorio para gestionar de forma segura los parámetros de autenticación y conectividad de la base de datos:

```env
DB_USER=admin
DB_PASSWORD=admin
DB_HOST=postgres
DB_PORT=5432
DB_NAME=streaming_db
DATABASE_URL=postgresql://admin:admin@postgres:5432/streaming_db