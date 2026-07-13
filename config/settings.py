from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    """Configuración central del proyecto."""

    # Directorios
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    database_dir: Path = PROJECT_ROOT / "database"
    models_dir: Path = PROJECT_ROOT / "models"
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"

    # Fuentes
    streaming_csv: Path = PROJECT_ROOT / "data" / "usuarios_streaming.csv"

    # Productos intermedios
    consolidated_dataset: Path = (
        PROJECT_ROOT / "data" / "data_consolidada.csv"
    )
    clean_base_dataset: Path = (
        PROJECT_ROOT / "data" / "dataset_base_limpio.csv"
    )
    analytical_dataset: Path = (
        PROJECT_ROOT / "data" / "dataset_analitico.csv"
    )
    legacy_model_dataset: Path = (
    PROJECT_ROOT / "data" / "dataset_modelo.csv"
    )
    business_kpis: Path = (
        PROJECT_ROOT / "data" / "kpis_negocio.csv"
    )
    quality_report: Path = (
        PROJECT_ROOT / "data" / "reporte_calidad.json"
    )

    # PostgreSQL
    postgres_user: str = os.getenv("POSTGRES_USER", "admin")
    postgres_password: str = os.getenv("POSTGRES_PASSWORD", "admin")
    postgres_db: str = os.getenv("POSTGRES_DB", "streaming_db")
    postgres_host: str = os.getenv("POSTGRES_HOST", "localhost")
    postgres_port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    postgres_table: str = os.getenv(
        "POSTGRES_TABLE",
        "perfil_usuarios",
    )

    # Machine learning
    random_state: int = int(os.getenv("RANDOM_STATE", "29"))
    n_jobs: int = int(os.getenv("N_JOBS", "1"))

    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    @property
    def database_url(self) -> str:
        """Construye la URL de conexión a PostgreSQL."""
        return (
            f"postgresql+psycopg2://"
            f"{self.postgres_user}:"
            f"{self.postgres_password}@"
            f"{self.postgres_host}:"
            f"{self.postgres_port}/"
            f"{self.postgres_db}"
        )

    def create_directories(self) -> None:
        """Crea los directorios de salida requeridos."""
        directories = [
            self.data_dir,
            self.models_dir,
            self.artifacts_dir,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


settings = Settings()