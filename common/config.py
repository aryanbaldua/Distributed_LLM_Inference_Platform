"""Env-driven settings. Controller and worker each read their own section."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class ControllerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CONTROLLER_", env_file=".env", extra="ignore")

    host: str = "0.0.0.0"
    port: int = 8000

    # Timing policy lives here and is pushed to workers via RegisterResponse.
    heartbeat_interval_s: float = 2.0
    heartbeat_timeout_s: float = 6.0
    reaper_interval_s: float = 1.0


class WorkerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="WORKER_", env_file=".env", extra="ignore")

    worker_id: str = "worker-1"
    host: str = "0.0.0.0"
    port: int = 8001
    model: str = "mock-model"
    max_concurrency: int = 8

    controller_url: str = "http://127.0.0.1:8000"
    register_retry_s: float = 2.0

    @property
    def advertised_address(self) -> str:
        """What the controller should dial. 0.0.0.0 is a bind address, not a destination."""
        host = "127.0.0.1" if self.host in ("0.0.0.0", "::") else self.host
        return f"{host}:{self.port}"
