"""Env-driven settings. Master and worker each read their own section."""

from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class MasterSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MASTER_", env_file=".env", extra="ignore")

    host: str = "0.0.0.0"
    port: int = 8000

    # Timing policy lives here and is pushed to workers via RegisterResponse.
    heartbeat_interval_s: float = 4.0
    heartbeat_timeout_s: float = 10.0
    reaper_interval_s: float = 1.0

    # How long an unreachable worker stays in the registry before being dropped.
    # Unhealthy workers are kept deliberately: a worker that is visibly dead is
    # the point of the operator view, and one that comes back should find its
    # own record rather than a stranger's. Eviction only stops the registry
    # growing without bound across a long-lived master.
    evict_after_s: float = 300.0

    # Bounds one forwarded inference request. Unrelated to the heartbeat
    # timeouts above: generation legitimately takes far longer than a heartbeat,
    # so this is generous where those are tight.
    forward_timeout_s: float = 60.0


class WorkerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="WORKER_", env_file=".env", extra="ignore")

    worker_id: str = "worker-1"
    host: str = "0.0.0.0"
    port: int = 8001
    model: str = "mock-model"
    max_concurrency: int = 3

    master_url: str = "http://127.0.0.1:8000"

    # Registration retries back off exponentially from register_retry_s up to
    # register_backoff_max_s, so a worker started before the master waits
    # without hammering it. Heartbeats deliberately do not back off.
    register_retry_s: float = 2.0
    register_backoff_max_s: float = 30.0

    # Where answers come from. "mock" echoes the prompt after a delay and needs
    # nothing installed, which keeps the tests and CI fast. "ollama" sends the
    # request to a real model; `model` must then be a name Ollama has pulled.
    backend: Literal["mock", "ollama"] = "mock"

    # Stands in for generation time on the mock backend. Adjustable at runtime
    # through POST /debug/delay, to make one worker the slow one.
    mock_delay_s: float = 0.25

    ollama_url: str = "http://127.0.0.1:11434"

    # Kept below the master's forward_timeout_s on purpose. If the master gave
    # up first it would see a timeout, take the worker for dead and retry the
    # whole generation elsewhere; giving up here first means it hears an error
    # instead, which it does not retry.
    ollama_timeout_s: float = 55.0

    @property
    def advertised_address(self) -> str:
        """What the master should dial. 0.0.0.0 is a bind address, not a destination."""
        host = "127.0.0.1" if self.host in ("0.0.0.0", "::") else self.host
        return f"{host}:{self.port}"
