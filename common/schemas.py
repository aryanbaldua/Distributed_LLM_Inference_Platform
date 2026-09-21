"""Shared wire contract between controller, workers, and clients.

Every model here is part of an interface crossing a process boundary, so changes
ripple. Controller-internal state that never leaves the process does not belong
in this file.
"""

from __future__ import annotations

from enum import Enum
from time import time
from typing import Literal

from pydantic import BaseModel, Field


class WorkerStatus(str, Enum):
    HEALTHY = "HEALTHY"
    UNHEALTHY = "UNHEALTHY"


# --- Worker -> Controller: membership (design 3.1) ---------------------------


class RegisterRequest(BaseModel):
    worker_id: str
    address: str = Field(description="host:port the controller forwards inference to")
    model: str = Field(description="exact model id served, e.g. 'Qwen/Qwen2.5-1.5B-Instruct'")
    max_concurrency: int = Field(default=8, ge=1)


class RegisterResponse(BaseModel):
    """The controller owns the timing policy and hands it to the worker.

    Keeping the interval on the controller side means it can never be configured
    with a timeout shorter than the interval its workers actually use.
    """

    heartbeat_interval_s: float
    heartbeat_timeout_s: float


class HeartbeatRequest(BaseModel):
    worker_id: str
    active_requests: int = Field(ge=0)
    status: WorkerStatus = WorkerStatus.HEALTHY
    gpu_utilization: float | None = Field(default=None, ge=0.0, le=1.0)
    free_vram_mb: int | None = Field(default=None, ge=0)


class HeartbeatResponse(BaseModel):
    ok: bool = True


# --- Controller-internal cluster state (design 4.1) --------------------------


class WorkerRecord(BaseModel):
    worker_id: str
    address: str
    model: str
    max_concurrency: int

    # Bumped every time a worker re-registers under an id the controller already
    # knows. A restarted worker process is a new generation, which is what tells
    # "rejoined after a crash" apart from "never left" when reading the logs.
    generation: int = 1

    status: WorkerStatus = WorkerStatus.HEALTHY
    last_heartbeat: float = Field(default_factory=time)
    active_requests: int = 0
    gpu_utilization: float | None = None
    free_vram_mb: int | None = None

    @classmethod
    def from_registration(
        cls, req: RegisterRequest, now: float | None = None, generation: int = 1
    ) -> "WorkerRecord":
        """Build a fresh record. `now` is injectable so tests never touch the clock."""
        return cls(
            worker_id=req.worker_id,
            address=req.address,
            model=req.model,
            max_concurrency=req.max_concurrency,
            generation=generation,
            last_heartbeat=time() if now is None else now,
        )

    def heartbeat_age(self, now: float | None = None) -> float:
        return (time() if now is None else now) - self.last_heartbeat


class ClusterView(BaseModel):
    """Response for GET /cluster/workers - the operator/debug view."""

    as_of: float = Field(
        default_factory=time,
        description=(
            "Controller clock when the snapshot was taken. Heartbeat age is "
            "as_of - last_heartbeat, so readers never have to trust that their "
            "own clock agrees with the controller's."
        ),
    )
    workers: list[WorkerRecord]


# --- Client -> Controller: inference (design 5) ------------------------------


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    max_tokens: int = Field(default=256, ge=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
