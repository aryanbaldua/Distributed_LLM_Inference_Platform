"""Controller service: owns cluster membership and answers the operator view.

Failure detection - marking a worker unhealthy once its heartbeats stop - lands
next, and adds a background reaper alongside these endpoints.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request

from common.config import ControllerSettings
from common.logging import get_logger
from common.schemas import (
    ClusterView,
    HeartbeatRequest,
    HeartbeatResponse,
    RegisterRequest,
    RegisterResponse,
)
from controller.registry import WorkerRegistry

settings = ControllerSettings()
log = get_logger("controller")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Built per-app rather than at import time so each test gets a clean cluster.
    app.state.registry = WorkerRegistry()
    log.info(
        "controller up on %s:%d (heartbeat %.1fs, timeout %.1fs)",
        settings.host,
        settings.port,
        settings.heartbeat_interval_s,
        settings.heartbeat_timeout_s,
    )
    yield
    log.info("controller shutting down")


app = FastAPI(title="LLM Inference Controller", version="0.0.1", lifespan=lifespan)


def get_registry(request: Request) -> WorkerRegistry:
    return request.app.state.registry


Registry = Annotated[WorkerRegistry, Depends(get_registry)]


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok", "role": "controller"}


@app.post("/workers/register")
async def register_worker(req: RegisterRequest, registry: Registry) -> RegisterResponse:
    """Join the cluster, and receive the timing policy to heartbeat against.

    The interval comes from the controller rather than the worker's own config so
    that the sender's cadence and the detector's timeout can never be configured
    independently of each other.
    """
    registry.register(req)
    return RegisterResponse(
        heartbeat_interval_s=settings.heartbeat_interval_s,
        heartbeat_timeout_s=settings.heartbeat_timeout_s,
    )


@app.post("/workers/heartbeat")
async def worker_heartbeat(req: HeartbeatRequest, registry: Registry) -> HeartbeatResponse:
    if not registry.heartbeat(req):
        # Not an error the worker should retry as-is: it has to register first.
        raise HTTPException(
            status_code=404,
            detail=f"unknown worker {req.worker_id!r}; register before heartbeating",
        )
    return HeartbeatResponse()


@app.get("/cluster/workers")
async def cluster_workers(registry: Registry) -> ClusterView:
    """Full registry dump. The operator view, and the evidence for every demo."""
    return ClusterView(workers=registry.snapshot())


def main() -> None:
    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port, log_level="warning")


if __name__ == "__main__":
    main()
