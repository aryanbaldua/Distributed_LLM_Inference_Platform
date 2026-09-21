"""Worker service: registers with the controller and heartbeats for as long as
it is up. The inference endpoint lands with routing.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress

import httpx
from fastapi import FastAPI

from common.config import WorkerSettings
from common.logging import get_logger
from worker.controller_client import ControllerClient

settings = WorkerSettings()
log = get_logger(f"worker:{settings.worker_id}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info(
        "worker %s up at %s serving %s",
        settings.worker_id,
        settings.advertised_address,
        settings.model,
    )

    app.state.http = httpx.AsyncClient()
    app.state.membership = asyncio.create_task(
        ControllerClient(settings, app.state.http, log).run()
    )

    yield

    # No deregistration on the way out. A worker that politely announces its own
    # death is a worker whose ordinary shutdown never exercises the failure
    # detection this cluster depends on; the controller notices the silence.
    app.state.membership.cancel()
    with suppress(asyncio.CancelledError):
        await app.state.membership
    await app.state.http.aclose()
    log.info("worker %s shutting down", settings.worker_id)


app = FastAPI(title="LLM Inference Worker", version="0.0.1", lifespan=lifespan)


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok", "role": "worker", "worker_id": settings.worker_id}


def main() -> None:
    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port, log_level="warning")


if __name__ == "__main__":
    main()
