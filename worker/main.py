"""Worker service. M0: boots and answers /healthz.

M1 adds registration + the heartbeat loop; M3 adds the inference endpoint.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from common.config import WorkerSettings
from common.logging import get_logger

settings = WorkerSettings()
log = get_logger(f"worker:{settings.worker_id}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # M1 starts registration + the heartbeat loop here.
    log.info(
        "worker %s up at %s serving %s",
        settings.worker_id,
        settings.advertised_address,
        settings.model,
    )
    yield
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
