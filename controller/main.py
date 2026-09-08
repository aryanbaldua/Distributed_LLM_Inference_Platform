"""Controller service. M0: boots and answers /healthz.

M1 adds the worker registry and membership endpoints.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from common.config import ControllerSettings
from common.logging import get_logger

settings = ControllerSettings()
log = get_logger("controller")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # M1 starts the heartbeat reaper task here.
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


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok", "role": "controller"}


def main() -> None:
    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port, log_level="warning")


if __name__ == "__main__":
    main()
