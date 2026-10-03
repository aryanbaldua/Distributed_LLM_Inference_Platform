"""Master service: owns cluster membership and health, routes client requests
to the workers that can serve them, and exposes the operator view.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
from typing import Annotated

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, Response

from common.config import MasterSettings
from common.context import REQUEST_ID_HEADER, RequestIdHeaderMiddleware, set_request_id
from common.logging import get_logger
from common.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ClusterView,
    HeartbeatRequest,
    HeartbeatResponse,
    RegisterRequest,
    RegisterResponse,
)
from master.reaper import reaper_loop
from master.registry import WorkerRegistry
from master.scheduler import ALL_WORKERS_AT_CAPACITY, refusal_reason, select
from master.trace import RequestTrace

settings = MasterSettings()
log = get_logger("master")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Built per-app rather than at import time so each test gets a clean cluster.
    app.state.registry = WorkerRegistry()
    app.state.http = httpx.AsyncClient()
    app.state.reaper = asyncio.create_task(reaper_loop(app.state.registry, settings))
    log.info(
        "master up on %s:%d (heartbeat %.1fs, timeout %.1fs)",
        settings.host,
        settings.port,
        settings.heartbeat_interval_s,
        settings.heartbeat_timeout_s,
    )

    yield

    # Awaited after cancelling, not just cancelled: without this the loop is
    # still pending when the event loop closes, and shutdown races it.
    app.state.reaper.cancel()
    with suppress(asyncio.CancelledError):
        await app.state.reaper
    await app.state.http.aclose()
    log.info("master shutting down")


app = FastAPI(title="LLM Inference Master", version="0.0.1", lifespan=lifespan)
app.add_middleware(RequestIdHeaderMiddleware)


def get_registry(request: Request) -> WorkerRegistry:
    return request.app.state.registry


def get_http(request: Request) -> httpx.AsyncClient:
    return request.app.state.http


Registry = Annotated[WorkerRegistry, Depends(get_registry)]
Http = Annotated[httpx.AsyncClient, Depends(get_http)]


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok", "role": "master"}


@app.post("/workers/register")
async def register_worker(req: RegisterRequest, registry: Registry) -> RegisterResponse:
    """Join the cluster, and receive the timing policy to heartbeat against.

    The interval comes from the master rather than the worker's own config so
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


def _refusal_detail(reason: str | None, model: str) -> str:
    """Say which of the two refusals this is in the words a client reads."""
    if reason == ALL_WORKERS_AT_CAPACITY:
        return f"every worker serving model {model!r} is at capacity"
    return f"no healthy worker is serving model {model!r}"


@app.post("/v1/chat/completions")
async def chat_completions(
    req: ChatCompletionRequest,
    registry: Registry,
    http: Http,
    request: Request,
    response: Response,
) -> ChatCompletionResponse:
    """The one endpoint a client needs. Chooses a worker and forwards to it.

    An id is minted here, or adopted from the caller's header, and every line
    this request writes in either process carries it. Only this endpoint is
    traced: membership traffic is a steady background hum rather than something
    anyone follows one message at a time.
    """
    request_id = set_request_id(request.headers.get(REQUEST_ID_HEADER))

    if req.stream:
        raise HTTPException(status_code=400, detail="streaming is not supported yet")

    with RequestTrace(request_id, req.model, registry, log) as trace:

        def choose(workers):
            chosen = select(workers, req.model, settings.heartbeat_timeout_s)
            reason = (
                None
                if chosen is not None
                else refusal_reason(workers, req.model, settings.heartbeat_timeout_s)
            )
            trace.record_selection(chosen, workers, reason)
            return chosen

        chosen = registry.reserve(choose)
        if chosen is None:
            # The same reason the log line carries, so a client chasing a 503 is
            # told what the operator can already see rather than a generic one.
            raise HTTPException(status_code=503, detail=_refusal_detail(trace.reason, req.model))

        try:
            upstream = await http.post(
                f"http://{chosen.address}/v1/chat/completions",
                json=req.model_dump(mode="json"),
                headers={REQUEST_ID_HEADER: request_id},
                timeout=settings.forward_timeout_s,
            )
            upstream.raise_for_status()
            # Parsed rather than relayed untouched, so a worker answering with
            # something that is not a completion is caught here and not by the
            # client. ValueError covers both a non-JSON body and a failed validation.
            completion = ChatCompletionResponse.model_validate(upstream.json())
        except (httpx.HTTPError, ValueError) as exc:
            log.warning("worker %s failed to serve a request: %s", chosen.worker_id, exc)
            raise HTTPException(
                status_code=502, detail=f"worker {chosen.worker_id} did not complete the request"
            ) from exc

        response.headers["X-Worker-Id"] = chosen.worker_id
        return completion


def main() -> None:
    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port, log_level="warning")


if __name__ == "__main__":
    main()
