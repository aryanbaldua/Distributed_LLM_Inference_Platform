"""Worker service: registers with the controller, heartbeats for as long as it
is up, and serves the requests the controller forwards to it.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
from uuid import uuid4

import httpx
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from common.config import WorkerSettings
from common.logging import get_logger
from common.schemas import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
)
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

    # Held on app.state rather than in settings because /debug/delay changes it.
    app.state.delay_s = settings.mock_delay_s
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


class DelaySetting(BaseModel):
    delay_s: float = Field(ge=0.0)


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok", "role": "worker", "worker_id": settings.worker_id}


@app.post("/v1/chat/completions")
async def chat_completions(
    req: ChatCompletionRequest, request: Request
) -> ChatCompletionResponse:
    """Stand in for a model. The delay is what makes concurrent load observable:
    without it every request finishes before the next one is dispatched.
    """
    await asyncio.sleep(request.app.state.delay_s)
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid4().hex[:24]}",
        model=settings.model,
        choices=[
            ChatCompletionChoice(
                message=ChatMessage(
                    role="assistant",
                    content=f"[{settings.worker_id}] echo: {req.messages[-1].content}",
                )
            )
        ],
    )


@app.post("/debug/delay")
async def set_delay(setting: DelaySetting, request: Request) -> DelaySetting:
    """Make this worker the slow one while it is running, to watch load move to
    its peers without restarting anything.
    """
    request.app.state.delay_s = setting.delay_s
    log.info("mock delay now %.2fs", setting.delay_s)
    return setting


def main() -> None:
    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port, log_level="warning")


if __name__ == "__main__":
    main()
