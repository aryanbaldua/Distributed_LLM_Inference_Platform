"""The real-model backend: hands a request to a local Ollama server.

Ollama speaks the same OpenAI-style chat format the cluster already uses, so
this is a forward and a validation rather than a translation. The model name
sent is the one this worker registered with, which is also the name the master
routed on - one name end to end, so there is nothing to keep in sync.
"""

from __future__ import annotations

import httpx
from fastapi import HTTPException

from common.schemas import ChatCompletionRequest, ChatCompletionResponse


async def complete(
    http: httpx.AsyncClient,
    base_url: str,
    model: str,
    req: ChatCompletionRequest,
    timeout_s: float,
) -> ChatCompletionResponse:
    """One non-streaming completion, or an HTTPException saying what went wrong.

    Every failure becomes an HTTP error from this worker rather than a dropped
    connection, because the master treats the two differently: a worker that
    answers with an error is not retried, and a worker that never answers is.
    A model that is missing or a request that is too slow will be missing or too
    slow on the next worker too, so it should not be.
    """
    try:
        response = await http.post(
            f"{base_url.rstrip('/')}/v1/chat/completions",
            json={
                "model": model,
                "messages": [message.model_dump() for message in req.messages],
                "max_tokens": req.max_tokens,
                "temperature": req.temperature,
                "stream": False,
            },
            timeout=timeout_s,
        )
        response.raise_for_status()
        return ChatCompletionResponse.model_validate(response.json())
    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=504, detail=f"ollama did not finish within {timeout_s:g}s"
        ) from exc
    except httpx.HTTPStatusError as exc:
        # Ollama's own message is the useful part, e.g. "model not found, try
        # pulling it first". Truncated so a large error page cannot flood a log.
        raise HTTPException(
            status_code=502,
            detail=f"ollama returned {exc.response.status_code}: {exc.response.text[:200]}",
        ) from exc
    except httpx.TransportError as exc:
        raise HTTPException(
            status_code=502, detail=f"ollama is not reachable at {base_url}: {exc}"
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=502, detail="ollama answered with something that is not a completion"
        ) from exc
