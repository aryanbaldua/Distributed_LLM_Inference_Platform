"""Request-scoped context, carried implicitly instead of passed by hand.

One request_id has to appear on every line a request produces, in both
processes, including the lines written from error paths. Threading it through
every signature that might log would touch most of the codebase and would be
forgotten in exactly those error paths, so it lives in a ContextVar instead:
set once where a request enters a process, read by the log formatter.

Nothing resets the variable. Each request is handled in its own asyncio task,
and a task copies the context at creation, so a value set inside one handler is
invisible to every other.
"""

from __future__ import annotations

from contextvars import ContextVar
from uuid import uuid4

from starlette.datastructures import MutableHeaders

# The header the id travels on. Master to worker is the only hop in V0, but
# naming it once means the two ends cannot disagree about the spelling.
REQUEST_ID_HEADER = "X-Request-Id"

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)


def new_request_id() -> str:
    """Short rather than a full uuid: these are read by humans, side by side in
    two logs, and only have to be unique among the requests currently in flight.
    """
    return uuid4().hex[:12]


def set_request_id(request_id: str | None) -> str:
    """Bind an id to this request, minting one when the caller arrived without.

    Takes the optional header value directly so a caller never has to decide
    whether it is the origin of the id or a relay of someone else's.
    """
    request_id = request_id or new_request_id()
    _request_id.set(request_id)
    return request_id


def get_request_id() -> str | None:
    return _request_id.get()


class RequestIdHeaderMiddleware:
    """Puts whatever id a request was traced under onto the response.

    This cannot be done in the handler. Raising an HTTPException discards the
    Response object the handler was given and builds a fresh one, so a header set
    there survives only on the paths that return normally - losing it on exactly
    the 503s and 502s a client is most likely to come asking about.

    Plain ASGI rather than BaseHTTPMiddleware: BaseHTTPMiddleware runs the rest
    of the app in a separate task, and a task gets a copy of the context, so the
    id the handler set would not be visible here.
    """

    def __init__(self, app) -> None:
        self._app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        async def send_with_request_id(message):
            if message["type"] == "http.response.start":
                request_id = get_request_id()
                if request_id is not None:
                    MutableHeaders(scope=message)[REQUEST_ID_HEADER] = request_id
            await send(message)

        await self._app(scope, receive, send_with_request_id)
