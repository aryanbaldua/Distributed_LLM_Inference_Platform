"""Logging setup. One formatter, configured in one place, so every process in the
cluster produces lines that can be read side by side.

Lines are JSON by default, because the interesting fields on a request - which
worker, how busy its peers were, how long each stage took - are values to be
filtered and compared, not prose to be read once. Set LOG_FORMAT=text for the
old human-readable form while watching a demo in a terminal.

Structured fields are passed as a single dict under `fields`:

    log.info("request finished", extra={"fields": {"worker_id": "worker-a"}})

One nested key rather than loose kwargs, so the formatter never has to tell a
caller's field apart from one of LogRecord's three dozen built-in attributes.

Every process also keeps its most recent lines in memory, in JSON form whatever
LOG_FORMAT says, so they can be served over HTTP to the command center page
rather than read out of a terminal per process.
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import sys
from collections import deque
from uuid import uuid4

from common.context import get_request_id


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        # default=str so an unserializable value degrades to its repr instead of
        # raising inside the logging call and losing the line entirely.
        return json.dumps(self.payload(record), default=str)

    def payload(self, record: logging.LogRecord) -> dict:
        payload = {
            "ts": self.formatTime(record, "%H:%M:%S"),
            # Epoch seconds alongside the readable time: lines from several
            # processes are merged by this, and `ts` is too coarse to order them.
            "t": round(record.created, 4),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }

        # Pulled from the context rather than from the record, so every line is
        # attributed without its call site having to know a request is in play.
        request_id = get_request_id()
        if request_id is not None:
            payload["request_id"] = request_id

        payload.update(getattr(record, "fields", {}))

        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return payload


class TextFormatter(logging.Formatter):
    """The readable form. Fields are appended as key=value so the same
    information survives the switch, just less conveniently.
    """

    def __init__(self) -> None:
        super().__init__("%(asctime)s %(levelname)-5s [%(name)s] %(message)s", "%H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        line = super().format(record)
        request_id = get_request_id()
        if request_id is not None:
            line = f"{line} request_id={request_id}"
        fields = getattr(record, "fields", {})
        if fields:
            line = f"{line} " + " ".join(f"{key}={value}" for key, value in fields.items())
        return line


class LogBuffer:
    """The newest lines this process has written, for serving over HTTP.

    Bounded, because it lives for as long as the process does. Each entry gets
    a sequence number and this process's boot id, so a reader polling it can
    tell lines it has already seen from new ones - including after a restart,
    when the sequence starts again from zero.
    """

    def __init__(self, maxlen: int = 1000) -> None:
        self.boot = uuid4().hex[:8]
        self._lines: deque[dict] = deque(maxlen=maxlen)
        self._seq = itertools.count()

    def append(self, payload: dict) -> None:
        self._lines.append({**payload, "seq": next(self._seq), "boot": self.boot})

    def recent(self, limit: int) -> list[dict]:
        """The newest `limit` lines, oldest first."""
        return list(self._lines)[-limit:] if limit > 0 else []


# One per process, shared by every logger in it.
log_buffer = LogBuffer()


class BufferHandler(logging.Handler):
    def __init__(self, buffer: LogBuffer) -> None:
        super().__init__()
        self._buffer = buffer
        self._formatter = JsonFormatter()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            # Round-tripped through JSON so what is stored is exactly what the
            # stream would print: an unserializable field is already its repr
            # here, rather than an error later in whatever serves the buffer.
            self._buffer.append(json.loads(self._formatter.format(record)))
        except Exception:
            self.handleError(record)


def _formatter() -> logging.Formatter:
    if os.environ.get("LOG_FORMAT", "json").lower() == "text":
        return TextFormatter()
    return JsonFormatter()


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_formatter())
    logger.addHandler(handler)
    logger.addHandler(BufferHandler(log_buffer))
    logger.setLevel(level)
    logger.propagate = False
    return logger
