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
"""

from __future__ import annotations

import json
import logging
import os
import sys

from common.context import get_request_id


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%H:%M:%S"),
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

        # default=str so an unserializable value degrades to its repr instead of
        # raising inside the logging call and losing the line entirely.
        return json.dumps(payload, default=str)


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
    logger.setLevel(level)
    logger.propagate = False
    return logger
