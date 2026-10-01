"""The log line itself: that it is parseable, and that it carries the id.

These assert on the formatters directly rather than through get_logger, which
caches a handler per logger name and so cannot be reconfigured once a module
has imported it.
"""

import json
import logging

import pytest

from common.context import _request_id, set_request_id
from common.logging import JsonFormatter, TextFormatter


@pytest.fixture(autouse=True)
def outside_a_request():
    """Start every test with no id bound.

    A ContextVar set from a synchronous test outlives that test, so without this
    the first test to bind an id would silently attribute every line written by
    the tests that follow it.
    """
    token = _request_id.set(None)
    yield
    _request_id.reset(token)


def record(msg="hello", fields=None, level=logging.INFO, exc_info=None):
    made = logging.LogRecord("master", level, __file__, 1, msg, None, exc_info)
    if fields is not None:
        made.fields = fields
    return made


def formatted(fields=None, **kwargs):
    return json.loads(JsonFormatter().format(record(fields=fields, **kwargs)))


# --- the envelope ---------------------------------------------------------------


def test_every_line_is_one_json_object():
    line = JsonFormatter().format(record())

    assert "\n" not in line, "a line that wraps is two lines to whatever reads the log"
    assert json.loads(line)["msg"] == "hello"


def test_a_line_names_its_level_and_logger():
    payload = formatted(level=logging.WARNING)

    assert payload["level"] == "WARNING"
    assert payload["logger"] == "master"


def test_structured_fields_are_merged_into_the_object():
    payload = formatted(fields={"worker_id": "worker-a", "total_ms": 12.5})

    assert payload["worker_id"] == "worker-a"
    assert payload["total_ms"] == 12.5


def test_a_value_that_cannot_be_serialized_does_not_lose_the_line():
    payload = formatted(fields={"worker": object()})

    assert "worker" in payload, "the line survives; the value degrades to its repr"


def test_a_traceback_is_carried_on_the_line_that_reported_it():
    try:
        raise ValueError("upstream said no")
    except ValueError as exc:
        payload = formatted(exc_info=(type(exc), exc, exc.__traceback__))

    assert "upstream said no" in payload["exc"]


# --- the request id ------------------------------------------------------------


def test_a_line_written_outside_a_request_has_no_id():
    assert "request_id" not in formatted()


def test_a_line_written_during_a_request_is_attributed_to_it():
    set_request_id("abc123")

    assert formatted()["request_id"] == "abc123"


def test_the_text_format_keeps_the_id_and_the_fields():
    set_request_id("abc123")

    line = TextFormatter().format(record(fields={"status": 503}))

    assert "request_id=abc123" in line
    assert "status=503" in line
