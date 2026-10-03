"""What RequestTrace guarantees: the slot comes back, and one line explains why.

Driven against a real WorkerRegistry rather than a stand-in, because the thing
being checked is that release and the recorded counts agree with the registry's
own accounting.
"""

import pytest
from fastapi import HTTPException

from common.schemas import RegisterRequest
from master.registry import WorkerRegistry
from master.scheduler import (
    ALL_WORKERS_AT_CAPACITY,
    NO_WORKER_FOR_MODEL,
    refusal_reason,
    select,
)
from master.trace import RequestTrace


class CapturingLog:
    """Keeps the fields of every line, which is all these tests look at."""

    def __init__(self):
        self.lines = []

    def info(self, msg, extra=None):
        self.lines.append(extra["fields"])


@pytest.fixture
def log():
    return CapturingLog()


@pytest.fixture
def registry():
    registry = WorkerRegistry()
    for worker_id in ("worker-a", "worker-b"):
        registry.register(
            RegisterRequest(
                worker_id=worker_id,
                address=f"127.0.0.1:{8000 + int(worker_id[-1] == 'b')}",
                model="mock-model",
                max_concurrency=4,
            )
        )
    return registry


def serve(registry, log, model="mock-model", raising=None):
    """One request through a trace, shaped the way the handler shapes it."""
    with RequestTrace("req-1", model, registry, log) as trace:

        def choose(workers):
            chosen = select(workers, model, timeout_s=60.0)
            reason = None if chosen is not None else refusal_reason(workers, model, 60.0)
            trace.record_selection(chosen, workers, reason)
            return chosen

        chosen = registry.reserve(choose)
        if raising is not None:
            raise raising
        return chosen


def in_flight(registry):
    return {record.worker_id: record.active_requests for record in registry.snapshot()}


# --- giving the slot back -------------------------------------------------------


def test_a_served_request_leaves_nothing_in_flight(registry, log):
    chosen = serve(registry, log)

    assert chosen is not None
    assert set(in_flight(registry).values()) == {0}


def test_a_request_that_raised_still_gives_its_slot_back(registry, log):
    with pytest.raises(HTTPException):
        serve(registry, log, raising=HTTPException(status_code=502, detail="worker died"))

    assert set(in_flight(registry).values()) == {0}


def test_nothing_is_released_when_no_worker_was_chosen(registry, log):
    """The release has to be gated on a worker actually having been reserved.

    Releasing anyway would warn about a slot that was never taken, and on a
    cluster where nothing matches the model that warning would be every request.
    """
    serve(registry, log, model="some-other-model")

    assert log.lines[0]["worker_id"] is None
    assert set(in_flight(registry).values()) == {0}


# --- what the line says ---------------------------------------------------------


def test_the_line_names_the_worker_and_its_generation(registry, log):
    serve(registry, log)

    line = log.lines[0]
    assert line["worker_id"] in ("worker-a", "worker-b")
    assert line["worker_generation"] == 1


def test_the_line_records_the_load_the_decision_was_made_on(registry, log):
    """Counts from before the chosen worker was incremented.

    Read after the increment, every line would show the worker it picked already
    holding the request, which says nothing about why it was picked.
    """
    busy = "worker-a"
    registry.reserve(lambda workers: next(w for w in workers if w.worker_id == busy))

    serve(registry, log)

    line = log.lines[0]
    assert line["cluster_active"] == {"worker-a": 1, "worker-b": 0}
    assert line["worker_id"] == "worker-b", "the idle worker is the point of the comparison"


def test_a_served_request_is_timed_in_two_stages(registry, log):
    serve(registry, log)

    line = log.lines[0]
    assert line["select_ms"] + line["forward_ms"] <= line["total_ms"] + 0.1


def test_a_request_that_never_left_the_master_has_no_forward_time(registry, log):
    serve(registry, log, model="some-other-model")

    line = log.lines[0]
    assert "forward_ms" not in line
    assert "select_ms" in line, "choosing nothing is still a decision that took time"


# --- outcome --------------------------------------------------------------------


def test_a_completed_request_is_recorded_as_a_200(registry, log):
    serve(registry, log)

    assert log.lines[0]["status"] == 200


def test_the_status_is_taken_from_the_raised_http_error(registry, log):
    with pytest.raises(HTTPException):
        serve(registry, log, raising=HTTPException(status_code=502, detail="worker died"))

    assert log.lines[0]["status"] == 502


def test_an_unexpected_error_is_recorded_as_a_500(registry, log):
    """A bug in the handler is still an outcome the log has to account for."""
    with pytest.raises(RuntimeError):
        serve(registry, log, raising=RuntimeError("boom"))

    assert log.lines[0]["status"] == 500


# --- which refusal the line records ---------------------------------------------


def test_a_request_nobody_can_serve_says_so(registry, log):
    serve(registry, log, model="nobody-serves-this")

    assert log.lines[0]["reason"] == NO_WORKER_FOR_MODEL


def test_a_request_refused_for_room_says_that_instead(registry, log):
    """Same status code as the line above, opposite problem."""
    for _ in range(4):
        registry.reserve(lambda records: next(r for r in records if r.worker_id == "worker-a"))
    for _ in range(4):
        registry.reserve(lambda records: next(r for r in records if r.worker_id == "worker-b"))

    serve(registry, log)

    assert log.lines[0]["reason"] == ALL_WORKERS_AT_CAPACITY
    assert log.lines[0]["cluster_active"] == {"worker-a": 4, "worker-b": 4}


def test_a_served_request_has_no_reason_to_give(registry, log):
    """The field is the explanation for a refusal, so its absence is meaningful."""
    serve(registry, log)

    assert "reason" not in log.lines[0]
