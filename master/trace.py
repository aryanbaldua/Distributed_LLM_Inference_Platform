"""The single place a client request ends.

Three things have to happen exactly once per request, on every path including
the ones that raise: the worker's slot goes back, a summary line is written,
and - once there are metrics - durations are observed. Repeating them in each
branch of the handler means they are correct only for as long as nobody adds a
branch, so they live in one __exit__ instead.

Deliberately a plain context manager rather than an async one. Nothing it does
awaits, and keeping it synchronous means it still works unchanged from inside
the streaming generator it will eventually be exited from.
"""

from __future__ import annotations

from time import perf_counter

from fastapi import HTTPException

from common.schemas import WorkerRecord
from master.registry import WorkerRegistry


def _ms(seconds: float) -> float:
    return round(seconds * 1000, 1)


class RequestTrace:
    def __init__(self, request_id: str, model: str, registry: WorkerRegistry, log) -> None:
        self.request_id = request_id
        self.model = model
        self.worker_id: str | None = None
        self.worker_generation: int | None = None

        # Set only when no worker was chosen. A 503 for a model nobody serves
        # and a 503 for a cluster with no room left are the same status code and
        # entirely different problems, so the line says which.
        self.reason: str | None = None

        # Every worker's in-flight count at the instant the choice was made.
        # The field that turns "this request went to worker-b" into "this request
        # went to worker-b *because* worker-a already had three" - and, on a 503,
        # into the reason nothing was eligible.
        self.cluster_active: dict[str, int] = {}

        self._registry = registry
        self._log = log
        self._started = perf_counter()
        self._selected_at: float | None = None

    def record_selection(
        self,
        chosen: WorkerRecord | None,
        candidates: list[WorkerRecord],
        reason: str | None = None,
    ) -> None:
        """Note what the scheduler decided and what it was looking at.

        Called from inside `registry.reserve`, so it runs under the registry
        lock: it only reads the records it is handed, and the counts it copies
        are from before the chosen worker is incremented, which is what makes
        them the counts the decision was actually made on.

        `reason` belongs with nothing chosen, and is the caller's to supply: it
        comes from the scheduler, which owns the rule that produced the refusal.
        """
        self._selected_at = perf_counter()
        self.cluster_active = {worker.worker_id: worker.active_requests for worker in candidates}
        if chosen is None:
            self.reason = reason
        else:
            self.worker_id = chosen.worker_id
            self.worker_generation = chosen.generation

    def _fields(self, exc: BaseException | None) -> dict:
        now = perf_counter()
        fields: dict = {
            "event": "request",
            "model": self.model,
            "status": _status_of(exc),
            "worker_id": self.worker_id,
            "cluster_active": self.cluster_active,
            "total_ms": _ms(now - self._started),
        }
        if self.reason is not None:
            fields["reason"] = self.reason
        if self._selected_at is not None:
            fields["select_ms"] = _ms(self._selected_at - self._started)
        if self.worker_id is not None:
            fields["worker_generation"] = self.worker_generation
            # Only meaningful once there was somewhere to forward to; on a 503
            # the request never left the master.
            fields["forward_ms"] = _ms(now - self._selected_at)
        return fields

    def __enter__(self) -> "RequestTrace":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self.worker_id is not None:
            # Unconditional: a slot that is never given back leaves the worker
            # permanently looking busier than it is, and the scheduler stops
            # using it.
            self._registry.release(self.worker_id)
        self._log.info("request finished", extra={"fields": self._fields(exc)})


def _status_of(exc: BaseException | None) -> int:
    """What the client is about to be told.

    Read off the exception rather than set by the handler, so a path that raises
    cannot forget to record its own outcome.
    """
    if exc is None:
        return 200
    if isinstance(exc, HTTPException):
        return exc.status_code
    return 500
