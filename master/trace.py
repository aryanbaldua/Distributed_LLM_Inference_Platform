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
        # The last worker this request was dispatched to, which is the one the
        # line names. Kept after a failed attempt hands its slot back: a 502 that
        # does not say who failed is most of the way to useless.
        self.worker_id: str | None = None
        self.worker_generation: int | None = None

        # How many workers were dispatched to. 1 on a request that worked first
        # time, 0 on one that never found a worker at all.
        self.attempts = 0

        # Set only when no worker was chosen. A 503 for a model nobody serves
        # and a 503 for a cluster with no room left are the same status code and
        # entirely different problems, so the line says which.
        self.reason: str | None = None

        # Every worker's in-flight count at the instant the choice was made.
        # The field that turns "this request went to worker-b" into "this request
        # went to worker-b *because* worker-a already had three" - and, on a 503,
        # into the reason nothing was eligible.
        self.cluster_active: dict[str, int] = {}

        # Whose slot is reserved right now, which is not the same question as
        # which worker the line names: a retried attempt has given its slot back
        # but is still part of the story.
        self._holding: str | None = None

        self._registry = registry
        self._log = log
        self._started = perf_counter()
        # Two marks rather than one: the first is when scheduling finished, which
        # a retry must not appear to have taken longer; the latest is what the
        # forward in progress is measured from.
        self._first_selected_at: float | None = None
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
        if self._first_selected_at is None:
            self._first_selected_at = self._selected_at

        # A retry overwrites these rather than appending to them: the line
        # answers where the request ended up, and `attempts` says how hard that
        # was. Keeping every candidate set it ever looked at would be a history
        # nobody reads.
        self.cluster_active = {worker.worker_id: worker.active_requests for worker in candidates}
        if chosen is None:
            self.reason = reason
        else:
            self.attempts += 1
            self.worker_id = chosen.worker_id
            self.worker_generation = chosen.generation
            self._holding = chosen.worker_id

    def attempt_failed(self) -> None:
        """Hand back the slot of an attempt that is about to be retried elsewhere.

        Here rather than in the handler's except clause so that reserve and
        release stay paired by this class even when one request does both twice.
        """
        self._release()

    def _fields(self, exc: BaseException | None) -> dict:
        now = perf_counter()
        fields: dict = {
            "event": "request",
            "model": self.model,
            "status": _status_of(exc),
            "worker_id": self.worker_id,
            "cluster_active": self.cluster_active,
            "attempts": self.attempts,
            "total_ms": _ms(now - self._started),
        }
        if self.reason is not None:
            fields["reason"] = self.reason
        if self._first_selected_at is not None:
            fields["select_ms"] = _ms(self._first_selected_at - self._started)
        if self.worker_id is not None:
            fields["worker_generation"] = self.worker_generation
            # Only meaningful once there was somewhere to forward to; on a 503
            # the request never left the master. Measured from the latest
            # selection, so on a retried request this is the attempt that
            # counted and total_ms - forward_ms is what the retry cost.
            fields["forward_ms"] = _ms(now - self._selected_at)
        return fields

    def _release(self) -> None:
        """Give back the slot this request is holding, if it is holding one.

        Clearing `_holding` is what keeps one reserve to one release: whichever of
        a failed attempt and the exit runs second finds nothing left to give back.
        """
        if self._holding is None:
            return
        self._registry.release(self._holding)
        self._holding = None

    def __enter__(self) -> "RequestTrace":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        # Unconditional: a slot that is never given back leaves the worker
        # permanently looking busier than it is, and the scheduler stops using it.
        self._release()
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
