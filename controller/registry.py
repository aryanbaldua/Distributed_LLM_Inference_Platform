"""In-memory worker registry: the controller's source of truth for membership.

Every read and write of worker state goes through this class. Handlers never
touch the underlying dict: the reaper mutates status on a timer while request
handlers are writing heartbeats into the same records, and request dispatch
will add another writer on top of that.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from time import time

from common.logging import get_logger
from common.schemas import HeartbeatRequest, RegisterRequest, WorkerRecord, WorkerStatus

log = get_logger("controller.registry")


@dataclass(frozen=True)
class Transition:
    """A worker changing health state.

    Returned rather than only logged so the reaper can count transitions and,
    later, emit them as metrics. Not a pydantic model and not in schemas.py:
    this never crosses a process boundary.
    """

    worker_id: str
    from_status: WorkerStatus
    to_status: WorkerStatus
    reason: str


def _log_transition(transition: Transition) -> None:
    """Every health change is logged in one format, from one place.

    These lines are the deliverable for the failure and recovery demos, not a
    side effect of them, so they are worth keeping uniform.
    """
    log.info(
        "worker %s %s -> %s (%s)",
        transition.worker_id,
        transition.from_status.value,
        transition.to_status.value,
        transition.reason,
    )


class WorkerRegistry:
    """Maps worker_id -> WorkerRecord.

    Guarded by a plain threading.Lock rather than an asyncio.Lock: no critical
    section here awaits, so there is nothing for an async lock to buy, and this
    stays correct if a caller ever runs outside the event loop.
    """

    def __init__(self) -> None:
        self._workers: dict[str, WorkerRecord] = {}
        self._lock = threading.Lock()

    def register(self, req: RegisterRequest, now: float | None = None) -> WorkerRecord:
        """Add a worker, or replace the record of one already registered.

        Re-registration builds a *fresh* record rather than patching the old one.
        A worker only re-registers because its process restarted, so the previous
        incarnation's load counters describe requests that no longer exist;
        carrying them over would leave the scheduler permanently mis-reading how
        busy this worker is. The address and model are re-read for the same
        reason - a restarted worker may come back on a different port.
        """
        now = time() if now is None else now

        with self._lock:
            previous = self._workers.get(req.worker_id)
            generation = 1 if previous is None else previous.generation + 1
            record = WorkerRecord.from_registration(req, now=now, generation=generation)
            self._workers[req.worker_id] = record
            snapshot = record.model_copy()

        if previous is None:
            log.info(
                "worker %s joined at %s serving %s (max_concurrency=%d)",
                record.worker_id,
                record.address,
                record.model,
                record.max_concurrency,
            )
        else:
            log.info(
                "worker %s re-registered as generation %d (previously %s at %s, %d active)",
                record.worker_id,
                record.generation,
                previous.status.value,
                previous.address,
                previous.active_requests,
            )
        return snapshot

    def heartbeat(self, req: HeartbeatRequest, now: float | None = None) -> bool:
        """Refresh a worker's liveness and reported load.

        Returns False when the worker_id is unknown, which the caller turns into
        a 404 so the worker knows to re-register. That happens when a worker
        process outlives its registry entry - it was evicted after a long outage
        and then came back - so the heartbeat cannot simply create the record:
        HeartbeatRequest carries no address or model to create it from.

        A worker's own opinion of its health wins over the timeout. The reaper
        can only infer death from silence, whereas a worker that is still
        talking can say directly that its model server is broken. So a heartbeat
        carrying UNHEALTHY marks the worker unhealthy even though it has just
        proved it is alive, and it stays that way: refreshing last_heartbeat
        keeps the reaper off its back, but only a HEALTHY heartbeat revives it.
        """
        now = time() if now is None else now
        transition: Transition | None = None

        with self._lock:
            record = self._workers.get(req.worker_id)
            if record is None:
                return False

            record.last_heartbeat = now
            record.active_requests = req.active_requests
            record.gpu_utilization = req.gpu_utilization
            record.free_vram_mb = req.free_vram_mb

            if record.status is not req.status:
                transition = Transition(
                    worker_id=record.worker_id,
                    from_status=record.status,
                    to_status=req.status,
                    reason=(
                        "heartbeat resumed"
                        if req.status is WorkerStatus.HEALTHY
                        else "worker reported itself unhealthy"
                    ),
                )
                record.status = req.status

        # Logged outside the lock; nothing below here touches registry state.
        if transition is not None:
            _log_transition(transition)
        return True

    def sweep(self, timeout_s: float, now: float | None = None) -> list[Transition]:
        """Mark every worker whose heartbeats have stopped.

        The only path from HEALTHY to UNHEALTHY by silence. Idempotent by
        construction - an already-unhealthy worker is skipped - so a worker that
        stays dead for an hour produces exactly one transition, not one per
        tick. Nothing is removed here; see evict.
        """
        now = time() if now is None else now
        transitions: list[Transition] = []

        with self._lock:
            for record in self._workers.values():
                if record.status is not WorkerStatus.HEALTHY:
                    continue
                age = record.heartbeat_age(now)
                if age <= timeout_s:
                    continue
                transitions.append(
                    Transition(
                        worker_id=record.worker_id,
                        from_status=record.status,
                        to_status=WorkerStatus.UNHEALTHY,
                        reason=f"no heartbeat for {age:.1f}s",
                    )
                )
                record.status = WorkerStatus.UNHEALTHY

        for transition in transitions:
            _log_transition(transition)
        return transitions

    def evict(self, evict_after_s: float, now: float | None = None) -> list[str]:
        """Forget workers that have been unreachable long enough to write off.

        Gated on being UNHEALTHY rather than on age alone, so eviction can never
        overtake failure detection however the two timeouts are configured: a
        worker has to be marked dead before it can be forgotten.
        """
        now = time() if now is None else now

        with self._lock:
            evicted = [
                worker_id
                for worker_id, record in self._workers.items()
                if record.status is WorkerStatus.UNHEALTHY
                and record.heartbeat_age(now) > evict_after_s
            ]
            for worker_id in evicted:
                del self._workers[worker_id]

        for worker_id in evicted:
            log.info("worker %s evicted after %.0fs unreachable", worker_id, evict_after_s)
        return evicted

    def snapshot(self) -> list[WorkerRecord]:
        """Copies of every record, ordered by worker_id.

        Copies, so a caller cannot mutate live registry state from outside the
        lock. Ordered, so the operator view and the demo output do not reshuffle
        between reads for no reason.
        """
        with self._lock:
            records = [record.model_copy() for record in self._workers.values()]
        return sorted(records, key=lambda record: record.worker_id)
