"""In-memory worker registry: the controller's source of truth for membership.

Every read and write of worker state goes through this class. Handlers never
touch the underlying dict, because the things that will mutate it are about to
multiply - the heartbeat reaper flips status on a timer, and request dispatch
adjusts active_requests from inside concurrent handlers.
"""

from __future__ import annotations

import threading
from time import time

from common.logging import get_logger
from common.schemas import HeartbeatRequest, RegisterRequest, WorkerRecord

log = get_logger("controller.registry")


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

        `req.status` is deliberately ignored for now. Honouring a worker's
        self-reported health is a state transition, and status transitions all
        arrive together with the timeout reaper.
        """
        now = time() if now is None else now

        with self._lock:
            record = self._workers.get(req.worker_id)
            if record is None:
                return False

            record.last_heartbeat = now
            record.active_requests = req.active_requests
            record.gpu_utilization = req.gpu_utilization
            record.free_vram_mb = req.free_vram_mb
            return True

    def snapshot(self) -> list[WorkerRecord]:
        """Copies of every record, ordered by worker_id.

        Copies, so a caller cannot mutate live registry state from outside the
        lock. Ordered, so the operator view and the demo output do not reshuffle
        between reads for no reason.
        """
        with self._lock:
            records = [record.model_copy() for record in self._workers.values()]
        return sorted(records, key=lambda record: record.worker_id)
