"""Worker selection for inference requests."""

from __future__ import annotations

import random

from common.schemas import WorkerRecord, WorkerStatus


def select(
    workers: list[WorkerRecord],
    model: str,
    timeout_s: float,
    now: float | None = None,
) -> WorkerRecord | None:
    """Choose the worker that should serve a request for `model`.

    Returns None when nothing is eligible, which the caller turns into a 503.

    Eligibility looks at the heartbeat timestamp as well as the status field.
    Status only changes when the reaper next runs, so a worker whose heartbeats
    have just stopped still reads HEALTHY for up to one tick; its timestamp
    already says otherwise, and dispatching to it would only fail.
    """
    candidates = [
        worker
        for worker in workers
        if worker.status is WorkerStatus.HEALTHY
        and worker.model == model
        and worker.heartbeat_age(now) <= timeout_s
    ]
    if not candidates:
        return None

    # Ties are broken at random rather than by a fixed order. An idle cluster is
    # nothing but ties, and resolving them the same way every time would send
    # every request to whichever worker happened to sort first.
    fewest = min(worker.active_requests for worker in candidates)
    return random.choice([worker for worker in candidates if worker.active_requests == fewest])
