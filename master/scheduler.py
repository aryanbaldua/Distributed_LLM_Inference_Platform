"""Worker selection for inference requests."""

from __future__ import annotations

import random

from common.schemas import WorkerRecord, WorkerStatus

# Why nothing was chosen. Two situations that share a status code and share
# nothing else: one means the cluster is misconfigured, the other means it is
# out of room. Named here, next to the rule that decides between them.
NO_WORKER_FOR_MODEL = "no_worker_for_model"
ALL_WORKERS_AT_CAPACITY = "all_workers_at_capacity"


def select(
    workers: list[WorkerRecord],
    model: str,
    timeout_s: float,
    now: float | None = None,
) -> tuple[WorkerRecord | None, str | None]:
    """Choose the worker that should serve a request for `model`.

    Returns the worker and no reason, or no worker and the reason there was
    none, which the caller turns into a 503. Both come back from one call
    because the reason is read off the very lists the choice was made from:
    asked separately, a caller could hand the two a different view of the
    cluster, or forget to ask at all, and the log would explain a refusal that
    did not happen.

    Eligibility looks at the heartbeat timestamp as well as the status field.
    Status only changes when the reaper next runs, so a worker whose heartbeats
    have just stopped still reads HEALTHY for up to one tick; its timestamp
    already says otherwise, and dispatching to it would only fail.

    A worker already holding max_concurrency requests is not eligible either.
    Without that, the capacity each worker advertises at registration is a
    number the master collects and ignores, and a burst is dispatched in full
    however small the cluster is - which a mock worker absorbs happily and a
    real engine does not.
    """
    serving = [
        worker
        for worker in workers
        if worker.status is WorkerStatus.HEALTHY
        and worker.model == model
        and worker.heartbeat_age(now) <= timeout_s
    ]
    free = [worker for worker in serving if worker.active_requests < worker.max_concurrency]
    if not free:
        # Checked in this order because health and model are checked first: a
        # worker that is full but dead is a missing worker, not a busy one.
        return None, ALL_WORKERS_AT_CAPACITY if serving else NO_WORKER_FOR_MODEL

    # Ties are broken at random rather than by a fixed order. An idle cluster is
    # nothing but ties, and resolving them the same way every time would send
    # every request to whichever worker happened to sort first.
    fewest = min(worker.active_requests for worker in free)
    return random.choice([worker for worker in free if worker.active_requests == fewest]), None
