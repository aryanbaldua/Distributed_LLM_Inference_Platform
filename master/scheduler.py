"""Worker selection for inference requests."""

from __future__ import annotations

import random

from common.schemas import WorkerRecord, WorkerStatus

# Why select() found nothing. Two situations that share a status code and share
# nothing else: one means the cluster is misconfigured, the other means it is
# out of room. Named here, next to the rule that decides between them.
NO_WORKER_FOR_MODEL = "no_worker_for_model"
ALL_WORKERS_AT_CAPACITY = "all_workers_at_capacity"


def _serving(
    workers: list[WorkerRecord], model: str, timeout_s: float, now: float | None
) -> list[WorkerRecord]:
    """Workers that could serve `model` if they had room.

    Eligibility looks at the heartbeat timestamp as well as the status field.
    Status only changes when the reaper next runs, so a worker whose heartbeats
    have just stopped still reads HEALTHY for up to one tick; its timestamp
    already says otherwise, and dispatching to it would only fail.
    """
    return [
        worker
        for worker in workers
        if worker.status is WorkerStatus.HEALTHY
        and worker.model == model
        and worker.heartbeat_age(now) <= timeout_s
    ]


def select(
    workers: list[WorkerRecord],
    model: str,
    timeout_s: float,
    now: float | None = None,
) -> WorkerRecord | None:
    """Choose the worker that should serve a request for `model`.

    Returns None when nothing is eligible, which the caller turns into a 503.

    A worker already holding max_concurrency requests is not eligible. Without
    that, the capacity each worker advertises at registration is a number the
    master collects and ignores, and a burst is dispatched in full however small
    the cluster is - which a mock worker absorbs happily and a real engine does
    not.
    """
    candidates = [
        worker
        for worker in _serving(workers, model, timeout_s, now)
        if worker.active_requests < worker.max_concurrency
    ]
    if not candidates:
        return None

    # Ties are broken at random rather than by a fixed order. An idle cluster is
    # nothing but ties, and resolving them the same way every time would send
    # every request to whichever worker happened to sort first.
    fewest = min(worker.active_requests for worker in candidates)
    return random.choice([worker for worker in candidates if worker.active_requests == fewest])


def refusal_reason(
    workers: list[WorkerRecord],
    model: str,
    timeout_s: float,
    now: float | None = None,
) -> str:
    """Which 503 a caller is about to send, for the log line and the message.

    Asked only once select() has returned None, and from the same records, so
    the two agree about what the cluster looked like.
    """
    if _serving(workers, model, timeout_s, now):
        return ALL_WORKERS_AT_CAPACITY
    return NO_WORKER_FOR_MODEL
