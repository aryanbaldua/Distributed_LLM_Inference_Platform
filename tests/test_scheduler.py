"""Worker selection, exercised with an injected clock so nothing sleeps."""

from common.schemas import WorkerRecord, WorkerStatus
from controller.scheduler import select

NOW = 1000.0
TIMEOUT = 6.0


def worker(
    worker_id="w1",
    model="mock-model",
    status=WorkerStatus.HEALTHY,
    active_requests=0,
    last_heartbeat=NOW,
):
    return WorkerRecord(
        worker_id=worker_id,
        address="127.0.0.1:8001",
        model=model,
        max_concurrency=4,
        status=status,
        active_requests=active_requests,
        last_heartbeat=last_heartbeat,
    )


def choose(workers, model="mock-model"):
    return select(workers, model, timeout_s=TIMEOUT, now=NOW)


def test_an_empty_cluster_yields_no_worker():
    assert choose([]) is None


def test_unhealthy_workers_are_not_eligible():
    assert choose([worker(status=WorkerStatus.UNHEALTHY)]) is None


def test_workers_serving_another_model_are_not_eligible():
    assert choose([worker(model="Qwen/Qwen2.5-1.5B-Instruct")]) is None


def test_a_silent_worker_is_skipped_before_the_reaper_has_marked_it():
    """Status changes on a timer, so it lags the timestamp it is derived from."""
    assert choose([worker(last_heartbeat=NOW - 9.0)]) is None


def test_a_heartbeat_exactly_at_the_timeout_is_still_eligible():
    assert choose([worker(last_heartbeat=NOW - TIMEOUT)]) is not None


def test_the_least_busy_eligible_worker_wins():
    workers = [worker("w1", active_requests=3), worker("w2", active_requests=1)]

    assert choose(workers).worker_id == "w2"


def test_load_is_only_compared_between_eligible_workers():
    """An idle worker that cannot serve the request must not shut out a busy one."""
    workers = [
        worker("w1", active_requests=3),
        worker("w2", active_requests=0, model="other-model"),
        worker("w3", active_requests=5, status=WorkerStatus.UNHEALTHY),
    ]

    assert choose(workers).worker_id == "w1"


def test_ties_do_not_always_resolve_to_the_same_worker():
    workers = [worker("w1"), worker("w2")]

    picked = {choose(workers).worker_id for _ in range(50)}

    assert picked == {"w1", "w2"}


def test_ties_are_broken_only_among_the_least_busy():
    workers = [worker("w1"), worker("w2"), worker("w3", active_requests=5)]

    picked = {choose(workers).worker_id for _ in range(50)}

    assert picked == {"w1", "w2"}
