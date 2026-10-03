"""Worker selection, exercised with an injected clock so nothing sleeps."""

from common.schemas import WorkerRecord, WorkerStatus
from master.scheduler import (
    ALL_WORKERS_AT_CAPACITY,
    NO_WORKER_FOR_MODEL,
    refusal_reason,
    select,
)

NOW = 1000.0
TIMEOUT = 6.0


def worker(
    worker_id="w1",
    model="mock-model",
    status=WorkerStatus.HEALTHY,
    active_requests=0,
    last_heartbeat=NOW,
    max_concurrency=4,
):
    return WorkerRecord(
        worker_id=worker_id,
        address="127.0.0.1:8001",
        model=model,
        max_concurrency=max_concurrency,
        status=status,
        active_requests=active_requests,
        last_heartbeat=last_heartbeat,
    )


def choose(workers, model="mock-model"):
    return select(workers, model, timeout_s=TIMEOUT, now=NOW)


def why(workers, model="mock-model"):
    return refusal_reason(workers, model, timeout_s=TIMEOUT, now=NOW)


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


# --- capacity -------------------------------------------------------------------


def test_a_worker_at_capacity_is_not_eligible():
    assert choose([worker(active_requests=4, max_concurrency=4)]) is None


def test_the_last_free_slot_is_still_a_free_slot():
    assert choose([worker(active_requests=3, max_concurrency=4)]) is not None


def test_capacity_is_per_worker_rather_than_a_cluster_total():
    """A small worker filling up must not take its larger peers with it."""
    workers = [
        worker("w1", active_requests=1, max_concurrency=1),
        worker("w2", active_requests=2, max_concurrency=8),
    ]

    assert choose(workers).worker_id == "w2"


def test_a_full_worker_does_not_win_for_being_the_least_busy():
    """Fewest active requests is how the choice is made among workers that have
    room, not a reason to dispatch to one that has none."""
    workers = [
        worker("w1", active_requests=1, max_concurrency=1),
        worker("w2", active_requests=3, max_concurrency=8),
    ]

    assert choose(workers).worker_id == "w2"


# --- which refusal it was -------------------------------------------------------


def test_an_empty_cluster_is_refused_for_having_no_worker():
    assert why([]) == NO_WORKER_FOR_MODEL


def test_a_cluster_serving_only_other_models_is_refused_for_having_no_worker():
    assert why([worker(model="Qwen/Qwen2.5-1.5B-Instruct")]) == NO_WORKER_FOR_MODEL


def test_a_full_cluster_is_refused_for_capacity():
    workers = [worker("w1", active_requests=4), worker("w2", active_requests=4)]

    assert why(workers) == ALL_WORKERS_AT_CAPACITY


def test_one_free_worker_among_full_ones_is_not_a_refusal_at_all():
    workers = [worker("w1", active_requests=4), worker("w2", active_requests=0)]

    assert choose(workers).worker_id == "w2"


def test_a_full_but_dead_worker_is_a_missing_worker_rather_than_a_full_one():
    """Health is checked before capacity, so an unhealthy worker is not evidence
    that the cluster is merely busy - there is nothing to be busy."""
    workers = [worker(active_requests=4, status=WorkerStatus.UNHEALTHY)]

    assert why(workers) == NO_WORKER_FOR_MODEL


def test_a_full_but_silent_worker_is_a_missing_worker_too():
    workers = [worker(active_requests=4, last_heartbeat=NOW - 9.0)]

    assert why(workers) == NO_WORKER_FOR_MODEL
