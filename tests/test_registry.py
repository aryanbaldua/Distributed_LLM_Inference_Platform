"""Registry behaviour, exercised with an injected clock so nothing sleeps."""

from common.schemas import HeartbeatRequest, RegisterRequest, WorkerStatus
from controller.registry import WorkerRegistry


def registration(worker_id="w1", address="127.0.0.1:8001", model="mock-model", max_concurrency=4):
    return RegisterRequest(
        worker_id=worker_id, address=address, model=model, max_concurrency=max_concurrency
    )


def test_registering_a_new_worker_starts_it_healthy_idle_and_at_generation_one():
    registry = WorkerRegistry()

    record = registry.register(registration(), now=100.0)

    assert record.generation == 1
    assert record.status is WorkerStatus.HEALTHY
    assert record.active_requests == 0
    assert record.last_heartbeat == 100.0


def test_reregistering_bumps_the_generation():
    registry = WorkerRegistry()
    registry.register(registration(), now=100.0)

    record = registry.register(registration(), now=200.0)

    assert record.generation == 2
    assert len(registry.snapshot()) == 1, "re-registration upserts, it does not duplicate"


def test_reregistering_discards_the_previous_incarnations_load():
    """A worker only re-registers because it restarted, so its old active_requests
    count describes requests that died with the old process."""
    registry = WorkerRegistry()
    registry.register(registration(), now=100.0)
    registry.heartbeat(HeartbeatRequest(worker_id="w1", active_requests=5), now=110.0)

    record = registry.register(registration(), now=200.0)

    assert record.active_requests == 0


def test_reregistering_picks_up_a_changed_address_and_model():
    registry = WorkerRegistry()
    registry.register(registration(address="127.0.0.1:8001", model="mock-model"), now=100.0)

    record = registry.register(
        registration(address="127.0.0.1:9999", model="Qwen/Qwen2.5-1.5B-Instruct"), now=200.0
    )

    assert record.address == "127.0.0.1:9999"
    assert record.model == "Qwen/Qwen2.5-1.5B-Instruct"


def test_heartbeat_from_an_unregistered_worker_is_rejected():
    registry = WorkerRegistry()

    assert registry.heartbeat(HeartbeatRequest(worker_id="ghost", active_requests=0)) is False


def test_heartbeat_refreshes_liveness_and_reported_load():
    registry = WorkerRegistry()
    registry.register(registration(), now=100.0)

    accepted = registry.heartbeat(
        HeartbeatRequest(
            worker_id="w1", active_requests=3, gpu_utilization=0.72, free_vram_mb=8192
        ),
        now=142.0,
    )

    (record,) = registry.snapshot()
    assert accepted is True
    assert record.last_heartbeat == 142.0
    assert record.active_requests == 3
    assert record.gpu_utilization == 0.72
    assert record.free_vram_mb == 8192


def test_snapshot_hands_back_copies_not_live_records():
    registry = WorkerRegistry()
    registry.register(registration(), now=100.0)

    (leaked,) = registry.snapshot()
    leaked.active_requests = 999

    (actual,) = registry.snapshot()
    assert actual.active_requests == 0


def test_snapshot_is_ordered_by_worker_id():
    registry = WorkerRegistry()
    for worker_id in ("w3", "w1", "w2"):
        registry.register(registration(worker_id=worker_id), now=100.0)

    assert [record.worker_id for record in registry.snapshot()] == ["w1", "w2", "w3"]
