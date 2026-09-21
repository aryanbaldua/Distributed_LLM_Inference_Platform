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


# --- health transitions -------------------------------------------------------


def test_sweep_marks_a_worker_that_stopped_heartbeating():
    registry = WorkerRegistry()
    registry.register(registration(), now=100.0)

    (transition,) = registry.sweep(timeout_s=6.0, now=110.0)

    assert transition.worker_id == "w1"
    assert transition.from_status is WorkerStatus.HEALTHY
    assert transition.to_status is WorkerStatus.UNHEALTHY
    assert "10.0s" in transition.reason
    assert registry.snapshot()[0].status is WorkerStatus.UNHEALTHY


def test_sweep_leaves_a_worker_that_is_exactly_at_the_timeout_alone():
    """The timeout is the age a worker is allowed to reach, not the one it dies at."""
    registry = WorkerRegistry()
    registry.register(registration(), now=100.0)

    assert registry.sweep(timeout_s=6.0, now=106.0) == []
    assert registry.snapshot()[0].status is WorkerStatus.HEALTHY


def test_sweep_reports_a_dead_worker_once_not_once_per_tick():
    registry = WorkerRegistry()
    registry.register(registration(), now=100.0)

    first = registry.sweep(timeout_s=6.0, now=110.0)
    second = registry.sweep(timeout_s=6.0, now=111.0)
    third = registry.sweep(timeout_s=6.0, now=1_000.0)

    assert len(first) == 1
    assert second == [] and third == []


def test_sweep_only_touches_the_workers_that_went_quiet():
    registry = WorkerRegistry()
    registry.register(registration(worker_id="alive"), now=100.0)
    registry.register(registration(worker_id="dead"), now=100.0)
    registry.heartbeat(HeartbeatRequest(worker_id="alive", active_requests=0), now=109.0)

    (transition,) = registry.sweep(timeout_s=6.0, now=110.0)

    assert transition.worker_id == "dead"
    assert {r.worker_id: r.status for r in registry.snapshot()} == {
        "alive": WorkerStatus.HEALTHY,
        "dead": WorkerStatus.UNHEALTHY,
    }


def test_a_heartbeat_revives_an_unhealthy_worker_without_reregistration():
    """The worker was only slow or unreachable; its process never died."""
    registry = WorkerRegistry()
    registry.register(registration(), now=100.0)
    registry.sweep(timeout_s=6.0, now=110.0)

    registry.heartbeat(HeartbeatRequest(worker_id="w1", active_requests=2), now=111.0)

    record = registry.snapshot()[0]
    assert record.status is WorkerStatus.HEALTHY
    assert record.generation == 1, "reviving is not rejoining"


def test_a_worker_can_report_itself_unhealthy_while_still_heartbeating():
    """Silence is not the only way to be broken: the model server can die under a
    worker whose HTTP server is perfectly happy."""
    registry = WorkerRegistry()
    registry.register(registration(), now=100.0)

    registry.heartbeat(
        HeartbeatRequest(worker_id="w1", active_requests=0, status=WorkerStatus.UNHEALTHY),
        now=101.0,
    )

    assert registry.snapshot()[0].status is WorkerStatus.UNHEALTHY


def test_a_self_reported_unhealthy_worker_stays_unhealthy_while_it_keeps_heartbeating():
    registry = WorkerRegistry()
    registry.register(registration(), now=100.0)
    sick = HeartbeatRequest(worker_id="w1", active_requests=0, status=WorkerStatus.UNHEALTHY)

    registry.heartbeat(sick, now=101.0)
    # Fresh heartbeats keep the reaper away, but must not launder it back to healthy.
    assert registry.sweep(timeout_s=6.0, now=102.0) == []
    registry.heartbeat(sick, now=103.0)
    assert registry.snapshot()[0].status is WorkerStatus.UNHEALTHY

    registry.heartbeat(HeartbeatRequest(worker_id="w1", active_requests=0), now=104.0)
    assert registry.snapshot()[0].status is WorkerStatus.HEALTHY


def test_a_restarted_worker_rejoins_healthy():
    registry = WorkerRegistry()
    registry.register(registration(), now=100.0)
    registry.sweep(timeout_s=6.0, now=110.0)

    record = registry.register(registration(), now=200.0)

    assert record.status is WorkerStatus.HEALTHY
    assert record.generation == 2


# --- eviction -----------------------------------------------------------------


def test_eviction_forgets_a_worker_that_has_been_dead_long_enough():
    registry = WorkerRegistry()
    registry.register(registration(), now=100.0)
    registry.sweep(timeout_s=6.0, now=110.0)

    assert registry.evict(evict_after_s=300.0, now=401.0) == ["w1"]
    assert registry.snapshot() == []


def test_eviction_keeps_a_dead_worker_inside_the_grace_period():
    """An unhealthy worker stays visible so the operator view can show it, and so a
    worker that recovers finds its own record."""
    registry = WorkerRegistry()
    registry.register(registration(), now=100.0)
    registry.sweep(timeout_s=6.0, now=110.0)

    assert registry.evict(evict_after_s=300.0, now=399.0) == []
    assert len(registry.snapshot()) == 1


def test_eviction_never_removes_a_worker_that_was_not_marked_dead_first():
    """Gating on UNHEALTHY is what stops eviction overtaking failure detection if
    evict_after_s is ever configured below the heartbeat timeout."""
    registry = WorkerRegistry()
    registry.register(registration(), now=100.0)

    assert registry.evict(evict_after_s=1.0, now=10_000.0) == []
    assert len(registry.snapshot()) == 1
