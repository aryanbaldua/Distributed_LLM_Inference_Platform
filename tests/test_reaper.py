"""The reaper: one pass in isolation, then the loop that repeats it.

The pass is tested against an injected clock. The loop is tested for the things
only a real loop can get wrong - that it keeps running after a bad tick, and
that it stops when cancelled - using a worker that is already well past its
timeout, so the loop has something to find on its very first tick.
"""

import asyncio
from time import time

import pytest

from common.config import ControllerSettings
from common.schemas import RegisterRequest, WorkerStatus
from controller.reaper import reaper_loop, sweep_once
from controller.registry import WorkerRegistry
from tests.helpers import until


def long_silent():
    """A last-heartbeat far enough back to be stale under fast_settings, but not so
    far back that it also trips the eviction grace period."""
    return time() - 10.0


def registration(worker_id="w1"):
    return RegisterRequest(
        worker_id=worker_id, address="127.0.0.1:8001", model="mock-model", max_concurrency=4
    )


def fast_settings(**overrides):
    return ControllerSettings(
        reaper_interval_s=0.01, heartbeat_timeout_s=0.01, evict_after_s=1e9, **overrides
    )


def test_one_pass_marks_the_silent_and_then_forgets_the_long_dead():
    registry = WorkerRegistry()
    registry.register(registration(), now=100.0)
    settings = ControllerSettings(heartbeat_timeout_s=6.0, evict_after_s=300.0)

    marked = sweep_once(registry, settings, now=110.0)
    assert [t.to_status for t in marked] == [WorkerStatus.UNHEALTHY]
    assert len(registry.snapshot()) == 1, "still visible while it might come back"

    assert sweep_once(registry, settings, now=500.0) == []
    assert registry.snapshot() == [], "written off once the grace period passes"


async def test_the_loop_marks_a_worker_that_stopped_heartbeating():
    registry = WorkerRegistry()
    registry.register(registration(), now=long_silent())
    task = asyncio.create_task(reaper_loop(registry, fast_settings()))

    try:
        await until(lambda: registry.snapshot()[0].status is WorkerStatus.UNHEALTHY)
    finally:
        task.cancel()


async def test_a_failing_tick_does_not_stop_failure_detection():
    """An exception escaping the loop would silently leave the cluster with no
    failure detection at all, since nothing ever awaits this task."""

    class FlakyRegistry(WorkerRegistry):
        def __init__(self):
            super().__init__()
            self.sweeps = 0

        def sweep(self, timeout_s, now=None):
            self.sweeps += 1
            if self.sweeps == 1:
                raise RuntimeError("transient failure inside a tick")
            return super().sweep(timeout_s, now=now)

    registry = FlakyRegistry()
    registry.register(registration(), now=long_silent())
    task = asyncio.create_task(reaper_loop(registry, fast_settings()))

    try:
        await until(lambda: registry.snapshot()[0].status is WorkerStatus.UNHEALTHY)
        assert registry.sweeps > 1
        assert not task.done(), "the loop survived the bad tick"
    finally:
        task.cancel()


async def test_the_loop_stops_when_cancelled():
    """Shutdown hangs if the loop swallows cancellation."""
    task = asyncio.create_task(reaper_loop(WorkerRegistry(), fast_settings()))
    await asyncio.sleep(0.02)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert task.cancelled()
