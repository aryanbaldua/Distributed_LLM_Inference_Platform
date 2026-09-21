"""The worker's membership loops, driven against the real controller app.

Reached over ASGI rather than a socket, but they are the actual endpoints, not a
hand-written double - the contract between the two services is the thing being
claimed here, so a double would only prove the worker agrees with itself.
"""

import asyncio
from time import time

import httpx
import pytest
from fastapi.testclient import TestClient

import worker.main as worker_main
from common.config import WorkerSettings
from common.schemas import WorkerStatus
from controller.main import app as controller_app
from controller.main import settings as controller_settings
from controller.registry import WorkerRegistry
from tests.helpers import until
from worker.controller_client import ControllerClient, HeartbeatOutcome

POLICY = {"heartbeat_interval_s": 0.01, "heartbeat_timeout_s": 0.03}


def worker_settings(**overrides):
    return WorkerSettings(
        worker_id="worker-a",
        host="127.0.0.1",
        port=8001,
        model="mock-model",
        max_concurrency=4,
        controller_url="http://controller",
        register_retry_s=0.001,
        register_backoff_max_s=0.004,
        **overrides,
    )


class RecordingLog:
    """Captures log calls. caplog cannot see these: get_logger sets
    propagate=False, so records never reach the root logger."""

    def __init__(self):
        self.lines = []

    def _record(self, level, message, *args):
        self.lines.append((level, message % args if args else message))

    def info(self, message, *args):
        self._record("info", message, *args)

    def warning(self, message, *args):
        self._record("warning", message, *args)


@pytest.fixture
async def cluster():
    registry = WorkerRegistry()
    controller_app.state.registry = registry
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=controller_app), base_url="http://controller"
    ) as http:
        yield registry, http


def forget(registry):
    """What the controller does to a worker it has written off, from the outside."""
    written_off = time() + 1
    registry.sweep(timeout_s=0.0, now=written_off)
    registry.evict(evict_after_s=0.0, now=written_off)


# --- registration -------------------------------------------------------------


async def test_a_worker_registers_itself_into_the_cluster(cluster):
    registry, http = cluster
    client = ControllerClient(worker_settings(), http)

    policy = await client.register_until_accepted()

    (record,) = registry.snapshot()
    assert record.worker_id == "worker-a"
    assert record.address == "127.0.0.1:8001", "the dialable address, not the bind address"
    assert record.model == "mock-model"
    assert policy.heartbeat_interval_s == controller_settings.heartbeat_interval_s


async def test_registration_waits_out_a_controller_that_is_not_up_yet():
    """Start order must not matter: a worker launched first has to survive it."""
    attempts = []

    def handler(request):
        attempts.append(request)
        if len(attempts) < 3:
            raise httpx.ConnectError("connection refused")
        return httpx.Response(200, json=POLICY)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        policy = await ControllerClient(worker_settings(), http).register_until_accepted()

    assert len(attempts) == 3
    assert policy.heartbeat_interval_s == POLICY["heartbeat_interval_s"]


def test_backoff_grows_then_settles_at_the_ceiling():
    settings = WorkerSettings(register_retry_s=1.0, register_backoff_max_s=8.0)
    client = ControllerClient(settings, http=None)  # backoff_delay does no I/O

    for attempt, ceiling in enumerate([1.0, 2.0, 4.0, 8.0, 8.0, 8.0], start=1):
        delays = [client.backoff_delay(attempt) for _ in range(50)]
        assert all(ceiling * 0.5 <= delay <= ceiling for delay in delays)
        assert len(set(delays)) > 1, "jittered, so co-launched workers do not retry in step"


# --- heartbeats ---------------------------------------------------------------


async def test_a_heartbeat_revives_a_worker_the_controller_gave_up_on(cluster):
    registry, http = cluster
    client = ControllerClient(worker_settings(), http)
    await client.register_until_accepted()
    registry.sweep(timeout_s=0.0, now=time() + 1)
    assert registry.snapshot()[0].status is WorkerStatus.UNHEALTHY

    assert await client.send_heartbeat() is HeartbeatOutcome.ACCEPTED

    record = registry.snapshot()[0]
    assert record.status is WorkerStatus.HEALTHY
    assert record.generation == 1, "it recovered; it did not restart"


async def test_a_worker_notices_the_controller_has_forgotten_it(cluster):
    registry, http = cluster
    client = ControllerClient(worker_settings(), http)
    await client.register_until_accepted()
    forget(registry)

    assert await client.send_heartbeat() is HeartbeatOutcome.UNKNOWN_WORKER


async def test_the_loop_registers_again_after_being_forgotten(cluster, monkeypatch):
    """A worker process can outlive its registry entry. Only registration can put
    it back, since a heartbeat carries no address or model."""
    registry, http = cluster
    monkeypatch.setattr(controller_settings, "heartbeat_interval_s", 0.01)
    task = asyncio.create_task(ControllerClient(worker_settings(), http).run())

    try:
        await until(lambda: registry.snapshot() != [])
        forget(registry)
        assert registry.snapshot() == []

        await until(lambda: registry.snapshot() != [])
    finally:
        task.cancel()


async def test_a_controller_outage_does_not_take_the_worker_down_with_it():
    beats = []

    def handler(request):
        if request.url.path == "/workers/register":
            return httpx.Response(200, json=POLICY)
        beats.append(request)
        raise httpx.ConnectError("controller went away")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        task = asyncio.create_task(ControllerClient(worker_settings(), http).run())
        try:
            await until(lambda: len(beats) >= 3)
            assert not task.done(), "it keeps trying rather than exiting"
        finally:
            task.cancel()


async def test_a_long_outage_is_reported_once_and_so_is_the_recovery():
    """At a 2s interval, a controller down for ten minutes is three hundred
    identical lines burying every transition worth reading."""
    reachable = False

    def handler(request):
        if request.url.path == "/workers/register":
            return httpx.Response(200, json=POLICY)
        if not reachable:
            raise httpx.ConnectError("controller went away")
        return httpx.Response(200, json={"ok": True})

    recorder = RecordingLog()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = ControllerClient(worker_settings(), http, recorder)
        await client.register_until_accepted()
        for _ in range(5):
            await client.send_heartbeat()
        reachable = True
        await client.send_heartbeat()

    assert [level for level, _ in recorder.lines].count("warning") == 1
    assert sum("getting through again" in line for _, line in recorder.lines) == 1


# --- wiring -------------------------------------------------------------------


def test_the_worker_app_starts_its_membership_task(monkeypatch):
    """Every other test here drives ControllerClient directly, so all of them would
    still pass if lifespan never started it."""
    monkeypatch.setattr(worker_main.settings, "controller_url", "http://127.0.0.1:1")
    monkeypatch.setattr(worker_main.settings, "register_retry_s", 0.01)
    monkeypatch.setattr(worker_main.settings, "register_backoff_max_s", 0.01)

    with TestClient(worker_main.app) as client:
        assert client.get("/healthz").json()["worker_id"] == worker_main.settings.worker_id
        assert not worker_main.app.state.membership.done()
