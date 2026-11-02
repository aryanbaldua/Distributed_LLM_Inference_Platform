"""The log endpoints and the page behind the command center.

The master reaches workers through an httpx MockTransport, as in the routing
tests, so these check what the master asks each worker for and how it merges
the answers without needing worker processes.
"""

import httpx
import pytest

import worker.main as worker_main
from common.logging import LogBuffer
from common.schemas import RegisterRequest
from master import main as master_main
from master.main import app as master_app
from master.registry import WorkerRegistry


def line(msg, t, seq, **extra):
    return {"ts": "00:00:00", "t": t, "level": "INFO", "logger": "x", "msg": msg,
            "seq": seq, "boot": "b", **extra}


@pytest.fixture
async def master(monkeypatch):
    buffer = LogBuffer()
    monkeypatch.setattr(master_main, "log_buffer", buffer)
    registry = WorkerRegistry()
    worker_lines = {"127.0.0.1:8001": [line("generated", 2.0, 0, request_id="r1")]}
    asked = []

    def handle(request):
        asked.append(request)
        lines = worker_lines.get(request.url.netloc.decode())
        if lines is None:
            raise httpx.ConnectError("connection refused")
        return httpx.Response(200, json={"lines": lines})

    master_app.state.registry = registry
    master_app.state.http = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=master_app), base_url="http://master"
    ) as client:
        yield client, registry, buffer, asked
    await master_app.state.http.aclose()


def join(registry, worker_id, address):
    registry.register(RegisterRequest(worker_id=worker_id, address=address, model="m"))


async def test_master_lines_and_worker_lines_come_back_as_one_stream_in_time_order(master):
    client, registry, buffer, _ = master
    join(registry, "worker-a", "127.0.0.1:8001")
    buffer.append(line("request finished", 3.0, 0))
    buffer.append(line("master up", 1.0, 1))

    data = (await client.get("/debug/logs")).json()

    assert [(entry["source"], entry["msg"]) for entry in data["lines"]] == [
        ("master", "master up"),
        ("worker-a", "generated"),
        ("master", "request finished"),
    ]
    assert data["unreachable"] == []


async def test_a_worker_that_does_not_answer_is_named_rather_than_failing_the_call(master):
    client, registry, buffer, _ = master
    join(registry, "worker-a", "127.0.0.1:8001")
    join(registry, "worker-b", "127.0.0.1:8002")

    response = await client.get("/debug/logs")

    assert response.status_code == 200
    assert response.json()["unreachable"] == ["worker-b"]
    assert {entry["source"] for entry in response.json()["lines"]} == {"worker-a"}


async def test_an_unhealthy_worker_is_not_asked_for_its_logs(master):
    client, registry, _, asked = master
    join(registry, "worker-a", "127.0.0.1:8001")
    registry.sweep(timeout_s=0.0, now=10**10)

    data = (await client.get("/debug/logs")).json()

    assert asked == []
    assert data["unreachable"] == []


async def test_the_limit_is_passed_on_to_each_worker(master):
    client, registry, _, asked = master
    join(registry, "worker-a", "127.0.0.1:8001")

    await client.get("/debug/logs", params={"limit": 5})

    assert asked[0].url.params["limit"] == "5"


async def test_a_worker_serves_its_own_buffer(monkeypatch):
    buffer = LogBuffer()
    buffer.append(line("generated", 1.0, 0))
    monkeypatch.setattr(worker_main, "log_buffer", buffer)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=worker_main.app), base_url="http://worker"
    ) as client:
        data = (await client.get("/debug/logs")).json()

    assert [entry["msg"] for entry in data["lines"]] == ["generated"]


async def test_the_page_is_served_and_the_root_points_at_it(master):
    client, *_ = master

    page = await client.get("/ui")
    root = await client.get("/")

    assert page.status_code == 200
    assert page.headers["content-type"].startswith("text/html")
    assert root.status_code in (302, 307)
    assert root.headers["location"] == "/ui"
