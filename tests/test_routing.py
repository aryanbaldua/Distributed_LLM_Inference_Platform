"""Request routing, driven through the controller app with fake workers behind it.

The controller's own outbound client is swapped for one that answers in process,
so what a test asserts on is the address the controller actually chose to dial.
"""

import asyncio
from collections import Counter

import httpx
import pytest

from common.schemas import HeartbeatRequest, RegisterRequest
from controller.main import app as controller_app
from controller.registry import WorkerRegistry
from tests.helpers import until

WORKER_A = "127.0.0.1:8001"
WORKER_B = "127.0.0.1:8002"


def completion(address):
    return {
        "id": "chatcmpl-test",
        "model": "mock-model",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": f"from {address}"}}],
    }


class FakeWorkers:
    """Every worker the controller might dial, answering as itself.

    Records the address it was reached on, which is what makes the spread of
    requests across the cluster something a test can see.
    """

    def __init__(self):
        self.delay_s = 0.0
        self.served = []
        self.unreachable = set()
        self.in_flight = 0
        self.most_in_flight = 0

    async def handle(self, request):
        address = request.url.netloc.decode()
        self.served.append(address)
        if address in self.unreachable:
            raise httpx.ConnectError("connection refused")

        self.in_flight += 1
        self.most_in_flight = max(self.most_in_flight, self.in_flight)
        try:
            await asyncio.sleep(self.delay_s)
        finally:
            self.in_flight -= 1
        return httpx.Response(200, json=completion(address))


@pytest.fixture
async def cluster():
    registry = WorkerRegistry()
    workers = FakeWorkers()
    controller_app.state.registry = registry
    controller_app.state.http = httpx.AsyncClient(transport=httpx.MockTransport(workers.handle))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=controller_app), base_url="http://controller"
    ) as client:
        yield registry, workers, client
    await controller_app.state.http.aclose()


def join(registry, worker_id, address, model="mock-model"):
    registry.register(
        RegisterRequest(worker_id=worker_id, address=address, model=model, max_concurrency=4)
    )


async def ask(client, model="mock-model", **extra):
    return await client.post(
        "/v1/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": "hello"}], **extra},
    )


def active(registry, worker_id):
    return {record.worker_id: record.active_requests for record in registry.snapshot()}[worker_id]


# --- nothing to route to --------------------------------------------------------


async def test_an_empty_cluster_is_a_503(cluster):
    _, workers, client = cluster

    response = await ask(client)

    assert response.status_code == 503
    assert "mock-model" in response.json()["detail"]
    assert workers.served == [], "no worker should have been dialled"


async def test_a_cluster_with_nobody_serving_the_model_is_a_503(cluster):
    registry, _, client = cluster
    join(registry, "worker-a", WORKER_A, model="Qwen/Qwen2.5-1.5B-Instruct")

    assert (await ask(client)).status_code == 503


async def test_an_unhealthy_worker_is_not_routed_to(cluster):
    registry, _, client = cluster
    join(registry, "worker-a", WORKER_A)
    registry.sweep(timeout_s=0.0, now=registry.snapshot()[0].last_heartbeat + 1)

    assert (await ask(client)).status_code == 503


# --- the happy path -------------------------------------------------------------


async def test_a_request_reaches_the_one_worker_that_can_serve_it(cluster):
    registry, workers, client = cluster
    join(registry, "worker-a", WORKER_A)
    join(registry, "worker-b", WORKER_B, model="some-other-model")

    response = await ask(client)

    assert response.status_code == 200
    assert workers.served == [WORKER_A]
    assert response.headers["x-worker-id"] == "worker-a"
    assert response.json()["choices"][0]["message"]["content"] == f"from {WORKER_A}"


async def test_streaming_is_refused_rather_than_silently_ignored(cluster):
    registry, workers, client = cluster
    join(registry, "worker-a", WORKER_A)

    response = await ask(client, stream=True)

    assert response.status_code == 400
    assert workers.served == []


# --- spreading load -------------------------------------------------------------


async def test_concurrent_requests_are_spread_across_the_cluster(cluster):
    """The whole point of counting in flight requests: four at once against two
    workers is two each, not four against whichever was idle when they arrived.
    """
    registry, workers, client = cluster
    workers.delay_s = 0.05
    join(registry, "worker-a", WORKER_A)
    join(registry, "worker-b", WORKER_B)

    responses = await asyncio.gather(*(ask(client) for _ in range(4)))

    assert [r.status_code for r in responses] == [200] * 4
    assert Counter(workers.served) == {WORKER_A: 2, WORKER_B: 2}
    assert workers.most_in_flight > 1, "the requests have to overlap for this to mean anything"


async def test_sequential_requests_leave_nothing_in_flight(cluster):
    registry, _, client = cluster
    join(registry, "worker-a", WORKER_A)

    for _ in range(3):
        assert (await ask(client)).status_code == 200

    assert active(registry, "worker-a") == 0


async def test_a_heartbeat_does_not_disturb_a_request_in_flight(cluster):
    registry, workers, client = cluster
    workers.delay_s = 0.1
    join(registry, "worker-a", WORKER_A)

    in_flight = asyncio.create_task(ask(client))
    await until(lambda: active(registry, "worker-a") == 1)
    registry.heartbeat(HeartbeatRequest(worker_id="worker-a", active_requests=0))

    assert active(registry, "worker-a") == 1, "the worker's own count must not stand in"
    assert (await in_flight).status_code == 200
    assert active(registry, "worker-a") == 0


# --- failure --------------------------------------------------------------------


async def test_a_worker_that_cannot_be_reached_is_a_502(cluster):
    registry, workers, client = cluster
    workers.unreachable = {WORKER_A}
    join(registry, "worker-a", WORKER_A)

    response = await ask(client)

    assert response.status_code == 502
    assert "worker-a" in response.json()["detail"]


async def test_a_failed_request_still_gives_its_slot_back(cluster):
    """A slot leaked on the failure path makes the worker look permanently busy,
    so the scheduler quietly stops choosing it even once it recovers."""
    registry, workers, client = cluster
    workers.unreachable = {WORKER_A}
    join(registry, "worker-a", WORKER_A)

    await ask(client)

    assert active(registry, "worker-a") == 0


async def test_a_worker_answering_with_something_other_than_a_completion_is_a_502(cluster):
    registry, workers, client = cluster
    join(registry, "worker-a", WORKER_A)

    async def nonsense(request):
        return httpx.Response(200, json={"not": "a completion"})

    controller_app.state.http = httpx.AsyncClient(transport=httpx.MockTransport(nonsense))

    assert (await ask(client)).status_code == 502
    assert active(registry, "worker-a") == 0


async def test_load_moves_away_from_a_worker_that_is_already_busy(cluster):
    registry, workers, client = cluster
    join(registry, "worker-a", WORKER_A)
    join(registry, "worker-b", WORKER_B)
    registry.reserve(lambda records: next(r for r in records if r.worker_id == "worker-a"))

    response = await ask(client)

    assert response.headers["x-worker-id"] == "worker-b"
