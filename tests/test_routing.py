"""Request routing, driven through the master app with fake workers behind it.

The master's own outbound client is swapped for one that answers in process,
so what a test asserts on is the address the master actually chose to dial.
"""

import asyncio
from collections import Counter

import httpx
import pytest

from common.schemas import HeartbeatRequest, RegisterRequest
from master.main import app as master_app
from master.registry import WorkerRegistry
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
    """Every worker the master might dial, answering as itself.

    Records the address it was reached on, which is what makes the spread of
    requests across the cluster something a test can see.
    """

    def __init__(self):
        self.delay_s = 0.0
        self.served = []
        self.headers = []
        self.unreachable = set()
        self.nonsense = set()
        self.in_flight = 0
        self.most_in_flight = 0

    async def handle(self, request):
        address = request.url.netloc.decode()
        self.served.append(address)
        self.headers.append(request.headers)
        if address in self.unreachable:
            raise httpx.ConnectError("connection refused")
        if address in self.nonsense:
            return httpx.Response(200, json={"not": "a completion"})

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
    master_app.state.registry = registry
    master_app.state.http = httpx.AsyncClient(transport=httpx.MockTransport(workers.handle))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=master_app), base_url="http://master"
    ) as client:
        yield registry, workers, client
    await master_app.state.http.aclose()


def join(registry, worker_id, address, model="mock-model", max_concurrency=4):
    registry.register(
        RegisterRequest(
            worker_id=worker_id, address=address, model=model, max_concurrency=max_concurrency
        )
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

    master_app.state.http = httpx.AsyncClient(transport=httpx.MockTransport(nonsense))

    assert (await ask(client)).status_code == 502
    assert active(registry, "worker-a") == 0


async def test_load_moves_away_from_a_worker_that_is_already_busy(cluster):
    registry, workers, client = cluster
    join(registry, "worker-a", WORKER_A)
    join(registry, "worker-b", WORKER_B)
    registry.reserve(lambda records: next(r for r in records if r.worker_id == "worker-a"))

    response = await ask(client)

    assert response.headers["x-worker-id"] == "worker-b"


# --- tracing --------------------------------------------------------------------


async def test_a_client_is_told_the_id_its_request_was_traced_under(cluster):
    registry, _, client = cluster
    join(registry, "worker-a", WORKER_A)

    response = await ask(client)

    assert response.headers["x-request-id"]


async def test_the_worker_is_dialled_with_the_id_the_client_was_given(cluster):
    """The one assertion that proves the two logs can be read together."""
    registry, workers, client = cluster
    join(registry, "worker-a", WORKER_A)

    response = await ask(client)

    assert workers.headers[0]["x-request-id"] == response.headers["x-request-id"]


async def test_an_id_the_caller_brought_is_kept_rather_than_replaced(cluster):
    """So a trace that starts outside this service stays one trace."""
    registry, workers, client = cluster
    join(registry, "worker-a", WORKER_A)

    response = await client.post(
        "/v1/chat/completions",
        json={"model": "mock-model", "messages": [{"role": "user", "content": "hello"}]},
        headers={"X-Request-Id": "from-upstream"},
    )

    assert response.headers["x-request-id"] == "from-upstream"
    assert workers.headers[0]["x-request-id"] == "from-upstream"


async def test_a_refused_request_is_still_given_an_id(cluster):
    """A 503 is the case where the id matters most: nothing downstream logged
    anything, so the master's own line is the only record that it happened.
    """
    _, _, client = cluster

    response = await ask(client)

    assert response.status_code == 503
    assert response.headers["x-request-id"]


# --- capacity -------------------------------------------------------------------


async def test_a_worker_with_no_free_slot_is_not_dialled(cluster):
    """The cap has to be enforced before the request is sent, not apologised for
    after: a worker handed more than it advertised is the thing being avoided."""
    registry, workers, client = cluster
    join(registry, "worker-a", WORKER_A, max_concurrency=1)
    registry.reserve(lambda records: records[0])

    response = await ask(client)

    assert response.status_code == 503
    assert "at capacity" in response.json()["detail"]
    assert workers.served == []


async def test_the_two_refusals_do_not_read_the_same(cluster):
    """Both are 503s and they mean opposite things, so the message has to say
    which: one is a deployment mistake, the other is a reason to add workers."""
    registry, _, client = cluster
    join(registry, "worker-a", WORKER_A, max_concurrency=1)

    unknown_model = await ask(client, model="nobody-serves-this")
    registry.reserve(lambda records: records[0])
    at_capacity = await ask(client)

    assert unknown_model.status_code == at_capacity.status_code == 503
    assert "no healthy worker" in unknown_model.json()["detail"]
    assert "at capacity" in at_capacity.json()["detail"]


async def test_a_slot_given_back_makes_the_worker_usable_again(cluster):
    """The cap tracks current load, so it must not outlive the load that caused it."""
    registry, _, client = cluster
    join(registry, "worker-a", WORKER_A, max_concurrency=1)
    registry.reserve(lambda records: records[0])

    assert (await ask(client)).status_code == 503

    registry.release("worker-a")

    assert (await ask(client)).status_code == 200


async def test_a_burst_is_capped_rather_than_dispatched_in_full(cluster):
    """Four at once against a cluster advertising two slots is two served and two
    refused - not four dispatched and two workers quietly oversubscribed.
    """
    registry, workers, client = cluster
    workers.delay_s = 0.05
    join(registry, "worker-a", WORKER_A, max_concurrency=1)
    join(registry, "worker-b", WORKER_B, max_concurrency=1)

    responses = await asyncio.gather(*(ask(client) for _ in range(4)))

    assert Counter(r.status_code for r in responses) == {200: 2, 503: 2}
    assert workers.most_in_flight <= 2, "no worker may be given more than it advertised"


# --- retrying elsewhere ---------------------------------------------------------


def busy(registry, worker_id):
    """Put one request on a worker, to make the scheduler's first pick certain.

    Several of these tests are about what happens *after* the first choice, which
    a random tie-break between two idle workers would decide for them.
    """
    registry.reserve(lambda records: next(r for r in records if r.worker_id == worker_id))


async def test_a_request_to_a_dead_worker_is_served_by_another(cluster):
    """The reaper needs up to one timeout to notice a worker has died, and until
    it does that worker still looks eligible. Every request arriving in that
    window used to fail against a cluster perfectly able to serve it."""
    registry, workers, client = cluster
    workers.unreachable = {WORKER_A}
    join(registry, "worker-a", WORKER_A)
    join(registry, "worker-b", WORKER_B)
    busy(registry, "worker-b")

    response = await ask(client)

    assert response.status_code == 200
    assert workers.served == [WORKER_A, WORKER_B], "dialled the dead one first, then recovered"
    assert response.headers["x-worker-id"] == "worker-b"


async def test_a_retry_does_not_go_back_to_the_worker_that_just_failed(cluster):
    """Otherwise the retry is a slower way of returning the same error."""
    registry, workers, client = cluster
    workers.unreachable = {WORKER_A}
    join(registry, "worker-a", WORKER_A)

    response = await ask(client)

    assert response.status_code == 502
    assert workers.served == [WORKER_A], "one worker, so there was nowhere to retry"


async def test_retrying_stops_after_one_attempt_elsewhere(cluster):
    registry, workers, client = cluster
    workers.unreachable = {WORKER_A, WORKER_B}
    join(registry, "worker-a", WORKER_A)
    join(registry, "worker-b", WORKER_B)

    response = await ask(client)

    assert response.status_code == 502
    assert sorted(workers.served) == [WORKER_A, WORKER_B], "two attempts, not three"


async def test_a_failover_leaves_nothing_in_flight_on_either_worker(cluster):
    """Two reservations and two releases. A slot leaked by the attempt that was
    abandoned is the failure mode this whole path invites."""
    registry, workers, client = cluster
    workers.unreachable = {WORKER_A}
    join(registry, "worker-a", WORKER_A)
    join(registry, "worker-b", WORKER_B)
    busy(registry, "worker-b")
    registry.release("worker-b")

    assert (await ask(client)).status_code == 200
    assert active(registry, "worker-a") == 0
    assert active(registry, "worker-b") == 0


async def test_a_worker_that_answers_badly_is_not_retried(cluster):
    """It is alive and it rejected the request, so a second worker would reject it
    the same way. Retrying only doubles the load to reach the same error."""
    registry, workers, client = cluster
    workers.nonsense = {WORKER_A}
    join(registry, "worker-a", WORKER_A)
    join(registry, "worker-b", WORKER_B)
    busy(registry, "worker-b")

    response = await ask(client)

    assert response.status_code == 502
    assert workers.served == [WORKER_A], "the healthy worker must not have been troubled"


async def test_a_retry_respects_capacity_like_any_other_dispatch(cluster):
    """The fallback worker is chosen by the scheduler, so a full one is not a
    fallback at all."""
    registry, workers, client = cluster
    workers.unreachable = {WORKER_A}
    join(registry, "worker-a", WORKER_A)
    join(registry, "worker-b", WORKER_B, max_concurrency=1)
    busy(registry, "worker-b")

    response = await ask(client)

    assert response.status_code == 502
    assert workers.served == [WORKER_A], "worker-b was full, so there was nowhere to go"
