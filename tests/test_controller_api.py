"""Membership endpoints, driven through the app the way a worker would drive it.

The TestClient is used as a context manager on purpose: that runs the lifespan,
which is what builds the registry, so every test starts with an empty cluster.
"""

import pytest
from fastapi.testclient import TestClient

from controller.main import app, settings


@pytest.fixture
def client():
    with TestClient(app) as client:
        yield client


def register(client, worker_id="w1", address="127.0.0.1:8001", model="mock-model"):
    return client.post(
        "/workers/register",
        json={
            "worker_id": worker_id,
            "address": address,
            "model": model,
            "max_concurrency": 4,
        },
    )


def test_registration_returns_the_controllers_timing_policy():
    """The worker heartbeats on the controller's schedule, not its own config."""
    with TestClient(app) as client:
        body = register(client).json()

    assert body["heartbeat_interval_s"] == settings.heartbeat_interval_s
    assert body["heartbeat_timeout_s"] == settings.heartbeat_timeout_s


def test_cluster_view_lists_every_registered_worker(client):
    register(client, worker_id="worker-b", address="127.0.0.1:8002")
    register(client, worker_id="worker-a", address="127.0.0.1:8001")

    body = client.get("/cluster/workers").json()

    assert [w["worker_id"] for w in body["workers"]] == ["worker-a", "worker-b"]
    assert all(w["status"] == "HEALTHY" for w in body["workers"])


def test_cluster_view_timestamps_itself_so_heartbeat_age_is_computable(client):
    register(client)

    body = client.get("/cluster/workers").json()

    (worker,) = body["workers"]
    age = body["as_of"] - worker["last_heartbeat"]
    assert 0 <= age < 5


def test_heartbeat_updates_the_workers_load(client):
    register(client)

    response = client.post("/workers/heartbeat", json={"worker_id": "w1", "active_requests": 3})

    assert response.status_code == 200
    (worker,) = client.get("/cluster/workers").json()["workers"]
    assert worker["active_requests"] == 3


def test_heartbeat_from_an_unknown_worker_is_a_404(client):
    """A 404 is the signal to re-register rather than keep heartbeating into a void."""
    response = client.post(
        "/workers/heartbeat", json={"worker_id": "ghost", "active_requests": 0}
    )

    assert response.status_code == 404
    assert "register" in response.json()["detail"]


def test_a_restarted_worker_rejoins_as_a_new_generation(client):
    register(client)
    client.post("/workers/heartbeat", json={"worker_id": "w1", "active_requests": 7})

    register(client, address="127.0.0.1:8009")

    (worker,) = client.get("/cluster/workers").json()["workers"]
    assert worker["generation"] == 2
    assert worker["address"] == "127.0.0.1:8009"
    assert worker["active_requests"] == 0


def test_each_test_starts_from_an_empty_cluster(client):
    assert client.get("/cluster/workers").json()["workers"] == []
