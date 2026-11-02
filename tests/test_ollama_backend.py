"""The worker's real-model backend, against a stand-in Ollama.

Ollama is replaced with an httpx MockTransport rather than run for real: the
claim here is what the worker sends and how it treats each kind of answer, and
the suite has to keep passing on machines with no model installed.
"""

import httpx
import pytest

import worker.main as worker_main
from common.config import MasterSettings, WorkerSettings

REQUEST = {"model": "qwen2.5:0.5b", "messages": [{"role": "user", "content": "hi"}]}

COMPLETION = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 1,
    "model": "qwen2.5:0.5b",
    "system_fingerprint": "fp_ollama",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


@pytest.fixture
async def ollama_worker(monkeypatch):
    """The worker app on the ollama backend, dialling whatever handler a test sets."""
    monkeypatch.setattr(worker_main.settings, "backend", "ollama")
    monkeypatch.setattr(worker_main.settings, "model", "qwen2.5:0.5b")
    seen = []
    state = {"handler": lambda request: httpx.Response(200, json=COMPLETION)}

    def route(request):
        seen.append(request)
        return state["handler"](request)

    worker_main.app.state.http = httpx.AsyncClient(transport=httpx.MockTransport(route))

    async def post(handler=None):
        if handler is not None:
            state["handler"] = handler
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=worker_main.app), base_url="http://worker"
        ) as client:
            return await client.post("/v1/chat/completions", json=REQUEST)

    yield post, seen
    await worker_main.app.state.http.aclose()


async def test_a_request_is_answered_by_ollama(ollama_worker):
    post, seen = ollama_worker
    response = await post()
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "hello"
    assert seen[0].url == "http://127.0.0.1:11434/v1/chat/completions"


async def test_ollama_is_asked_for_the_model_this_worker_serves(ollama_worker):
    post, seen = ollama_worker
    await post()
    sent = httpx.Response(200, content=seen[0].content).json()
    assert sent["model"] == "qwen2.5:0.5b"
    assert sent["messages"] == [{"role": "user", "content": "hi"}]
    assert sent["stream"] is False


async def test_a_model_ollama_has_not_pulled_is_reported_not_dropped(ollama_worker):
    post, _ = ollama_worker
    response = await post(
        lambda request: httpx.Response(404, json={"error": {"message": "model not found"}})
    )
    assert response.status_code == 502
    assert "model not found" in response.json()["detail"]


async def test_ollama_not_running_is_an_answer_from_the_worker(ollama_worker):
    """An error status, not a dropped connection, so the master does not take
    the worker itself for dead and retry elsewhere."""
    post, _ = ollama_worker

    def refuse(request):
        raise httpx.ConnectError("connection refused")

    response = await post(refuse)
    assert response.status_code == 502
    assert "not reachable" in response.json()["detail"]


async def test_a_slow_generation_is_a_504(ollama_worker):
    post, _ = ollama_worker

    def stall(request):
        raise httpx.ReadTimeout("timed out")

    response = await post(stall)
    assert response.status_code == 504


async def test_a_body_that_is_not_a_completion_is_a_502(ollama_worker):
    post, _ = ollama_worker
    response = await post(lambda request: httpx.Response(200, json={"nope": True}))
    assert response.status_code == 502


def test_the_worker_gives_up_on_ollama_before_the_master_gives_up_on_the_worker():
    assert WorkerSettings().ollama_timeout_s < MasterSettings().forward_timeout_s
