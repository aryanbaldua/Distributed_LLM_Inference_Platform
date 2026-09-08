import pytest
from pydantic import ValidationError

from common.config import WorkerSettings
from common.schemas import (
    ChatCompletionRequest,
    HeartbeatRequest,
    RegisterRequest,
    WorkerRecord,
    WorkerStatus,
)


def test_worker_record_from_registration_defaults_to_healthy_and_idle():
    req = RegisterRequest(
        worker_id="w1", address="127.0.0.1:8001", model="mock-model", max_concurrency=4
    )
    rec = WorkerRecord.from_registration(req)
    assert rec.status is WorkerStatus.HEALTHY
    assert rec.active_requests == 0
    assert rec.max_concurrency == 4


def test_heartbeat_age_measures_against_supplied_now():
    rec = WorkerRecord(
        worker_id="w1", address="a:1", model="m", max_concurrency=1, last_heartbeat=100.0
    )
    assert rec.heartbeat_age(now=105.0) == pytest.approx(5.0)


def test_negative_active_requests_is_rejected():
    with pytest.raises(ValidationError):
        HeartbeatRequest(worker_id="w1", active_requests=-1)


def test_gpu_utilization_is_a_fraction_not_a_percentage():
    with pytest.raises(ValidationError):
        HeartbeatRequest(worker_id="w1", active_requests=0, gpu_utilization=72)


def test_chat_request_requires_at_least_one_message():
    with pytest.raises(ValidationError):
        ChatCompletionRequest(model="m", messages=[])


def test_advertised_address_rewrites_bind_all_to_loopback():
    assert WorkerSettings(host="0.0.0.0", port=8001).advertised_address == "127.0.0.1:8001"
    assert WorkerSettings(host="10.0.0.5", port=8001).advertised_address == "10.0.0.5:8001"
