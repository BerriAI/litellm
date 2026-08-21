"""Unit tests for POST /v1/callbacks/logs (replay logging payloads → callbacks)."""

import time

import pytest
from fastapi import HTTPException

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.logging_endpoints.callback_logs_endpoints import (
    CallbackLogsReplayer,
    ingest_callback_logs,
)
from litellm.types.proxy.callback_logs_endpoints import (
    CallbackLogRecord,
    CallbackLogsRequest,
)

REQ_ID = "cb-logs-unit-test-1"


def _sample_payload(**overrides):
    payload = {
        "id": REQ_ID,
        "litellm_call_id": REQ_ID,
        "call_type": "acompletion",
        "stream": False,
        "response_cost": 0.0123,
        "custom_llm_provider": "openai",
        "total_tokens": 42,
        "prompt_tokens": 30,
        "completion_tokens": 12,
        "startTime": time.time() - 2,
        "endTime": time.time(),
        "model": "gpt-4o-mini",
        "metadata": {
            "user_api_key_hash": "rust-gateway-test-key",
            "user_api_key_user_id": "user-cb-logs-test",
            "user_api_key_team_id": "team-cb-logs-test",
        },
        "messages": [{"role": "user", "content": "hi"}],
    }
    payload.update(overrides)
    return payload


def test_epoch_to_datetime_handles_float_and_fallback():
    dt = CallbackLogsReplayer._epoch_to_datetime(1_700_000_000.5)
    assert dt.year == 2023
    # Non-numeric input must not raise — falls back to "now".
    assert CallbackLogsReplayer._epoch_to_datetime(None) is not None


def test_build_logging_obj_seeds_model_call_details():
    obj = CallbackLogsReplayer._build_logging_obj(_sample_payload())
    details = obj.model_call_details
    # Prebuilt payload is set so the handler skips rebuilding it.
    assert details["standard_logging_object"]["id"] == REQ_ID
    assert details["response_cost"] == 0.0123
    assert details["call_type"] == "acompletion"
    # Metadata is mapped to the keys the cost-tracking callback reads.
    md = details["litellm_params"]["metadata"]
    assert md["user_api_key"] == "rust-gateway-test-key"
    assert md["user_api_key_user_id"] == "user-cb-logs-test"
    assert md["user_api_key_team_id"] == "team-cb-logs-test"


def test_response_obj_carries_usage():
    obj = CallbackLogsReplayer._response_obj_from_payload(_sample_payload())
    assert obj["usage"]["total_tokens"] == 42
    assert obj["usage"]["prompt_tokens"] == 30
    assert obj["usage"]["completion_tokens"] == 12


@pytest.mark.asyncio
async def test_success_record_invokes_success_handler(monkeypatch):
    captured = {}

    async def fake_success(self, result=None, start_time=None, end_time=None, **kwargs):
        captured["standard_logging_object"] = self.model_call_details.get(
            "standard_logging_object"
        )
        captured["result"] = result

    monkeypatch.setattr(LiteLLMLogging, "async_success_handler", fake_success)

    body = CallbackLogsRequest(
        records=[
            CallbackLogRecord(
                status="success", standard_logging_payload=_sample_payload()
            )
        ]
    )
    resp = await ingest_callback_logs(
        body, user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)
    )
    assert resp.processed == 1 and resp.failed == 0
    assert captured["standard_logging_object"]["id"] == REQ_ID
    assert captured["result"]["usage"]["total_tokens"] == 42


@pytest.mark.asyncio
async def test_failure_record_invokes_failure_handler(monkeypatch):
    captured = {}

    async def fake_failure(
        self, exception, traceback_exception, start_time=None, end_time=None
    ):
        captured["exception"] = str(exception)

    monkeypatch.setattr(LiteLLMLogging, "async_failure_handler", fake_failure)

    body = CallbackLogsRequest(
        records=[
            CallbackLogRecord(
                status="failure",
                standard_logging_payload=_sample_payload(),
                error="upstream exploded",
            )
        ]
    )
    resp = await ingest_callback_logs(
        body, user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)
    )
    assert resp.processed == 1 and resp.failed == 0
    assert captured["exception"] == "upstream exploded"


@pytest.mark.asyncio
async def test_non_admin_is_rejected(monkeypatch):
    async def fake_success(self, **kwargs):
        return None

    monkeypatch.setattr(LiteLLMLogging, "async_success_handler", fake_success)

    body = CallbackLogsRequest(
        records=[
            CallbackLogRecord(
                status="success", standard_logging_payload=_sample_payload()
            )
        ]
    )
    with pytest.raises(HTTPException) as exc_info:
        await ingest_callback_logs(
            body,
            user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER),
        )
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_one_bad_record_does_not_sink_the_batch(monkeypatch):
    calls = {"n": 0}

    async def flaky_success(
        self, result=None, start_time=None, end_time=None, **kwargs
    ):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("boom on first record")

    monkeypatch.setattr(LiteLLMLogging, "async_success_handler", flaky_success)

    body = CallbackLogsRequest(
        records=[
            CallbackLogRecord(
                status="success", standard_logging_payload=_sample_payload()
            ),
            CallbackLogRecord(
                status="success", standard_logging_payload=_sample_payload()
            ),
        ]
    )
    resp = await ingest_callback_logs(
        body, user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)
    )
    assert resp.processed == 1 and resp.failed == 1
    # The failed record is reported back by index + error, not silently dropped.
    assert len(resp.failures) == 1
    assert resp.failures[0].index == 0
    assert "boom on first record" in resp.failures[0].error


def test_batch_over_limit_is_rejected():
    from litellm.constants import MAX_CALLBACK_LOG_RECORDS
    from pydantic import ValidationError

    # One over the cap must fail validation (422 at the API boundary), bounding
    # the callback/DB fan-out a single POST can trigger.
    too_many = [
        CallbackLogRecord(status="success", standard_logging_payload=_sample_payload())
        for _ in range(MAX_CALLBACK_LOG_RECORDS + 1)
    ]
    with pytest.raises(ValidationError):
        CallbackLogsRequest(records=too_many)
