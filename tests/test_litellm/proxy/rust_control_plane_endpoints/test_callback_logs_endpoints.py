"""Unit tests for POST /v1/callbacks/logs (replay logging payloads → callbacks)."""

import time

import pytest
from fastapi import HTTPException, Request

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
from litellm.proxy._types import LitellmUserRoles, ProxyException, UserAPIKeyAuth
from litellm.proxy.rust_control_plane_endpoints.callback_logs_endpoints import (
    CallbackLogsReplayer,
    DATA_PLANE_KEY_ENV_VAR,
    DATA_PLANE_KEY_HEADER,
    VerifyKeyRequest,
    _synthetic_request,
    ingest_callback_logs,
    require_data_plane_key,
    rust_control_plane_router,
    verify_key,
)
from litellm.types.proxy.callback_logs_endpoints import (
    CallbackLogRecord,
    CallbackLogsRequest,
)

REQ_ID = "cb-logs-unit-test-1"


def _make_request(headers: dict) -> Request:
    """Build a minimal ASGI Request with the given headers."""
    raw_headers = [
        (k.lower().encode("latin-1"), v.encode("latin-1")) for k, v in headers.items()
    ]
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/rust_control_plane/authentication",
        "headers": raw_headers,
    }
    return Request(scope)


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


def test_require_data_plane_key_500_when_env_unset(monkeypatch):
    monkeypatch.delenv(DATA_PLANE_KEY_ENV_VAR, raising=False)
    request = _make_request({DATA_PLANE_KEY_HEADER: "anything"})
    with pytest.raises(HTTPException) as exc_info:
        require_data_plane_key(request)
    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "data-plane auth not configured"


def test_require_data_plane_key_500_when_env_empty(monkeypatch):
    monkeypatch.setenv(DATA_PLANE_KEY_ENV_VAR, "")
    request = _make_request({DATA_PLANE_KEY_HEADER: "anything"})
    with pytest.raises(HTTPException) as exc_info:
        require_data_plane_key(request)
    assert exc_info.value.status_code == 500


def test_require_data_plane_key_401_when_header_missing(monkeypatch):
    monkeypatch.setenv(DATA_PLANE_KEY_ENV_VAR, "secret-dp-key")
    request = _make_request({})
    with pytest.raises(HTTPException) as exc_info:
        require_data_plane_key(request)
    assert exc_info.value.status_code == 401


def test_require_data_plane_key_401_when_header_wrong(monkeypatch):
    monkeypatch.setenv(DATA_PLANE_KEY_ENV_VAR, "secret-dp-key")
    request = _make_request({DATA_PLANE_KEY_HEADER: "wrong-key"})
    with pytest.raises(HTTPException) as exc_info:
        require_data_plane_key(request)
    assert exc_info.value.status_code == 401


def test_require_data_plane_key_does_not_accept_master_key(monkeypatch):
    """The data-plane key must be a dedicated secret, not the master key."""
    monkeypatch.setenv(DATA_PLANE_KEY_ENV_VAR, "secret-dp-key")
    monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-master-1234")
    request = _make_request({DATA_PLANE_KEY_HEADER: "sk-master-1234"})
    with pytest.raises(HTTPException) as exc_info:
        require_data_plane_key(request)
    assert exc_info.value.status_code == 401


def test_require_data_plane_key_passes_when_correct(monkeypatch):
    monkeypatch.setenv(DATA_PLANE_KEY_ENV_VAR, "secret-dp-key")
    request = _make_request({DATA_PLANE_KEY_HEADER: "secret-dp-key"})
    # Should not raise.
    assert require_data_plane_key(request) is None


def test_router_mounts_auth_verify_under_rust_control_plane():
    assert any(
        getattr(route, "path", None) == "/v1/rust_control_plane/authentication"
        for route in rust_control_plane_router.routes
    )


@pytest.mark.asyncio
async def test_synthetic_request_skips_budget_reservation():
    request = _synthetic_request(
        route="/v1/realtime",
        authorization_header="Bearer sk-test-key",
        model="gpt-realtime",
    )

    assert request.url.path == "/v1/realtime"
    assert request.state.skip_budget_reservation is True
    assert (await request.json()) == {"model": "gpt-realtime"}


@pytest.mark.asyncio
async def test_verify_key_returns_model_dump(monkeypatch):
    expected_auth = UserAPIKeyAuth(
        api_key="hashed-key", user_id="user-123", max_budget=100.0
    )

    captured = {}

    async def fake_user_api_key_auth(request, api_key):
        captured["api_key"] = api_key
        captured["request"] = request
        return expected_auth

    monkeypatch.setattr(
        "litellm.proxy.rust_control_plane_endpoints.callback_logs_endpoints.user_api_key_auth",
        fake_user_api_key_auth,
    )

    body = VerifyKeyRequest(
        api_key="sk-test-key", route="/v1/realtime", model="gpt-realtime"
    )
    result = await verify_key(body=body)

    # The key is forwarded WITH the Bearer prefix (user_api_key_auth strips it).
    assert captured["api_key"] == "Bearer sk-test-key"
    # Validation runs against a synthetic request carrying the gateway's route...
    assert captured["request"].url.path == "/v1/realtime"
    assert captured["request"].headers["authorization"] == "Bearer sk-test-key"
    # ...and the requested model in the body, so model-access checks enforce it.
    assert (await captured["request"].json())["model"] == "gpt-realtime"
    assert result == expected_auth.model_dump(exclude_none=True, mode="json")
    assert result["user_id"] == "user-123"


@pytest.mark.asyncio
async def test_verify_key_omits_model_when_absent(monkeypatch):
    captured = {}

    async def fake_user_api_key_auth(request, api_key):
        captured["request"] = request
        return UserAPIKeyAuth(api_key="hashed-key")

    monkeypatch.setattr(
        "litellm.proxy.rust_control_plane_endpoints.callback_logs_endpoints.user_api_key_auth",
        fake_user_api_key_auth,
    )

    body = VerifyKeyRequest(api_key="sk-test-key", route="/v1/realtime")
    await verify_key(body=body)
    # No model requested -> empty body, not {"model": null}.
    assert (await captured["request"].json()) == {}


@pytest.mark.asyncio
async def test_verify_key_does_not_double_prefix_existing_bearer(monkeypatch):
    captured = {}

    async def fake_user_api_key_auth(request, api_key):
        captured["api_key"] = api_key
        captured["request"] = request
        return UserAPIKeyAuth(api_key="hashed-key")

    monkeypatch.setattr(
        "litellm.proxy.rust_control_plane_endpoints.callback_logs_endpoints.user_api_key_auth",
        fake_user_api_key_auth,
    )

    body = VerifyKeyRequest(api_key="Bearer sk-test-key", route="/v1/realtime")
    await verify_key(body=body)

    assert captured["api_key"] == "Bearer sk-test-key"
    assert captured["request"].headers["authorization"] == "Bearer sk-test-key"


@pytest.mark.asyncio
async def test_verify_key_401_on_proxy_exception(monkeypatch):
    async def fake_user_api_key_auth(request, api_key):
        raise ProxyException(
            message="bad key",
            type="auth_error",
            param=None,
            code="401",
        )

    monkeypatch.setattr(
        "litellm.proxy.rust_control_plane_endpoints.callback_logs_endpoints.user_api_key_auth",
        fake_user_api_key_auth,
    )

    body = VerifyKeyRequest(api_key="sk-bad-key", route="/v1/realtime")
    with pytest.raises(HTTPException) as exc_info:
        await verify_key(body=body)
    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "invalid api key"


@pytest.mark.asyncio
async def test_verify_key_401_on_http_exception(monkeypatch):
    async def fake_user_api_key_auth(request, api_key):
        raise HTTPException(status_code=403, detail="forbidden internals")

    monkeypatch.setattr(
        "litellm.proxy.rust_control_plane_endpoints.callback_logs_endpoints.user_api_key_auth",
        fake_user_api_key_auth,
    )

    body = VerifyKeyRequest(api_key="sk-bad-key", route="/v1/realtime")
    with pytest.raises(HTTPException) as exc_info:
        await verify_key(body=body)
    assert exc_info.value.status_code == 401
    # Internals must not leak.
    assert exc_info.value.detail == "invalid api key"


@pytest.mark.asyncio
async def test_verify_key_propagates_http_5xx(monkeypatch):
    # A 5xx (e.g. DB outage) must NOT be masked as 401.
    async def fake_user_api_key_auth(request, api_key):
        raise HTTPException(status_code=503, detail="db unavailable")

    monkeypatch.setattr(
        "litellm.proxy.rust_control_plane_endpoints.callback_logs_endpoints.user_api_key_auth",
        fake_user_api_key_auth,
    )

    body = VerifyKeyRequest(api_key="sk-key", route="/v1/realtime")
    with pytest.raises(HTTPException) as exc_info:
        await verify_key(body=body)
    assert exc_info.value.status_code == 503


@pytest.mark.asyncio
async def test_verify_key_propagates_proxy_5xx(monkeypatch):
    # A ProxyException carrying a 5xx code propagates too (not converted to 401).
    async def fake_user_api_key_auth(request, api_key):
        raise ProxyException(
            message="internal", type="internal_error", param=None, code="500"
        )

    monkeypatch.setattr(
        "litellm.proxy.rust_control_plane_endpoints.callback_logs_endpoints.user_api_key_auth",
        fake_user_api_key_auth,
    )

    body = VerifyKeyRequest(api_key="sk-key", route="/v1/realtime")
    with pytest.raises(ProxyException):
        await verify_key(body=body)


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
