import asyncio
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from starlette.exceptions import HTTPException
from litellm.types.utils import GenericGuardrailAPIInputs
from litellm.proxy.guardrails.guardrail_registry import (
    guardrail_initializer_registry,
    guardrail_class_registry,
)
from litellm.proxy.guardrails.guardrail_hooks.akto.akto import AktoGuardrail


# ── Registry ──


def test_akto_in_guardrail_initializer_registry():
    assert "akto" in guardrail_initializer_registry


def test_akto_in_guardrail_class_registry():
    assert "akto" in guardrail_class_registry
    assert guardrail_class_registry["akto"] is AktoGuardrail


# ── Fixtures ──


@pytest.fixture
def akto_validate():
    return AktoGuardrail(
        akto_base_url="http://localhost:9090",
        akto_api_key="test-token",
        unreachable_fallback="fail_closed",
        guardrail_name="test-validate",
        event_hook="pre_call",
    )


@pytest.fixture
def akto_logging():
    return AktoGuardrail(
        akto_base_url="http://localhost:9090",
        akto_api_key="test-token",
        unreachable_fallback="fail_open",
        guardrail_name="test-logging",
        event_hook="logging_only",
    )


@pytest.fixture
def akto_combined():
    return AktoGuardrail(
        akto_base_url="http://localhost:9090",
        akto_api_key="test-token",
        guardrail_name="test-combined",
        event_hook=["pre_call", "logging_only"],
    )


@pytest.fixture
def sample_inputs() -> GenericGuardrailAPIInputs:
    return GenericGuardrailAPIInputs(texts=["Hello"], model="gpt-4")


@pytest.fixture
def sample_request_data() -> dict:
    return {
        "metadata": {
            "user_api_key_request_route": "/v1/chat/completions",
            "user_api_key_user_id": "user-1",
            "user_api_key_team_id": "team-1",
        },
        "proxy_server_request": {
            "url": "http://localhost:4000/v1/chat/completions",
            "headers": {
                "x-forwarded-for": "10.0.0.1",
                "content-type": "application/json",
            },
        },
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
        "model": "gpt-4",
    }


def _mock_allowed():
    mock = MagicMock(spec=httpx.Response)
    mock.status_code = 200
    mock.json.return_value = {
        "data": {"guardrailsResult": {"Allowed": True, "Reason": ""}}
    }
    return mock


def _mock_blocked(reason="PII detected"):
    mock = MagicMock(spec=httpx.Response)
    mock.status_code = 200
    mock.json.return_value = {
        "data": {"guardrailsResult": {"Allowed": False, "Reason": reason}}
    }
    return mock


# ── Init ──


def test_init_requires_base_url():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="akto_base_url is required"):
            AktoGuardrail(
                akto_base_url="",
                akto_api_key="tok",
                guardrail_name="t",
                event_hook="pre_call",
            )


def test_init_requires_api_key():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="akto_api_key is required"):
            AktoGuardrail(
                akto_base_url="http://x",
                akto_api_key="",
                guardrail_name="t",
                event_hook="pre_call",
            )


def test_init_from_env():
    with patch.dict(
        os.environ,
        {
            "AKTO_GUARDRAIL_API_BASE": "http://env:9090",
            "AKTO_API_KEY": "env-token",
            "AKTO_ACCOUNT_ID": "2000000",
            "AKTO_VXLAN_ID": "42",
        },
    ):
        g = AktoGuardrail(guardrail_name="t", event_hook="logging_only")
        assert g.akto_base_url == "http://env:9090"
        assert g.akto_api_key == "env-token"
        assert g.akto_account_id == "2000000"
        assert g.akto_vxlan_id == "42"


def test_init_defaults():
    g = AktoGuardrail(
        akto_base_url="http://x",
        akto_api_key="tok",
        guardrail_name="t",
        event_hook="pre_call",
    )
    assert g.unreachable_fallback == "fail_closed"
    assert g.guardrail_timeout == 5
    assert g.akto_account_id == "1000000"
    assert g.akto_vxlan_id == "0"


def test_background_tasks_per_instance():
    a = AktoGuardrail(
        akto_base_url="http://x",
        akto_api_key="tok",
        guardrail_name="a",
        event_hook="pre_call",
    )
    b = AktoGuardrail(
        akto_base_url="http://x",
        akto_api_key="tok",
        guardrail_name="b",
        event_hook="logging_only",
    )
    assert a.background_tasks is not b.background_tasks


# ── has_hook ──


def test_has_hook_string():
    g = AktoGuardrail(
        akto_base_url="http://x",
        akto_api_key="tok",
        guardrail_name="t",
        event_hook="pre_call",
    )
    assert g.has_hook("pre_call") is True
    assert g.has_hook("logging_only") is False


def test_has_hook_list():
    g = AktoGuardrail(
        akto_base_url="http://x",
        akto_api_key="tok",
        guardrail_name="t",
        event_hook=["pre_call", "logging_only"],
    )
    assert g.has_hook("pre_call") is True
    assert g.has_hook("logging_only") is True
    assert g.has_hook("post_call") is False


# ── Payload ──


def test_build_akto_payload(akto_validate, sample_request_data):
    payload = akto_validate.build_akto_payload(sample_request_data)

    assert payload["path"] == "/v1/chat/completions"
    assert payload["method"] == "POST"
    assert payload["akto_account_id"] == "1000000"
    assert payload["source"] == "MIRRORING"
    assert payload["contextSource"] == "AGENTIC"
    assert payload["ip"] == "10.0.0.1"

    req_body = json.loads(payload["requestPayload"])
    assert req_body["model"] == "gpt-4"
    assert req_body["messages"][0]["content"] == "Hello, how are you?"

    assert json.loads(payload["tag"]) == {
        "gen-ai": "Gen AI",
        "user_id": "user-1",
        "team_id": "team-1",
    }
    assert payload["responsePayload"] == json.dumps({})
    assert payload["time"].isdigit() and len(payload["time"]) >= 13


def test_build_akto_payload_with_response(akto_validate, sample_request_data):
    mock_resp = MagicMock()
    mock_resp.model_dump.return_value = {
        "choices": [{"message": {"content": "Fine!", "role": "assistant"}}]
    }
    payload = akto_validate.build_akto_payload(
        sample_request_data, response_obj=mock_resp
    )
    assert (
        json.loads(payload["responsePayload"])["choices"][0]["message"]["content"]
        == "Fine!"
    )


def test_build_akto_payload_custom_ids(sample_request_data):
    g = AktoGuardrail(
        akto_base_url="http://x",
        akto_api_key="tok",
        akto_account_id="9999",
        akto_vxlan_id="7",
        guardrail_name="t",
        event_hook="pre_call",
    )
    payload = g.build_akto_payload(sample_request_data)
    assert payload["akto_account_id"] == "9999"
    assert payload["akto_vxlan_id"] == "7"


def test_build_akto_payload_blocked_status(akto_validate, sample_request_data):
    payload = akto_validate.build_akto_payload(sample_request_data, status_code=403)
    assert payload["statusCode"] == "403"
    assert payload["status"] == "403"


def test_build_akto_payload_empty_data(akto_validate):
    payload = akto_validate.build_akto_payload({})
    assert json.loads(payload["requestPayload"]) == {}
    assert payload["ip"] == "0.0.0.0"


# ── Response parsing ──


def test_parse_allowed():
    assert AktoGuardrail.parse_guardrail_response(_mock_allowed()) == (True, "")


def test_parse_blocked():
    assert AktoGuardrail.parse_guardrail_response(_mock_blocked("PII")) == (
        False,
        "PII",
    )


def test_parse_missing_result():
    mock = MagicMock(spec=httpx.Response)
    mock.status_code = 200
    mock.json.return_value = {}
    assert AktoGuardrail.parse_guardrail_response(mock) == (True, "")


def test_parse_data_none():
    mock = MagicMock(spec=httpx.Response)
    mock.status_code = 200
    mock.json.return_value = {"data": None}
    assert AktoGuardrail.parse_guardrail_response(mock) == (True, "")


def test_parse_guardrails_result_not_dict():
    mock = MagicMock(spec=httpx.Response)
    mock.status_code = 200
    mock.json.return_value = {"data": {"guardrailsResult": "invalid"}}
    assert AktoGuardrail.parse_guardrail_response(mock) == (True, "")


def test_parse_non_dict():
    mock = MagicMock(spec=httpx.Response)
    mock.status_code = 200
    mock.json.return_value = "invalid"
    assert AktoGuardrail.parse_guardrail_response(mock) == (True, "")


def test_parse_error_status():
    mock = MagicMock(spec=httpx.Response)
    mock.status_code = 500
    mock.request = MagicMock()
    with pytest.raises(httpx.HTTPStatusError):
        AktoGuardrail.parse_guardrail_response(mock)


def test_parse_non_json():
    mock = MagicMock(spec=httpx.Response)
    mock.status_code = 200
    mock.request = MagicMock()
    mock.json.side_effect = json.JSONDecodeError("err", "", 0)
    with pytest.raises(httpx.RequestError):
        AktoGuardrail.parse_guardrail_response(mock)


# ── Pre-call: allowed ──


@pytest.mark.asyncio
async def test_pre_call_allowed(akto_validate, sample_inputs, sample_request_data):
    akto_validate.async_handler.post = AsyncMock(return_value=_mock_allowed())
    result = await akto_validate.apply_guardrail(
        inputs=sample_inputs, request_data=sample_request_data, input_type="request"
    )

    assert result == sample_inputs
    akto_validate.async_handler.post.assert_called_once()
    assert (
        akto_validate.async_handler.post.call_args.kwargs["params"].get("guardrails")
        == "true"
    )
    assert (
        "ingest_data" not in akto_validate.async_handler.post.call_args.kwargs["params"]
    )


# ── Pre-call: blocked ──


@pytest.mark.asyncio
async def test_pre_call_blocked(akto_validate, sample_inputs, sample_request_data):
    akto_validate.async_handler.post = AsyncMock(
        side_effect=[_mock_blocked("PII"), _mock_allowed()]
    )

    with pytest.raises(HTTPException) as exc:
        await akto_validate.apply_guardrail(
            inputs=sample_inputs, request_data=sample_request_data, input_type="request"
        )

    await asyncio.gather(*akto_validate.background_tasks)

    assert exc.value.status_code == 403
    assert akto_validate.async_handler.post.call_count == 2

    # Second call: fire-and-forget ingest of blocked request
    second = akto_validate.async_handler.post.call_args_list[1].kwargs
    assert second["params"].get("ingest_data") == "true"
    assert "guardrails" not in second["params"]
    payload = json.loads(second["data"])
    assert payload["statusCode"] == "403"
    assert json.loads(payload["responsePayload"])["x-blocked-by"] == "Akto Proxy"


# ── Pre-call: response input is no-op ──


@pytest.mark.asyncio
async def test_pre_call_response_noop(
    akto_validate, sample_inputs, sample_request_data
):
    akto_validate.async_handler.post = AsyncMock()
    result = await akto_validate.apply_guardrail(
        inputs=sample_inputs, request_data=sample_request_data, input_type="response"
    )
    assert result == sample_inputs
    akto_validate.async_handler.post.assert_not_called()


# ── Logging_only: ingestion ──


@pytest.mark.asyncio
async def test_logging_ingests(akto_logging):
    akto_logging.async_handler.post = AsyncMock(return_value=_mock_allowed())
    mock_resp = MagicMock()
    mock_resp.model_dump.return_value = {
        "choices": [{"message": {"content": "hi", "role": "assistant"}}]
    }

    kwargs = {
        "messages": [{"role": "user", "content": "original prompt"}],
        "model": "gemini/gemini-flash",
        "litellm_params": {
            "metadata": {
                "user_api_key_user_id": "u1",
                "user_api_key_team_id": "t1",
                "user_api_key_request_route": "/v1/chat/completions",
            },
            "proxy_server_request": {
                "headers": {
                    "host": "my-litellm.example.com",
                    "content-type": "application/json",
                },
            },
        },
    }
    await akto_logging.async_log_success_event(
        kwargs=kwargs, response_obj=mock_resp, start_time=None, end_time=None
    )
    await asyncio.gather(*akto_logging.background_tasks)

    akto_logging.async_handler.post.assert_called_once()
    call = akto_logging.async_handler.post.call_args.kwargs
    assert call["params"].get("ingest_data") == "true"
    assert "guardrails" not in call["params"]
    payload = json.loads(call["data"])
    assert (
        json.loads(payload["requestPayload"])["messages"][0]["content"]
        == "original prompt"
    )
    assert json.loads(payload["requestPayload"])["model"] == "gemini/gemini-flash"
    assert (
        json.loads(payload["responsePayload"])["choices"][0]["message"]["content"]
        == "hi"
    )
    assert payload["path"] == "/v1/chat/completions"
    req_headers = json.loads(payload["requestHeaders"])
    assert req_headers["host"] == "my-litellm.example.com"


@pytest.mark.asyncio
async def test_logging_skips_for_pre_call():
    g = AktoGuardrail(
        akto_base_url="http://x",
        akto_api_key="tok",
        guardrail_name="t",
        event_hook="pre_call",
    )
    g.async_handler.post = AsyncMock()
    await g.async_log_success_event(
        kwargs={}, response_obj=MagicMock(), start_time=None, end_time=None
    )
    g.async_handler.post.assert_not_called()


@pytest.mark.asyncio
async def test_failure_event_ingests(akto_logging):
    akto_logging.async_handler.post = AsyncMock(return_value=_mock_allowed())
    kwargs = {
        "messages": [{"role": "user", "content": "bad prompt"}],
        "model": "gpt-4",
        "litellm_params": {
            "metadata": {"user_api_key_request_route": "/v1/chat/completions"},
            "proxy_server_request": {},
        },
    }
    await akto_logging.async_log_failure_event(
        kwargs=kwargs, response_obj=None, start_time=None, end_time=None
    )
    await asyncio.gather(*akto_logging.background_tasks)

    akto_logging.async_handler.post.assert_called_once()
    call = akto_logging.async_handler.post.call_args.kwargs
    assert call["params"].get("ingest_data") == "true"
    payload = json.loads(call["data"])
    assert payload["statusCode"] == "500"
    assert payload["path"] == "/v1/chat/completions"


@pytest.mark.asyncio
async def test_failure_event_skips_for_pre_call():
    g = AktoGuardrail(
        akto_base_url="http://x",
        akto_api_key="tok",
        guardrail_name="t",
        event_hook="pre_call",
    )
    g.async_handler.post = AsyncMock()
    await g.async_log_failure_event(
        kwargs={}, response_obj=None, start_time=None, end_time=None
    )
    g.async_handler.post.assert_not_called()


# ── Combined mode (pre_call + logging_only) ──


@pytest.mark.asyncio
async def test_combined_pre_call_works(
    akto_combined, sample_inputs, sample_request_data
):
    akto_combined.async_handler.post = AsyncMock(return_value=_mock_allowed())
    result = await akto_combined.apply_guardrail(
        inputs=sample_inputs, request_data=sample_request_data, input_type="request"
    )
    assert result == sample_inputs
    akto_combined.async_handler.post.assert_called_once()


@pytest.mark.asyncio
async def test_combined_logging_works(akto_combined):
    akto_combined.async_handler.post = AsyncMock(return_value=_mock_allowed())
    mock_resp = MagicMock()
    mock_resp.model_dump.return_value = {"choices": []}
    await akto_combined.async_log_success_event(
        kwargs={
            "messages": [{"role": "user", "content": "hi"}],
            "model": "m",
            "litellm_params": {"metadata": {}, "proxy_server_request": {}},
        },
        response_obj=mock_resp,
        start_time=None,
        end_time=None,
    )
    await asyncio.gather(*akto_combined.background_tasks)
    akto_combined.async_handler.post.assert_called_once()


# ── Fail-open / fail-closed ──


@pytest.mark.asyncio
async def test_fail_open():
    g = AktoGuardrail(
        akto_base_url="http://x",
        akto_api_key="tok",
        unreachable_fallback="fail_open",
        guardrail_name="t",
        event_hook="pre_call",
    )
    g.async_handler.post = AsyncMock(side_effect=httpx.ConnectError("refused"))
    inputs = GenericGuardrailAPIInputs(texts=["test"], model="gpt-4")
    result = await g.apply_guardrail(
        inputs=inputs, request_data={}, input_type="request"
    )
    assert result.get("texts") == ["test"]


@pytest.mark.asyncio
async def test_fail_closed():
    g = AktoGuardrail(
        akto_base_url="http://x",
        akto_api_key="tok",
        unreachable_fallback="fail_closed",
        guardrail_name="t",
        event_hook="pre_call",
    )
    g.async_handler.post = AsyncMock(side_effect=httpx.ConnectError("refused"))
    with pytest.raises(HTTPException) as exc:
        await g.apply_guardrail(
            inputs=GenericGuardrailAPIInputs(texts=["test"], model="gpt-4"),
            request_data={},
            input_type="request",
        )
    assert exc.value.status_code == 503
    assert "internal" not in exc.value.detail.lower()


# ── Static helpers ──


def test_extract_request_path():
    assert (
        AktoGuardrail.extract_request_path(
            {"metadata": {"user_api_key_request_route": "/v1/embeddings"}}
        )
        == "/v1/embeddings"
    )
    assert AktoGuardrail.extract_request_path({}) == "/v1/chat/completions"
    assert (
        AktoGuardrail.extract_request_path({"metadata": "invalid"})
        == "/v1/chat/completions"
    )


def test_resolve_metadata_value():
    assert AktoGuardrail.resolve_metadata_value({"metadata": {"k": "v"}}, "k") == "v"
    assert (
        AktoGuardrail.resolve_metadata_value({"litellm_metadata": {"k": "v"}}, "k")
        == "v"
    )
    assert AktoGuardrail.resolve_metadata_value({}, "k") is None
    assert AktoGuardrail.resolve_metadata_value(None, "k") is None


def test_build_tag_metadata(akto_validate, sample_request_data):
    assert akto_validate.build_tag_metadata(sample_request_data) == {
        "gen-ai": "Gen AI",
        "user_id": "user-1",
        "team_id": "team-1",
    }


def test_build_request_headers_default_host():
    headers = AktoGuardrail.build_request_headers({})
    assert headers["host"] == "litellm.ai"
    assert headers["content-type"] == "application/json"


def test_build_request_headers_from_proxy():
    headers = AktoGuardrail.build_request_headers(
        {
            "proxy_server_request": {
                "headers": {
                    "host": "myhost.com",
                    "content-type": "application/json",
                    "user-agent": "test-agent",
                }
            }
        }
    )
    assert headers["host"] == "myhost.com"
    assert headers["user-agent"] == "test-agent"


def test_extract_client_ip_from_forwarded():
    assert (
        AktoGuardrail.extract_client_ip(
            {
                "proxy_server_request": {
                    "headers": {"x-forwarded-for": "1.2.3.4, 5.6.7.8"}
                }
            }
        )
        == "1.2.3.4"
    )


def test_extract_client_ip_no_headers_fallback():
    assert (
        AktoGuardrail.extract_client_ip(
            {"metadata": {"user_api_key_end_user_id": "end-user-123"}}
        )
        == "0.0.0.0"
    )


def test_extract_client_ip_fallback():
    assert AktoGuardrail.extract_client_ip({}) == "0.0.0.0"
