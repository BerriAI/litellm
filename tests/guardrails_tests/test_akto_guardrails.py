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


# ---------------------------------------------------------------------------
#  Registry tests
# ---------------------------------------------------------------------------


def test_akto_in_guardrail_initializer_registry():
    assert "akto" in guardrail_initializer_registry


def test_akto_in_guardrail_class_registry():
    assert "akto" in guardrail_class_registry
    assert guardrail_class_registry["akto"] is AktoGuardrail


# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def akto_validate():
    """AktoGuardrail configured for pre_call (akto-validate)."""
    return AktoGuardrail(
        akto_base_url="http://localhost:9090",
        akto_api_key="test-token",
        unreachable_fallback="fail_closed",
        guardrail_name="test-akto-validate",
        event_hook="pre_call",
    )


@pytest.fixture
def akto_ingest():
    """AktoGuardrail configured for post_call (akto-ingest)."""
    return AktoGuardrail(
        akto_base_url="http://localhost:9090",
        akto_api_key="test-token",
        unreachable_fallback="fail_open",
        guardrail_name="test-akto-ingest",
        event_hook="post_call",
    )


@pytest.fixture
def sample_inputs() -> GenericGuardrailAPIInputs:
    return GenericGuardrailAPIInputs(
        texts=["Hello, how are you?"],
        model="gpt-4",
    )


@pytest.fixture
def sample_request_data() -> dict:
    return {
        "metadata": {
            "user_api_key_request_route": "/v1/chat/completions",
            "user_api_key": "sk-test-123",
            "user_api_key_user_id": "user-1",
            "user_api_key_team_id": "team-1",
        },
        "proxy_server_request": {
            "headers": {
                "x-forwarded-for": "10.0.0.1",
            }
        },
    }


def _mock_allowed_response():
    mock = MagicMock(spec=httpx.Response)
    mock.status_code = 200
    mock.json.return_value = {
        "data": {"guardrailsResult": {"Allowed": True, "Reason": ""}}
    }
    return mock


def _mock_blocked_response(reason="Prompt injection detected"):
    mock = MagicMock(spec=httpx.Response)
    mock.status_code = 200
    mock.json.return_value = {
        "data": {"guardrailsResult": {"Allowed": False, "Reason": reason}}
    }
    return mock


# ---------------------------------------------------------------------------
#  Initialization tests
# ---------------------------------------------------------------------------


def test_init_requires_akto_base_url():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="akto_base_url is required"):
            AktoGuardrail(
                akto_base_url="",
                akto_api_key="test-token",
                guardrail_name="test",
                event_hook="pre_call",
            )


def test_init_requires_api_key():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="akto_api_key is required"):
            AktoGuardrail(
                akto_base_url="http://localhost:9090",
                akto_api_key="",
                guardrail_name="test",
                event_hook="pre_call",
            )


def test_init_from_env():
    with patch.dict(
        os.environ,
        {
            "AKTO_GUARDRAIL_API_BASE": "http://env-host:9090",
            "AKTO_API_KEY": "env-token",
            "AKTO_ACCOUNT_ID": "2000000",
            "AKTO_VXLAN_ID": "42",
        },
    ):
        g = AktoGuardrail(guardrail_name="env-test", event_hook="post_call")
        assert g.akto_base_url == "http://env-host:9090"
        assert g.akto_api_key == "env-token"
        assert g.guardrail_timeout == 5
        assert g.akto_account_id == "2000000"
        assert g.akto_vxlan_id == "42"


def test_init_defaults():
    g = AktoGuardrail(
        akto_base_url="http://localhost:9090",
        akto_api_key="test-token",
        guardrail_name="default-test",
        event_hook="pre_call",
    )
    assert g.unreachable_fallback == "fail_closed"
    assert g.guardrail_timeout == 5
    assert g.akto_account_id == "1000000"
    assert g.akto_vxlan_id == "0"


def test_background_tasks_per_instance():
    a = AktoGuardrail(
        akto_base_url="http://localhost:9090",
        akto_api_key="test-token",
        guardrail_name="instance-a",
        event_hook="pre_call",
    )
    b = AktoGuardrail(
        akto_base_url="http://localhost:9090",
        akto_api_key="test-token",
        guardrail_name="instance-b",
        event_hook="post_call",
    )
    assert a.background_tasks is not b.background_tasks


# ---------------------------------------------------------------------------
#  Payload format tests
# ---------------------------------------------------------------------------


def test_build_akto_payload_format(akto_validate, sample_inputs, sample_request_data):
    payload = akto_validate.build_akto_payload(
        sample_inputs, sample_request_data, include_response=False
    )

    assert payload["path"] == "/v1/chat/completions"
    assert payload["method"] == "POST"
    assert payload["type"] == "HTTP/1.1"
    assert payload["akto_account_id"] == "1000000"
    assert payload["akto_vxlan_id"] == "0"
    assert payload["is_pending"] == "false"
    assert payload["source"] == "MIRRORING"
    assert payload["contextSource"] == "AGENTIC"
    assert payload["ip"] == "10.0.0.1"

    req_headers = json.loads(payload["requestHeaders"])
    assert "content-type" in req_headers

    req_wrapper = json.loads(payload["requestPayload"])
    req_body = json.loads(req_wrapper["body"])
    assert req_body["model"] == "gpt-4"
    assert req_body["messages"][0]["content"] == "Hello, how are you?"

    tag = json.loads(payload["tag"])
    assert tag["gen-ai"] == "Gen AI"

    assert payload["responsePayload"] == json.dumps({})
    assert payload["time"].isdigit()
    assert len(payload["time"]) >= 13


def test_build_akto_payload_with_response(
    akto_validate, sample_inputs, sample_request_data
):
    payload = akto_validate.build_akto_payload(
        sample_inputs, sample_request_data, include_response=True
    )
    resp_wrapper = json.loads(payload["responsePayload"])
    resp_body = json.loads(resp_wrapper["body"])
    assert "choices" in resp_body


def test_build_akto_payload_custom_account_ids(sample_inputs, sample_request_data):
    g = AktoGuardrail(
        akto_base_url="http://localhost:9090",
        akto_api_key="test-token",
        akto_account_id="9999",
        akto_vxlan_id="7",
        guardrail_name="custom-ids-test",
        event_hook="pre_call",
    )
    payload = g.build_akto_payload(
        sample_inputs, sample_request_data, include_response=False
    )
    assert payload["akto_account_id"] == "9999"
    assert payload["akto_vxlan_id"] == "7"


def test_build_query_params():
    params = AktoGuardrail.build_query_params(guardrails=True, ingest_data=False)
    assert params == {"akto_connector": "litellm", "guardrails": "true"}

    params = AktoGuardrail.build_query_params(guardrails=False, ingest_data=True)
    assert params == {"akto_connector": "litellm", "ingest_data": "true"}

    params = AktoGuardrail.build_query_params(guardrails=True, ingest_data=True)
    assert params == {
        "akto_connector": "litellm",
        "guardrails": "true",
        "ingest_data": "true",
    }


# ---------------------------------------------------------------------------
#  Guardrail response handling
# ---------------------------------------------------------------------------


def test_handle_guardrail_response_allowed():
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "data": {"guardrailsResult": {"Allowed": True, "Reason": ""}}
    }
    allowed, reason = AktoGuardrail.handle_guardrail_response(mock_resp)
    assert allowed is True
    assert reason == ""


def test_handle_guardrail_response_blocked():
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "data": {"guardrailsResult": {"Allowed": False, "Reason": "PII detected"}}
    }
    allowed, reason = AktoGuardrail.handle_guardrail_response(mock_resp)
    assert allowed is False
    assert reason == "PII detected"


def test_handle_guardrail_response_missing_result():
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {}
    allowed, _ = AktoGuardrail.handle_guardrail_response(mock_resp)
    assert allowed is True


def test_handle_guardrail_response_data_none():
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"data": None}
    allowed, reason = AktoGuardrail.handle_guardrail_response(mock_resp)
    assert allowed is True
    assert reason == ""


def test_handle_guardrail_response_guardrails_result_not_dict():
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"data": {"guardrailsResult": "invalid"}}
    allowed, reason = AktoGuardrail.handle_guardrail_response(mock_resp)
    assert allowed is True
    assert reason == ""


def test_handle_guardrail_response_non_dict():
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = "invalid"
    allowed, _ = AktoGuardrail.handle_guardrail_response(mock_resp)
    assert allowed is True


def test_handle_guardrail_response_error_status():
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 500
    mock_resp.request = MagicMock()
    with pytest.raises(httpx.HTTPStatusError):
        AktoGuardrail.handle_guardrail_response(mock_resp)


def test_handle_guardrail_response_non_json_body():
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.request = MagicMock()
    mock_resp.text = "<html>not json</html>"
    mock_resp.json.side_effect = json.JSONDecodeError("Expecting value", "<html>", 0)

    with pytest.raises(httpx.RequestError):
        AktoGuardrail.handle_guardrail_response(mock_resp)


# ---------------------------------------------------------------------------
#  Pre-call (akto-validate) — allowed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pre_call_allowed(akto_validate, sample_inputs, sample_request_data):
    akto_validate.async_handler.post = AsyncMock(return_value=_mock_allowed_response())

    result = await akto_validate.apply_guardrail(
        inputs=sample_inputs,
        request_data=sample_request_data,
        input_type="request",
    )

    assert result == sample_inputs
    akto_validate.async_handler.post.assert_called_once()
    call_params = akto_validate.async_handler.post.call_args.kwargs["params"]
    assert call_params.get("guardrails") == "true"
    assert "ingest_data" not in call_params


# ---------------------------------------------------------------------------
#  Pre-call (akto-validate) — blocked
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pre_call_blocked(akto_validate, sample_inputs, sample_request_data):
    akto_validate.async_handler.post = AsyncMock(
        side_effect=[
            _mock_blocked_response("PII detected"),
            _mock_allowed_response(),
        ]
    )

    with pytest.raises(HTTPException) as exc_info:
        await akto_validate.apply_guardrail(
            inputs=sample_inputs,
            request_data=sample_request_data,
            input_type="request",
        )

    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert exc_info.value.status_code == 403

    assert akto_validate.async_handler.post.call_count == 2

    first_call_params = akto_validate.async_handler.post.call_args_list[0].kwargs[
        "params"
    ]
    assert first_call_params.get("guardrails") == "true"

    second_call_params = akto_validate.async_handler.post.call_args_list[1].kwargs[
        "params"
    ]
    assert second_call_params.get("ingest_data") == "true"
    assert "guardrails" not in second_call_params
    second_payload = json.loads(
        akto_validate.async_handler.post.call_args_list[1].kwargs["data"]
    )
    assert second_payload["statusCode"] == "403"
    resp_body = json.loads(second_payload["responsePayload"])
    inner = json.loads(resp_body["body"])
    assert inner["x-blocked-by"] == "Akto Proxy"
    assert inner["reason"] == "PII detected"


# ---------------------------------------------------------------------------
#  Pre-call (akto-validate) — response input is no-op
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_validate_response_noop(
    akto_validate, sample_inputs, sample_request_data
):
    akto_validate.async_handler.post = AsyncMock()

    result = await akto_validate.apply_guardrail(
        inputs=sample_inputs,
        request_data=sample_request_data,
        input_type="response",
    )

    assert result == sample_inputs
    akto_validate.async_handler.post.assert_not_called()


# ---------------------------------------------------------------------------
#  Post-call (akto-ingest) — combined guardrail + ingest
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_post_call_combined(akto_ingest, sample_inputs, sample_request_data):
    akto_ingest.async_handler.post = AsyncMock(return_value=_mock_allowed_response())

    result = await akto_ingest.apply_guardrail(
        inputs=sample_inputs,
        request_data=sample_request_data,
        input_type="response",
    )

    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert result == sample_inputs
    akto_ingest.async_handler.post.assert_called_once()
    call_params = akto_ingest.async_handler.post.call_args.kwargs["params"]
    assert call_params.get("guardrails") == "true"
    assert call_params.get("ingest_data") == "true"


# ---------------------------------------------------------------------------
#  Post-call (akto-ingest) — request input is no-op
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_request_noop(akto_ingest, sample_inputs, sample_request_data):
    akto_ingest.async_handler.post = AsyncMock()

    result = await akto_ingest.apply_guardrail(
        inputs=sample_inputs,
        request_data=sample_request_data,
        input_type="request",
    )

    assert result == sample_inputs
    akto_ingest.async_handler.post.assert_not_called()


# ---------------------------------------------------------------------------
#  Fail-open / fail-closed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fail_open_on_unreachable():
    g = AktoGuardrail(
        akto_base_url="http://localhost:9090",
        akto_api_key="test-token",
        unreachable_fallback="fail_open",
        guardrail_name="fail-open-test",
        event_hook="pre_call",
    )
    g.async_handler.post = AsyncMock(
        side_effect=httpx.ConnectError("Connection refused")
    )

    inputs = GenericGuardrailAPIInputs(texts=["test"], model="gpt-4")
    result = await g.apply_guardrail(
        inputs=inputs, request_data={}, input_type="request"
    )

    assert result.get("texts") == ["test"]


@pytest.mark.asyncio
async def test_fail_closed_on_unreachable():
    g = AktoGuardrail(
        akto_base_url="http://localhost:9090",
        akto_api_key="test-token",
        unreachable_fallback="fail_closed",
        guardrail_name="fail-closed-test",
        event_hook="pre_call",
    )
    g.async_handler.post = AsyncMock(
        side_effect=httpx.ConnectError("Connection refused")
    )

    inputs = GenericGuardrailAPIInputs(texts=["test"], model="gpt-4")
    with pytest.raises(HTTPException) as exc_info:
        await g.apply_guardrail(inputs=inputs, request_data={}, input_type="request")
    assert exc_info.value.status_code == 503


def test_fail_closed_generic_message():
    g = AktoGuardrail(
        akto_base_url="http://localhost:9090",
        akto_api_key="test-token",
        unreachable_fallback="fail_closed",
        guardrail_name="msg-test",
        event_hook="pre_call",
    )
    with pytest.raises(HTTPException) as exc_info:
        g.handle_unreachable(
            inputs=GenericGuardrailAPIInputs(texts=["test"], model="gpt-4"),
            error=Exception("http://internal-host:9090/secret-path"),
        )
    assert "internal-host" not in exc_info.value.detail
    assert exc_info.value.detail == "Akto guardrail service unreachable"


# ---------------------------------------------------------------------------
#  Helper method tests
# ---------------------------------------------------------------------------


def test_extract_request_path_from_metadata():
    path = AktoGuardrail.extract_request_path(
        {"metadata": {"user_api_key_request_route": "/v1/embeddings"}}
    )
    assert path == "/v1/embeddings"


def test_extract_request_path_fallback():
    path = AktoGuardrail.extract_request_path({})
    assert path == "/v1/chat/completions"


def test_extract_request_path_non_dict_metadata():
    path = AktoGuardrail.extract_request_path({"metadata": "invalid"})
    assert path == "/v1/chat/completions"


def test_resolve_metadata_value():
    assert (
        AktoGuardrail.resolve_metadata_value(
            {"metadata": {"user_api_key_user_id": "u1"}}, "user_api_key_user_id"
        )
        == "u1"
    )
    assert (
        AktoGuardrail.resolve_metadata_value(
            {"litellm_metadata": {"user_api_key_team_id": "t1"}},
            "user_api_key_team_id",
        )
        == "t1"
    )
    assert AktoGuardrail.resolve_metadata_value({}, "some_key") is None
    assert AktoGuardrail.resolve_metadata_value(None, "some_key") is None


def test_resolve_metadata_value_non_dict_containers():
    assert (
        AktoGuardrail.resolve_metadata_value(
            {"metadata": "invalid", "litellm_metadata": ["bad"]},
            "some_key",
        )
        is None
    )


def test_build_tag_metadata(akto_validate, sample_request_data):
    tag = akto_validate.build_tag_metadata(sample_request_data)
    assert tag["gen-ai"] == "Gen AI"
    assert tag["user_id"] == "user-1"
    assert tag["team_id"] == "team-1"
