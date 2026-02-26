"""
Unit tests for the Akto Guardrail integration.

Tests cover both sync (blocking) and async (non-blocking) modes:
  SYNC:  pre-call guardrails check + post-call ingest
  ASYNC: single post-call with guardrails+ingest
"""

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from litellm.exceptions import GuardrailRaisedException
from litellm.types.utils import GenericGuardrailAPIInputs


# ---------------------------------------------------------------------------
#  Registry tests
# ---------------------------------------------------------------------------


def test_akto_in_guardrail_initializer_registry():
    from litellm.proxy.guardrails.guardrail_registry import (
        guardrail_initializer_registry,
    )

    assert "akto" in guardrail_initializer_registry


def test_akto_in_guardrail_class_registry():
    from litellm.proxy.guardrails.guardrail_hooks.akto.akto import AktoGuardrail
    from litellm.proxy.guardrails.guardrail_registry import guardrail_class_registry

    assert "akto" in guardrail_class_registry
    assert guardrail_class_registry["akto"] is AktoGuardrail


# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def akto_sync():
    """AktoGuardrail in sync (blocking) mode."""
    from litellm.proxy.guardrails.guardrail_hooks.akto.akto import AktoGuardrail

    return AktoGuardrail(
        api_base="http://localhost:9090",
        sync_mode=True,
        akto_account_id="1000000",
        akto_vxlan_id="test-vxlan-id",
        unreachable_fallback="fail_closed",
        guardrail_name="test-akto-sync",
        event_hook="pre_call",
    )


@pytest.fixture
def akto_async():
    """AktoGuardrail in async (non-blocking) mode."""
    from litellm.proxy.guardrails.guardrail_hooks.akto.akto import AktoGuardrail

    return AktoGuardrail(
        api_base="http://localhost:9090",
        sync_mode=False,
        unreachable_fallback="fail_open",
        guardrail_name="test-akto-async",
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
        }
    }


def _mock_allowed_response():
    mock = MagicMock(spec=httpx.Response)
    mock.status_code = 200
    mock.raise_for_status = MagicMock()
    mock.json.return_value = {
        "data": {"guardrailsResult": {"Allowed": True, "Reason": ""}}
    }
    return mock


def _mock_blocked_response(reason="Prompt injection detected"):
    mock = MagicMock(spec=httpx.Response)
    mock.status_code = 200
    mock.raise_for_status = MagicMock()
    mock.json.return_value = {
        "data": {"guardrailsResult": {"Allowed": False, "Reason": reason}}
    }
    return mock


# ---------------------------------------------------------------------------
#  Initialization tests
# ---------------------------------------------------------------------------


def test_init_requires_api_base():
    from litellm.proxy.guardrails.guardrail_hooks.akto.akto import AktoGuardrail

    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="api_base is required"):
            AktoGuardrail(
                api_base="",
                guardrail_name="test",
                event_hook="pre_call",
            )


def test_init_from_env():
    from litellm.proxy.guardrails.guardrail_hooks.akto.akto import AktoGuardrail

    with patch.dict(
        os.environ,
        {
            "AKTO_DATA_INGESTION_URL": "http://env-host:9090",
            "AKTO_SYNC_MODE": "false",
            "AKTO_ACCOUNT_ID": "2000000",
            "AKTO_VXLAN_ID": "env-vxlan",
        },
    ):
        g = AktoGuardrail(guardrail_name="env-test", event_hook="post_call")
        assert g.api_base == "http://env-host:9090"
        assert g.sync_mode is False
        assert g.akto_account_id == "2000000"
        assert g.akto_vxlan_id == "env-vxlan"


def test_sync_mode_default_true():
    from litellm.proxy.guardrails.guardrail_hooks.akto.akto import AktoGuardrail

    g = AktoGuardrail(
        api_base="http://localhost:9090",
        guardrail_name="default-test",
        event_hook="pre_call",
    )
    assert g.sync_mode is True


# ---------------------------------------------------------------------------
#  Payload format tests
# ---------------------------------------------------------------------------


def test_build_akto_payload_format(akto_sync, sample_inputs, sample_request_data):
    """Verify payload matches the exact Akto data-ingestion schema."""
    payload = akto_sync._build_akto_payload(
        sample_inputs, sample_request_data, include_response=False
    )

    assert payload["path"] == "/v1/chat/completions"
    assert payload["method"] == "POST"
    assert payload["akto_account_id"] == "1000000"
    assert payload["akto_vxlan_id"] == "test-vxlan-id"
    assert payload["is_pending"] == "false"
    assert payload["source"] == "MIRRORING"
    assert payload["contextSource"] == "AGENTIC"
    assert payload["ip"] == "user-1"

    req_headers = json.loads(payload["requestHeaders"])
    assert "content-type" in req_headers

    req_body = json.loads(payload["requestPayload"])
    assert req_body["model"] == "gpt-4"
    assert req_body["messages"][0]["content"] == "Hello, how are you?"

    tag = json.loads(payload["tag"])
    assert tag["gen-ai"] == "Gen AI"

    assert payload["responsePayload"] == ""
    assert payload["time"].isdigit()
    assert len(payload["time"]) >= 13


def test_build_akto_payload_with_response(
    akto_sync, sample_inputs, sample_request_data
):
    payload = akto_sync._build_akto_payload(
        sample_inputs, sample_request_data, include_response=True
    )
    resp_body = json.loads(payload["responsePayload"])
    assert "choices" in resp_body


def test_build_query_params():
    from litellm.proxy.guardrails.guardrail_hooks.akto.akto import AktoGuardrail

    params = AktoGuardrail._build_query_params(guardrails=True, ingest_data=False)
    assert params == {"akto_connector": "litellm", "guardrails": "true"}

    params = AktoGuardrail._build_query_params(guardrails=False, ingest_data=True)
    assert params == {"akto_connector": "litellm", "ingest_data": "true"}

    params = AktoGuardrail._build_query_params(guardrails=True, ingest_data=True)
    assert params == {
        "akto_connector": "litellm",
        "guardrails": "true",
        "ingest_data": "true",
    }


# ---------------------------------------------------------------------------
#  Guardrail result parsing
# ---------------------------------------------------------------------------


def test_parse_guardrails_result_allowed():
    from litellm.proxy.guardrails.guardrail_hooks.akto.akto import AktoGuardrail

    allowed, reason = AktoGuardrail._parse_guardrails_result(
        {"data": {"guardrailsResult": {"Allowed": True, "Reason": ""}}}
    )
    assert allowed is True
    assert reason == ""


def test_parse_guardrails_result_blocked():
    from litellm.proxy.guardrails.guardrail_hooks.akto.akto import AktoGuardrail

    allowed, reason = AktoGuardrail._parse_guardrails_result(
        {"data": {"guardrailsResult": {"Allowed": False, "Reason": "PII detected"}}}
    )
    assert allowed is False
    assert reason == "PII detected"


def test_parse_guardrails_result_missing():
    from litellm.proxy.guardrails.guardrail_hooks.akto.akto import AktoGuardrail

    allowed, _ = AktoGuardrail._parse_guardrails_result({})
    assert allowed is True

    allowed, _ = AktoGuardrail._parse_guardrails_result("invalid")
    assert allowed is True


# ---------------------------------------------------------------------------
#  SYNC MODE — pre-call allowed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_pre_call_allowed(akto_sync, sample_inputs, sample_request_data):
    """Sync pre-call: Allowed=true → request passes through, 1 API call made."""
    akto_sync.async_handler.post = AsyncMock(return_value=_mock_allowed_response())

    result = await akto_sync.apply_guardrail(
        inputs=sample_inputs,
        request_data=sample_request_data,
        input_type="request",
    )

    assert result == sample_inputs
    # Only 1 call: guardrails=true
    akto_sync.async_handler.post.assert_called_once()
    call_params = akto_sync.async_handler.post.call_args.kwargs["params"]
    assert call_params.get("guardrails") == "true"
    assert "ingest_data" not in call_params


# ---------------------------------------------------------------------------
#  SYNC MODE — pre-call blocked → ingest blocked + raise error
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_pre_call_blocked(akto_sync, sample_inputs, sample_request_data):
    """Sync pre-call: Allowed=false → ingests blocked request + raises error. 2 API calls."""
    akto_sync.async_handler.post = AsyncMock(
        side_effect=[
            _mock_blocked_response("PII detected"),  # 1st call: guardrails
            _mock_allowed_response(),  # 2nd call: ingest blocked
        ]
    )

    with pytest.raises(GuardrailRaisedException):
        await akto_sync.apply_guardrail(
            inputs=sample_inputs,
            request_data=sample_request_data,
            input_type="request",
        )

    # 2 calls: guardrails check + ingest blocked request
    assert akto_sync.async_handler.post.call_count == 2

    # 1st call: guardrails=true
    first_call_params = akto_sync.async_handler.post.call_args_list[0].kwargs["params"]
    assert first_call_params.get("guardrails") == "true"

    # 2nd call: ingest_data=true (blocked details)
    second_call_params = akto_sync.async_handler.post.call_args_list[1].kwargs["params"]
    assert second_call_params.get("ingest_data") == "true"
    second_payload = akto_sync.async_handler.post.call_args_list[1].kwargs["json"]
    assert second_payload["statusCode"] == "403"


# ---------------------------------------------------------------------------
#  SYNC MODE — post-call ingestion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_post_call_ingest(akto_sync, sample_inputs, sample_request_data):
    """Sync post-call: ingests request+response. 1 API call with ingest_data=true."""
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    akto_sync.async_handler.post = AsyncMock(return_value=mock_resp)

    result = await akto_sync.apply_guardrail(
        inputs=sample_inputs,
        request_data=sample_request_data,
        input_type="response",
    )

    assert result == sample_inputs
    akto_sync.async_handler.post.assert_called_once()
    call_params = akto_sync.async_handler.post.call_args.kwargs["params"]
    assert call_params.get("ingest_data") == "true"
    assert "guardrails" not in call_params


# ---------------------------------------------------------------------------
#  ASYNC MODE — single post-call with guardrails+ingest
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_post_call(akto_async, sample_inputs, sample_request_data):
    """Async post-call: single call with guardrails=true&ingest_data=true."""
    akto_async.async_handler.post = AsyncMock(return_value=_mock_allowed_response())

    result = await akto_async.apply_guardrail(
        inputs=sample_inputs,
        request_data=sample_request_data,
        input_type="response",
    )

    assert result == sample_inputs
    akto_async.async_handler.post.assert_called_once()
    call_params = akto_async.async_handler.post.call_args.kwargs["params"]
    assert call_params.get("guardrails") == "true"
    assert call_params.get("ingest_data") == "true"


# ---------------------------------------------------------------------------
#  ASYNC MODE — flagged response (log-only, no block)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_post_call_flagged_no_block(
    akto_async, sample_inputs, sample_request_data
):
    """Async post-call: Allowed=false → logs but does NOT block."""
    akto_async.async_handler.post = AsyncMock(
        return_value=_mock_blocked_response("suspicious content")
    )

    # Should NOT raise
    result = await akto_async.apply_guardrail(
        inputs=sample_inputs,
        request_data=sample_request_data,
        input_type="response",
    )

    assert result == sample_inputs


# ---------------------------------------------------------------------------
#  ASYNC MODE — pre-call is no-op
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_pre_call_noop(akto_async, sample_inputs, sample_request_data):
    """Async mode: pre-call returns inputs immediately without API call."""
    akto_async.async_handler.post = AsyncMock()

    result = await akto_async.apply_guardrail(
        inputs=sample_inputs,
        request_data=sample_request_data,
        input_type="request",
    )

    assert result == sample_inputs
    akto_async.async_handler.post.assert_not_called()


# ---------------------------------------------------------------------------
#  Fail-open / fail-closed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fail_open_on_unreachable():
    from litellm.proxy.guardrails.guardrail_hooks.akto.akto import AktoGuardrail

    g = AktoGuardrail(
        api_base="http://localhost:9090",
        sync_mode=True,
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

    assert result["texts"] == ["test"]


@pytest.mark.asyncio
async def test_fail_closed_on_unreachable():
    from litellm.proxy.guardrails.guardrail_hooks.akto.akto import AktoGuardrail

    g = AktoGuardrail(
        api_base="http://localhost:9090",
        sync_mode=True,
        unreachable_fallback="fail_closed",
        guardrail_name="fail-closed-test",
        event_hook="pre_call",
    )
    g.async_handler.post = AsyncMock(
        side_effect=httpx.ConnectError("Connection refused")
    )

    inputs = GenericGuardrailAPIInputs(texts=["test"], model="gpt-4")
    with pytest.raises(Exception, match="Akto guardrail failed"):
        await g.apply_guardrail(
            inputs=inputs, request_data={}, input_type="request"
        )


# ---------------------------------------------------------------------------
#  Helper method tests
# ---------------------------------------------------------------------------


def test_extract_request_path_from_metadata():
    from litellm.proxy.guardrails.guardrail_hooks.akto.akto import AktoGuardrail

    path = AktoGuardrail._extract_request_path(
        {"metadata": {"user_api_key_request_route": "/v1/embeddings"}}
    )
    assert path == "/v1/embeddings"


def test_extract_request_path_fallback():
    from litellm.proxy.guardrails.guardrail_hooks.akto.akto import AktoGuardrail

    path = AktoGuardrail._extract_request_path({})
    assert path == "/v1/chat/completions"


def test_resolve_metadata_value():
    from litellm.proxy.guardrails.guardrail_hooks.akto.akto import AktoGuardrail

    assert (
        AktoGuardrail._resolve_metadata_value(
            {"metadata": {"user_api_key_user_id": "u1"}}, "user_api_key_user_id"
        )
        == "u1"
    )
    assert (
        AktoGuardrail._resolve_metadata_value(
            {"litellm_metadata": {"user_api_key_team_id": "t1"}},
            "user_api_key_team_id",
        )
        == "t1"
    )
    assert AktoGuardrail._resolve_metadata_value({}, "some_key") is None
    assert AktoGuardrail._resolve_metadata_value(None, "some_key") is None


def test_build_tag_metadata(akto_sync, sample_request_data):
    tag = akto_sync._build_tag_metadata(sample_request_data)
    assert tag["gen-ai"] == "Gen AI"
    assert tag["user_id"] == "user-1"
    assert tag["team_id"] == "team-1"
