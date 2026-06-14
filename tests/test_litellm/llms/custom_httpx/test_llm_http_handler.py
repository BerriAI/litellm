import asyncio
import os
import sys
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path
import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.custom_httpx.llm_http_handler import (
    BaseLLMHTTPHandler,
    _google_genai_streaming_hidden_params,
)
from litellm.types.router import GenericLiteLLMParams


def test_prepare_fake_stream_request():
    # Initialize the BaseLLMHTTPHandler
    handler = BaseLLMHTTPHandler()

    # Test case 1: fake_stream is True
    stream = True
    data = {
        "stream": True,
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    fake_stream = True

    result_stream, result_data = handler._prepare_fake_stream_request(
        stream=stream, data=data, fake_stream=fake_stream
    )

    # Verify that stream is set to False
    assert result_stream is False
    # Verify that "stream" key is removed from data
    assert "stream" not in result_data
    # Verify other data remains unchanged
    assert result_data["model"] == "gpt-4"
    assert result_data["messages"] == [{"role": "user", "content": "Hello"}]

    # Test case 2: fake_stream is False
    stream = True
    data = {
        "stream": True,
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    fake_stream = False

    result_stream, result_data = handler._prepare_fake_stream_request(
        stream=stream, data=data, fake_stream=fake_stream
    )

    # Verify that stream remains True
    assert result_stream is True
    # Verify that data remains unchanged
    assert "stream" in result_data
    assert result_data["stream"] is True
    assert result_data["model"] == "gpt-4"
    assert result_data["messages"] == [{"role": "user", "content": "Hello"}]

    # Test case 3: data doesn't have stream key but fake_stream is True
    stream = True
    data = {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}
    fake_stream = True

    result_stream, result_data = handler._prepare_fake_stream_request(
        stream=stream, data=data, fake_stream=fake_stream
    )

    # Verify that stream is set to False
    assert result_stream is False
    # Verify that data remains unchanged (since there was no stream key to remove)
    assert "stream" not in result_data
    assert result_data["model"] == "gpt-4"
    assert result_data["messages"] == [{"role": "user", "content": "Hello"}]


def test_get_agentic_loop_settings_defaults_and_overrides():
    handler = BaseLLMHTTPHandler()

    depth, max_loops, fingerprints = handler._get_agentic_loop_settings(kwargs={})
    assert depth == 0
    assert max_loops == 3
    assert fingerprints == []

    depth, max_loops, fingerprints = handler._get_agentic_loop_settings(
        kwargs={
            "_agentic_loop_depth": 2,
            "max_agentic_loops": 7,
            "_agentic_loop_fingerprints": ["fp-1", "fp-2"],
        }
    )
    assert depth == 2
    assert max_loops == 7
    assert fingerprints == ["fp-1", "fp-2"]


def test_has_agentic_completion_hook_detection(monkeypatch):
    """The streaming path skips the agentic wrapper only when no callback
    overrides async_should_run_agentic_loop. Verify both directions."""
    from litellm.integrations.custom_logger import CustomLogger

    handler = BaseLLMHTTPHandler()
    logging_obj = Mock()
    logging_obj.dynamic_success_callbacks = []

    # No callbacks at all -> no agentic hook.
    monkeypatch.setattr(litellm, "callbacks", [])
    assert handler._has_agentic_completion_hook(logging_obj) is False

    # A plain CustomLogger that does NOT override the gate -> still no hook
    # (so the wrapper is safely skipped).
    class _PlainLogger(CustomLogger):
        pass

    monkeypatch.setattr(litellm, "callbacks", [_PlainLogger()])
    assert handler._has_agentic_completion_hook(logging_obj) is False

    # A logger that overrides the gate (directly) -> hook present.
    class _AgenticLogger(CustomLogger):
        async def async_should_run_agentic_loop(
            self, response, model, messages, tools, stream, custom_llm_provider, kwargs
        ):
            return True, {}

    monkeypatch.setattr(litellm, "callbacks", [_AgenticLogger()])
    assert handler._has_agentic_completion_hook(logging_obj) is True

    # Override inherited through an intermediate class is still detected
    # (function-identity check, not a leaf __dict__ check).
    class _DerivedAgenticLogger(_AgenticLogger):
        pass

    monkeypatch.setattr(litellm, "callbacks", [_DerivedAgenticLogger()])
    assert handler._has_agentic_completion_hook(logging_obj) is True

    # Hook supplied via logging_obj.dynamic_success_callbacks is detected too.
    monkeypatch.setattr(litellm, "callbacks", [])
    logging_obj.dynamic_success_callbacks = [_AgenticLogger()]
    assert handler._has_agentic_completion_hook(logging_obj) is True

    # String-named callback entry (e.g. "datadog") must be resolved to its
    # CustomLogger instance via get_custom_logger_compatible_class -- the same
    # way ProxyLogging._callback_capabilities handles them. Without that
    # resolution a string-registered agentic callback would be silently
    # skipped and the buffering wrapper would never fire.
    logging_obj.dynamic_success_callbacks = []
    agentic_via_string = _AgenticLogger()
    monkeypatch.setattr(litellm, "callbacks", ["fake_string_callback"])
    monkeypatch.setattr(
        "litellm.litellm_core_utils.litellm_logging.get_custom_logger_compatible_class",
        lambda name: agentic_via_string if name == "fake_string_callback" else None,
    )
    assert handler._has_agentic_completion_hook(logging_obj) is True

    # Unresolvable string (returns None) is skipped, no false positive.
    monkeypatch.setattr(litellm, "callbacks", ["unknown_callback"])
    monkeypatch.setattr(
        "litellm.litellm_core_utils.litellm_logging.get_custom_logger_compatible_class",
        lambda name: None,
    )
    assert handler._has_agentic_completion_hook(logging_obj) is False


def test_fingerprint_agentic_tools_is_deterministic():
    handler = BaseLLMHTTPHandler()
    tools_a = {"tool_calls": [{"id": "1", "input": {"q": "abc"}, "name": "web_search"}]}
    tools_b = {"tool_calls": [{"name": "web_search", "input": {"q": "abc"}, "id": "1"}]}

    assert handler._fingerprint_agentic_tools(
        tools_a
    ) == handler._fingerprint_agentic_tools(tools_b)


@pytest.mark.asyncio
async def test_async_anthropic_messages_handler_extra_headers():
    """
    Test that async_anthropic_messages_handler correctly extracts and merges
    extra_headers from kwargs with proper priority.
    """
    handler = BaseLLMHTTPHandler()

    # Mock the config
    mock_config = Mock()
    mock_config.validate_anthropic_messages_environment = Mock(
        return_value=({"x-api-key": "test-key"}, "https://api.anthropic.com")
    )
    mock_config.transform_anthropic_messages_request = Mock(
        return_value={"model": "claude-3-opus-20240229", "messages": []}
    )

    # Mock the client
    mock_client = AsyncMock()
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello!"}],
        "model": "claude-3-opus-20240229",
        "stop_reason": "end_turn",
    }
    mock_client.post = AsyncMock(return_value=mock_response)

    # Mock logging object
    mock_logging_obj = Mock()
    mock_logging_obj.update_environment_variables = Mock()
    mock_logging_obj.model_call_details = {}
    mock_logging_obj.stream = False

    # Test case 1: Only extra_headers in kwargs
    kwargs = {
        "extra_headers": {
            "X-Custom-Header": "from-kwargs",
            "X-Auth-Token": "token123",
        }
    }

    with patch(
        "litellm.litellm_core_utils.get_provider_specific_headers.ProviderSpecificHeaderUtils.get_provider_specific_headers"
    ) as mock_provider_headers:
        mock_provider_headers.return_value = None

        # Capture what headers are passed to validate_anthropic_messages_environment
        captured_headers = {}

        def capture_validate(*args, **kwargs):
            captured_headers.update(kwargs.get("headers", {}))
            return ({"x-api-key": "test-key"}, "https://api.anthropic.com")

        mock_config.validate_anthropic_messages_environment = capture_validate

        try:
            await handler.async_anthropic_messages_handler(
                model="claude-3-opus-20240229",
                messages=[{"role": "user", "content": "Hello"}],
                anthropic_messages_provider_config=mock_config,
                anthropic_messages_optional_request_params={},
                custom_llm_provider="anthropic",
                litellm_params=GenericLiteLLMParams(),
                logging_obj=mock_logging_obj,
                client=mock_client,
                kwargs=kwargs,
            )
        except Exception:
            pass  # We're testing header extraction, not the full flow

        # Verify extra_headers were extracted and merged
        assert "X-Custom-Header" in captured_headers
        assert captured_headers["X-Custom-Header"] == "from-kwargs"
        assert "X-Auth-Token" in captured_headers
        assert captured_headers["X-Auth-Token"] == "token123"


@pytest.mark.asyncio
async def test_async_anthropic_messages_handler_passes_litellm_metadata():
    """Ensure litellm_metadata from kwargs is forwarded via update_from_kwargs.

    Routes like /messages store model_info under kwargs['litellm_metadata'].
    The handler must forward this so that use_custom_pricing_for_model can
    detect custom pricing. Regression test for #23185.
    """
    handler = BaseLLMHTTPHandler()

    mock_config = Mock()
    mock_config.validate_anthropic_messages_environment = Mock(
        return_value=({"x-api-key": "test-key"}, "https://api.anthropic.com")
    )
    mock_config.transform_anthropic_messages_request = Mock(
        return_value={"model": "claude-sonnet-4-20250514", "messages": []}
    )

    mock_client = AsyncMock()
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello!"}],
        "model": "claude-sonnet-4-20250514",
        "stop_reason": "end_turn",
    }
    mock_client.post = AsyncMock(return_value=mock_response)

    mock_logging_obj = Mock()
    mock_logging_obj.update_from_kwargs = Mock()
    mock_logging_obj.model_call_details = {}
    mock_logging_obj.stream = False

    custom_model_info = {
        "id": "claude-sonnet-4-custom-pricing",
        "input_cost_per_token": 0.0003,
        "output_cost_per_token": 0.0015,
    }
    kwargs = {
        "litellm_metadata": {
            "model_info": custom_model_info,
            "deployment": "anthropic/claude-sonnet-4-20250514",
        },
    }

    try:
        await handler.async_anthropic_messages_handler(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "Hello"}],
            anthropic_messages_provider_config=mock_config,
            anthropic_messages_optional_request_params={},
            custom_llm_provider="anthropic",
            litellm_params=GenericLiteLLMParams(),
            logging_obj=mock_logging_obj,
            client=mock_client,
            kwargs=kwargs,
        )
    except Exception:
        pass

    mock_logging_obj.update_from_kwargs.assert_called_once()
    call_kwargs = mock_logging_obj.update_from_kwargs.call_args
    kwargs_arg = (
        call_kwargs.kwargs.get("kwargs", call_kwargs[1].get("kwargs", {}))
        if call_kwargs.kwargs
        else call_kwargs[1].get("kwargs", {})
    )

    assert "litellm_metadata" in kwargs_arg
    assert kwargs_arg["litellm_metadata"]["model_info"] == custom_model_info


@pytest.mark.asyncio
async def test_async_anthropic_messages_handler_forwards_router_model_info():
    """Ensure router deployment model_info is forwarded into litellm_params.

    The Router stamps kwargs['model_info'] on every deployment dispatch via
    _update_kwargs_with_deployment. Downstream cooldown / success callbacks
    (router.deployment_callback_on_failure, deployment_callback_on_success)
    look up the deployment id via kwargs['litellm_params']['model_info']['id'].
    If async_anthropic_messages_handler builds its own litellm_params dict
    without forwarding model_info, the id is missing and cooldown is silently
    skipped for /v1/messages requests under the Router.
    """
    handler = BaseLLMHTTPHandler()

    mock_config = Mock()
    mock_config.validate_anthropic_messages_environment = Mock(
        return_value=({"x-api-key": "test-key"}, "https://api.anthropic.com")
    )
    mock_config.transform_anthropic_messages_request = Mock(
        return_value={"model": "claude-sonnet-4-20250514", "messages": []}
    )

    mock_client = AsyncMock()
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello!"}],
        "model": "claude-sonnet-4-20250514",
        "stop_reason": "end_turn",
    }
    mock_client.post = AsyncMock(return_value=mock_response)

    mock_logging_obj = Mock()
    mock_logging_obj.update_from_kwargs = Mock()
    mock_logging_obj.model_call_details = {}
    mock_logging_obj.stream = False

    deployment_model_info = {
        "id": "deployment-123",
        "db_model": False,
    }

    try:
        await handler.async_anthropic_messages_handler(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "Hello"}],
            anthropic_messages_provider_config=mock_config,
            anthropic_messages_optional_request_params={},
            custom_llm_provider="anthropic",
            litellm_params=GenericLiteLLMParams(),
            logging_obj=mock_logging_obj,
            client=mock_client,
            kwargs={"model_info": deployment_model_info},
        )
    except Exception:
        pass

    mock_logging_obj.update_from_kwargs.assert_called_once()
    call_kwargs = mock_logging_obj.update_from_kwargs.call_args
    litellm_params_arg = (
        call_kwargs.kwargs.get(
            "litellm_params", call_kwargs[1].get("litellm_params", {})
        )
        if call_kwargs.kwargs
        else call_kwargs[1].get("litellm_params", {})
    )

    assert litellm_params_arg.get("model_info") == deployment_model_info


@pytest.mark.asyncio
async def test_async_anthropic_messages_handler_header_priority():
    """
    Test that async_anthropic_messages_handler respects header priority:
    forwarded < extra_headers < provider_specific
    """
    handler = BaseLLMHTTPHandler()

    # Mock the config
    mock_config = Mock()
    mock_client = AsyncMock()
    mock_logging_obj = Mock()
    mock_logging_obj.update_environment_variables = Mock()
    mock_logging_obj.model_call_details = {}
    mock_logging_obj.stream = False

    # Test with all three header sources
    kwargs = {
        "headers": {"X-Priority": "forwarded", "X-Forwarded-Only": "keep"},
        "extra_headers": {"X-Priority": "extra", "X-Extra-Only": "also-keep"},
    }

    with patch(
        "litellm.litellm_core_utils.get_provider_specific_headers.ProviderSpecificHeaderUtils.get_provider_specific_headers"
    ) as mock_provider_headers:
        mock_provider_headers.return_value = {
            "X-Priority": "provider",
            "X-Provider-Only": "keep-this-too",
        }

        captured_headers = {}

        def capture_validate(*args, **kwargs):
            captured_headers.update(kwargs.get("headers", {}))
            return ({"x-api-key": "test-key"}, "https://api.anthropic.com")

        mock_config.validate_anthropic_messages_environment = capture_validate
        mock_config.transform_anthropic_messages_request = Mock(
            return_value={"model": "claude-3-opus-20240229", "messages": []}
        )

        try:
            await handler.async_anthropic_messages_handler(
                model="claude-3-opus-20240229",
                messages=[{"role": "user", "content": "Hello"}],
                anthropic_messages_provider_config=mock_config,
                anthropic_messages_optional_request_params={},
                custom_llm_provider="anthropic",
                litellm_params=GenericLiteLLMParams(),
                logging_obj=mock_logging_obj,
                client=mock_client,
                kwargs=kwargs,
            )
        except Exception:
            pass

        # Verify priority: provider_specific should win
        assert captured_headers["X-Priority"] == "provider"
        # Verify all unique headers from different sources are present
        assert captured_headers["X-Forwarded-Only"] == "keep"
        assert captured_headers["X-Extra-Only"] == "also-keep"
        assert captured_headers["X-Provider-Only"] == "keep-this-too"


def test_google_genai_streaming_hidden_params_model_info_and_router_fallback():
    logging_obj = Mock()
    logging_obj.get_router_model_id = Mock(return_value="router-model-id")

    from_model_info = _google_genai_streaming_hidden_params(
        api_base="https://generativelanguage.googleapis.com/v1beta",
        litellm_params=GenericLiteLLMParams(model_info={"id": "info-id"}),
        logging_obj=logging_obj,
        response_headers=httpx.Headers({"x-ratelimit-remaining": "10"}),
    )
    assert from_model_info["model_id"] == "info-id"
    assert (
        from_model_info["api_base"]
        == "https://generativelanguage.googleapis.com/v1beta"
    )
    assert isinstance(from_model_info["additional_headers"], dict)

    from_router = _google_genai_streaming_hidden_params(
        api_base="https://x",
        litellm_params=GenericLiteLLMParams(),
        logging_obj=logging_obj,
        response_headers=httpx.Headers({}),
    )
    assert from_router["model_id"] == "router-model-id"


def _build_delete_response_mock(captured: dict):
    """Returns a fake httpx delete that records its kwargs."""

    def _response() -> httpx.Response:
        return httpx.Response(
            status_code=200,
            headers={"content-type": "application/json"},
            content=b'{"id": "resp_x", "object": "response", "deleted": true}',
            request=httpx.Request(method="DELETE", url="https://test.openai.azure.com"),
        )

    async def fake_async_delete(*args, **kwargs):
        captured.update(kwargs)
        return _response()

    def fake_sync_delete(*args, **kwargs):
        captured.update(kwargs)
        return _response()

    return fake_async_delete, fake_sync_delete


def test_async_delete_responses_omits_body_for_azure():
    """Azure responses DELETE rejects requests with any body. Verify the handler
    does not pass `json=` to httpx when the transformer returns an empty dict."""
    captured: dict = {}
    fake_async_delete, _ = _build_delete_response_mock(captured)

    async def run():
        with patch.object(AsyncHTTPHandler, "delete", new=fake_async_delete):
            await litellm.adelete_responses(
                response_id="resp_xyz",
                custom_llm_provider="azure",
                api_base="https://test.openai.azure.com",
                api_key="test-key",
                api_version="2025-03-01-preview",
            )

    asyncio.run(run())

    assert "json" not in captured
    assert "data" not in captured
    assert captured["url"].endswith(
        "/openai/responses/resp_xyz?api-version=2025-03-01-preview"
    )


def test_sync_delete_responses_omits_body_for_azure():
    captured: dict = {}
    _, fake_sync_delete = _build_delete_response_mock(captured)

    with patch.object(HTTPHandler, "delete", new=fake_sync_delete):
        litellm.delete_responses(
            response_id="resp_xyz",
            custom_llm_provider="azure",
            api_base="https://test.openai.azure.com",
            api_key="test-key",
            api_version="2025-03-01-preview",
        )

    assert "json" not in captured
    assert "data" not in captured
    assert captured["url"].endswith(
        "/openai/responses/resp_xyz?api-version=2025-03-01-preview"
    )


def _content_type(headers: dict) -> str:
    for key, value in headers.items():
        if key.lower() == "content-type":
            return value
    return ""


def test_async_delete_responses_sets_json_content_type():
    """OpenAI rejects a responses DELETE with no Content-Type by treating it as
    application/octet-stream. The handler must declare application/json."""
    captured: dict = {}
    fake_async_delete, _ = _build_delete_response_mock(captured)

    async def run():
        with patch.object(AsyncHTTPHandler, "delete", new=fake_async_delete):
            await litellm.adelete_responses(
                response_id="resp_xyz",
                custom_llm_provider="openai",
                api_key="test-key",
            )

    asyncio.run(run())

    assert _content_type(captured["headers"]) == "application/json"


def test_sync_delete_responses_sets_json_content_type():
    captured: dict = {}
    _, fake_sync_delete = _build_delete_response_mock(captured)

    with patch.object(HTTPHandler, "delete", new=fake_sync_delete):
        litellm.delete_responses(
            response_id="resp_xyz",
            custom_llm_provider="openai",
            api_key="test-key",
        )

    assert _content_type(captured["headers"]) == "application/json"


# ---------------------------------------------------------------------------
# Parity tests: request-body is serialized once and reused for the wire.
# (_async_post_anthropic_messages_with_http_error_retry)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_anthropic_post_uses_prebuilt_body_without_redumping():
    """When the caller passes a pre-serialized (unsigned) body, attempt 0 must
    send exactly those bytes -- no second json.dumps of request_body."""
    import json as _json

    handler = BaseLLMHTTPHandler()
    request_body = {"model": "claude", "messages": [{"role": "user", "content": "hi"}]}
    prebuilt = _json.dumps(request_body)

    ok_resp = Mock()
    ok_resp.raise_for_status = Mock(return_value=None)
    http_client = Mock()
    http_client.post = AsyncMock(return_value=ok_resp)

    provider_config = Mock()
    provider_config.max_retry_on_anthropic_messages_http_error = 2

    logging_obj = Mock()
    logging_obj.model_call_details = {}

    out = await handler._async_post_anthropic_messages_with_http_error_retry(
        async_httpx_client=http_client,
        request_url="http://x/v1/messages",
        headers={},
        signed_json_body=prebuilt,
        request_body=request_body,
        stream=False,
        logging_obj=logging_obj,
        provider_config=provider_config,
        litellm_params=GenericLiteLLMParams(),
        api_key="k",
        model="claude",
    )
    assert out is ok_resp
    http_client.post.assert_awaited_once()
    sent = http_client.post.await_args.kwargs["data"]
    # Byte-identical to the legacy wire serialization, and the SAME object the
    # caller already used for the pre-call log (no re-serialization).
    assert sent == prebuilt
    assert sent is prebuilt


@pytest.mark.asyncio
async def test_anthropic_post_falls_back_to_json_dumps_when_unsigned_none():
    """signed_json_body=None keeps the exact legacy behavior."""
    import json as _json

    handler = BaseLLMHTTPHandler()
    request_body = {"model": "claude", "messages": [{"role": "user", "content": "yo"}]}

    ok_resp = Mock()
    ok_resp.raise_for_status = Mock(return_value=None)
    http_client = Mock()
    http_client.post = AsyncMock(return_value=ok_resp)

    provider_config = Mock()
    provider_config.max_retry_on_anthropic_messages_http_error = 1
    logging_obj = Mock()
    logging_obj.model_call_details = {}

    await handler._async_post_anthropic_messages_with_http_error_retry(
        async_httpx_client=http_client,
        request_url="http://x/v1/messages",
        headers={},
        signed_json_body=None,
        request_body=request_body,
        stream=False,
        logging_obj=logging_obj,
        provider_config=provider_config,
        litellm_params=GenericLiteLLMParams(),
        api_key="k",
        model="claude",
    )
    sent = http_client.post.await_args.kwargs["data"]
    assert sent == _json.dumps(request_body)


@pytest.mark.asyncio
async def test_anthropic_post_retry_reserializes_mutated_body():
    """On a retryable HTTP error the body is mutated + re-signed; the prebuilt
    body must NOT be reused -- attempt 1 sends the freshly serialized body."""
    import json as _json

    handler = BaseLLMHTTPHandler()
    request_body = {"model": "claude", "messages": [{"role": "user", "content": "a"}]}
    prebuilt = _json.dumps(request_body)

    err_resp = Mock()
    http_error = httpx.HTTPStatusError(
        "bad", request=Mock(), response=Mock(status_code=400)
    )
    err_resp.raise_for_status = Mock(side_effect=http_error)
    ok_resp = Mock()
    ok_resp.raise_for_status = Mock(return_value=None)
    http_client = Mock()
    http_client.post = AsyncMock(side_effect=[err_resp, ok_resp])

    def _mutate(e, request_data):
        request_data["messages"][0]["content"] = "MUTATED"

    provider_config = Mock()
    provider_config.max_retry_on_anthropic_messages_http_error = 2
    provider_config.should_retry_anthropic_messages_on_http_error = Mock(
        return_value=True
    )
    provider_config.transform_anthropic_messages_request_on_http_error = _mutate
    # Re-sign returns no signed body (native anthropic path) -> must re-dump.
    provider_config.sign_request = Mock(return_value=({}, None))

    logging_obj = Mock()
    logging_obj.model_call_details = {}

    await handler._async_post_anthropic_messages_with_http_error_retry(
        async_httpx_client=http_client,
        request_url="http://x/v1/messages",
        headers={},
        signed_json_body=prebuilt,
        request_body=request_body,
        stream=False,
        logging_obj=logging_obj,
        provider_config=provider_config,
        litellm_params=GenericLiteLLMParams(),
        api_key="k",
        model="claude",
    )
    assert http_client.post.await_count == 2
    first_sent = http_client.post.await_args_list[0].kwargs["data"]
    second_sent = http_client.post.await_args_list[1].kwargs["data"]
    assert first_sent == prebuilt  # attempt 0 used prebuilt
    assert second_sent == _json.dumps(request_body)  # attempt 1 re-serialized
    assert "MUTATED" in second_sent  # ... the mutated body
