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
from litellm.integrations.code_interpreter_interception.handler import (
    CodeInterpreterInterceptionLogger,
    LITELLM_CODE_EXECUTION_TOOL_NAME,
)
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.custom_httpx.llm_http_handler import (
    BaseLLMHTTPHandler,
    _google_genai_streaming_hidden_params,
)
from litellm.types.llms.openai import ResponsesAPIResponse
from litellm.types.router import GenericLiteLLMParams

_ACTIVE_KEY = "_code_interpreter_interception_active"
_SANDBOX_KEY = "_code_interpreter_interception_sandbox_key"


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


def test_response_api_handler_streams_when_provider_transform_adds_stream():
    handler = BaseLLMHTTPHandler()
    config = Mock()
    config.validate_environment.return_value = {}
    config.get_complete_url.return_value = "https://chatgpt.example.com/responses"
    config.transform_responses_api_request.return_value = {
        "model": "gpt-5.3-codex",
        "input": "hi",
        "stream": True,
    }
    config.sign_request.return_value = ({}, None)
    client = HTTPHandler(client=httpx.Client())
    client.post = Mock(
        return_value=httpx.Response(
            200,
            request=httpx.Request("POST", "https://chatgpt.example.com/responses"),
        )
    )
    logging_obj = Mock()

    handler.response_api_handler(
        model="gpt-5.3-codex",
        input="hi",
        responses_api_provider_config=config,
        response_api_optional_request_params={},
        custom_llm_provider="chatgpt",
        litellm_params=GenericLiteLLMParams(),
        logging_obj=logging_obj,
        client=client,
    )

    assert client.post.call_args.kwargs["stream"] is True
    assert client.post.call_args.kwargs["json"]["stream"] is True


def test_response_api_handler_runs_agentic_hooks_in_sync_path(monkeypatch):
    handler = BaseLLMHTTPHandler()
    config = Mock()
    config.validate_environment.return_value = {}
    config.get_complete_url.return_value = "https://chatgpt.example.com/responses"
    config.transform_responses_api_request.return_value = {
        "model": "gpt-5",
        "input": "hi",
    }
    config.sign_request.return_value = ({}, None)
    initial_response = Mock()
    final_response = Mock()
    config.transform_response_api_response.return_value = initial_response

    client = HTTPHandler(client=httpx.Client())
    client.post = Mock(
        return_value=httpx.Response(
            200,
            request=httpx.Request("POST", "https://chatgpt.example.com/responses"),
        )
    )
    logging_obj = Mock()

    monkeypatch.setattr(handler, "_has_agentic_completion_hook", Mock(return_value=True))
    hook_mock = AsyncMock(return_value=final_response)
    monkeypatch.setattr(handler, "_call_agentic_completion_hooks", hook_mock)

    response = handler.response_api_handler(
        model="gpt-5",
        input="hi",
        responses_api_provider_config=config,
        response_api_optional_request_params={},
        custom_llm_provider="openai",
        litellm_params=GenericLiteLLMParams(),
        logging_obj=logging_obj,
        client=client,
    )

    assert response is final_response
    hook_mock.assert_awaited_once()
    assert hook_mock.call_args.kwargs["api_surface"] == "responses"
    assert hook_mock.call_args.kwargs["messages"] == [
        {"role": "user", "content": "hi"}
    ]


def test_response_api_handler_runs_responses_pre_call_hook_before_transform():
    handler = BaseLLMHTTPHandler()
    config = Mock()
    config.validate_environment.return_value = {}
    config.get_complete_url.return_value = "https://api.openai.com/v1/responses"
    config.sign_request.return_value = ({}, None)
    initial_response = ResponsesAPIResponse(
        id="resp_1",
        created_at=0,
        output=[],
        status="completed",
        model="gpt-5",
    )
    config.transform_response_api_response.return_value = initial_response

    def transform_responses_api_request(**kwargs):
        return {
            "model": kwargs["model"],
            "input": kwargs["input"],
            **kwargs["response_api_optional_request_params"],
        }

    config.transform_responses_api_request.side_effect = transform_responses_api_request
    client = HTTPHandler(client=httpx.Client())
    client.post = Mock(
        return_value=httpx.Response(
            200,
            request=httpx.Request("POST", "https://api.openai.com/v1/responses"),
        )
    )
    logging_obj = Mock()
    logging_obj.dynamic_success_callbacks = []

    old_callbacks = list(litellm.callbacks)
    litellm.callbacks = [CodeInterpreterInterceptionLogger()]
    try:
        response = handler.response_api_handler(
            model="gpt-5",
            input="use code",
            responses_api_provider_config=config,
            response_api_optional_request_params={
                "tools": [{"type": "code_interpreter", "container": {"type": "auto"}}]
            },
            custom_llm_provider="openai",
            litellm_params=GenericLiteLLMParams(api_key="sk-test"),
            logging_obj=logging_obj,
            client=client,
        )
    finally:
        litellm.callbacks = old_callbacks

    assert response is initial_response
    transform_kwargs = config.transform_responses_api_request.call_args.kwargs
    tools = transform_kwargs["response_api_optional_request_params"]["tools"]
    assert not any(tool.get("type") == "code_interpreter" for tool in tools)
    assert any(
        tool.get("type") == "function"
        and tool.get("name") == LITELLM_CODE_EXECUTION_TOOL_NAME
        for tool in tools
    )
    hook_litellm_params = transform_kwargs["litellm_params"]
    assert hook_litellm_params.get(_ACTIVE_KEY) is True
    assert hook_litellm_params.get(_SANDBOX_KEY)


@pytest.mark.asyncio
async def test_async_response_api_handler_streams_when_provider_transform_adds_stream():
    handler = BaseLLMHTTPHandler()
    config = Mock()
    config.validate_environment.return_value = {}
    config.get_complete_url.return_value = "https://chatgpt.example.com/responses"
    config.transform_responses_api_request.return_value = {
        "model": "gpt-5.3-codex",
        "input": "hi",
        "stream": True,
    }
    config.sign_request.return_value = ({}, None)
    client = AsyncHTTPHandler()
    client.post = AsyncMock(
        return_value=httpx.Response(
            200,
            request=httpx.Request("POST", "https://chatgpt.example.com/responses"),
        )
    )
    logging_obj = Mock()

    await handler.async_response_api_handler(
        model="gpt-5.3-codex",
        input="hi",
        responses_api_provider_config=config,
        response_api_optional_request_params={},
        custom_llm_provider="chatgpt",
        litellm_params=GenericLiteLLMParams(),
        logging_obj=logging_obj,
        client=client,
    )

    assert client.post.call_args.kwargs["stream"] is True
    assert client.post.call_args.kwargs["json"]["stream"] is True


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


def test_base_responses_config_sign_request_is_noop_by_default():
    """Default responses sign_request must be a no-op: unchanged headers, no signed body.

    Guards the 15 existing responses providers from accidental signing when the
    handler starts calling sign_request.
    """
    from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig

    cfg = OpenAIResponsesAPIConfig()
    headers = {"Authorization": "Bearer sk-existing"}
    out_headers, signed_body = cfg.sign_request(
        headers=headers,
        optional_params={},
        request_data={"input": "hi"},
        api_base="https://api.openai.com/v1/responses",
    )
    assert out_headers == {"Authorization": "Bearer sk-existing"}
    assert signed_body is None


def _make_responses_handler_call(signed_body):
    """Drive BaseLLMHTTPHandler.response_api_handler with a fully mocked provider
    config + sync client, returning the kwargs the client.post was called with.

    signed_body=None simulates a no-op (non-signing) provider; bytes simulates a
    signing provider (e.g. Bedrock Mantle).
    """
    from unittest.mock import MagicMock
    from litellm.llms.custom_httpx.http_handler import HTTPHandler
    from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
    from litellm.types.router import GenericLiteLLMParams

    provider_config = MagicMock()
    provider_config.validate_environment.return_value = {}
    provider_config.get_complete_url.return_value = (
        "https://bedrock-mantle.us-east-2.api.aws/openai/v1/responses"
    )
    provider_config.transform_responses_api_request.return_value = {"input": "hi"}
    provider_config.should_fake_stream.return_value = False
    provider_config.sign_request.return_value = ({"X-Signed": "1"}, signed_body)

    mock_client = MagicMock(spec=HTTPHandler)
    mock_client.post.return_value = MagicMock()

    handler = BaseLLMHTTPHandler()
    handler.response_api_handler(
        model="openai.gpt-5.5",
        input="hi",
        responses_api_provider_config=provider_config,
        response_api_optional_request_params={},
        custom_llm_provider="bedrock_mantle",
        litellm_params=GenericLiteLLMParams(aws_region_name="us-east-2"),
        logging_obj=MagicMock(),
        client=mock_client,
        _is_async=False,
    )
    return mock_client.post.call_args.kwargs


def test_responses_handler_sends_json_when_not_signed():
    """No-op provider (signed_body is None) -> handler posts json=data, no data= bytes."""
    kwargs = _make_responses_handler_call(signed_body=None)
    assert kwargs.get("json") == {"input": "hi"}
    assert "data" not in kwargs


def test_responses_handler_sends_signed_bytes_when_signed():
    """Signing provider -> handler posts the exact signed bytes via data=, not json=."""
    kwargs = _make_responses_handler_call(signed_body=b'{"input": "hi"}')
    assert kwargs.get("data") == b'{"input": "hi"}'
    assert "json" not in kwargs
    assert kwargs["headers"] == {"X-Signed": "1"}


def test_responses_handler_signs_after_fake_stream_prep_strips_stream():
    """Fake-stream signing-order invariant: the bytes SIGNED must equal the bytes SENT.

    In the streaming + fake-stream path the handler first runs
    _prepare_fake_stream_request, which pops "stream" out of the body, and only
    then calls sign_request. If signing ran before that pop, the signed body
    would still carry "stream" while the body sent over the wire would not,
    producing a SigV4 payload-hash mismatch (401) for a real Mantle deployment.
    We snapshot request_data at sign time and assert "stream" is already gone.
    """
    from unittest.mock import MagicMock
    from litellm.llms.custom_httpx.http_handler import HTTPHandler
    from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
    from litellm.types.llms.openai import ResponsesAPIResponse
    from litellm.types.router import GenericLiteLLMParams

    provider_config = MagicMock()
    provider_config.validate_environment.return_value = {}
    provider_config.get_complete_url.return_value = (
        "https://bedrock-mantle.us-east-2.api.aws/openai/v1/responses"
    )
    provider_config.transform_responses_api_request.return_value = {
        "input": "hi",
        "stream": True,
    }
    provider_config.should_fake_stream.return_value = True
    provider_config.transform_response_api_response.return_value = ResponsesAPIResponse(
        id="resp_1",
        created_at=0,
        output=[],
        status="completed",
        model="openai.gpt-5.5",
    )

    captured = {}

    def _capture_sign(**kwargs):
        captured["request_data"] = dict(kwargs["request_data"])
        return ({"X-Signed": "1"}, b'{"input": "hi"}')

    provider_config.sign_request.side_effect = _capture_sign

    mock_client = MagicMock(spec=HTTPHandler)
    mock_client.post.return_value = MagicMock()

    handler = BaseLLMHTTPHandler()
    handler.response_api_handler(
        model="openai.gpt-5.5",
        input="hi",
        responses_api_provider_config=provider_config,
        response_api_optional_request_params={"stream": True},
        custom_llm_provider="bedrock_mantle",
        litellm_params=GenericLiteLLMParams(aws_region_name="us-east-2"),
        logging_obj=MagicMock(),
        client=mock_client,
        _is_async=False,
        fake_stream=True,
    )

    assert "stream" not in captured["request_data"]
    assert "input" in captured["request_data"]

    post_kwargs = mock_client.post.call_args.kwargs
    assert post_kwargs.get("data") == b'{"input": "hi"}'
    assert "json" not in post_kwargs
    assert "stream" in post_kwargs


def _make_compact_handler_call(signed_body, is_async):
    """Drive (async_)compact_response_api_handler with a fully mocked provider config
    + client, returning the kwargs the client.post was called with.

    signed_body=None simulates a no-op (non-signing) provider; bytes simulates a
    signing provider (e.g. Bedrock Mantle SigV4 / bearer).
    """
    from unittest.mock import MagicMock
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
    from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
    from litellm.types.router import GenericLiteLLMParams

    compact_url = "https://bedrock-mantle.us-east-2.api.aws/openai/v1/responses/compact"
    provider_config = MagicMock()
    provider_config.validate_environment.return_value = {}
    provider_config.get_complete_url.return_value = (
        "https://bedrock-mantle.us-east-2.api.aws/openai/v1/responses"
    )
    provider_config.transform_compact_response_api_request.return_value = (
        compact_url,
        {"model": "openai.gpt-5.5", "input": "hi"},
    )
    provider_config.sign_request.return_value = ({"X-Signed": "1"}, signed_body)
    provider_config.transform_compact_response_api_response.return_value = "ok"

    spec = AsyncHTTPHandler if is_async else HTTPHandler
    mock_client = MagicMock(spec=spec)
    if is_async:
        mock_client.post = AsyncMock(return_value=MagicMock())
    else:
        mock_client.post.return_value = MagicMock()

    handler = BaseLLMHTTPHandler()
    result = handler.compact_response_api_handler(
        model="openai.gpt-5.5",
        input="hi",
        responses_api_provider_config=provider_config,
        response_api_optional_request_params={},
        custom_llm_provider="bedrock_mantle",
        litellm_params=GenericLiteLLMParams(aws_region_name="us-east-2"),
        logging_obj=MagicMock(),
        client=mock_client,
        _is_async=is_async,
    )
    if is_async:
        asyncio.run(result)
    return provider_config, mock_client.post.call_args.kwargs


def test_compact_handler_sends_json_when_not_signed():
    """No-op provider on compact (signed_body is None) -> posts json=data, no data= bytes."""
    provider_config, kwargs = _make_compact_handler_call(
        signed_body=None, is_async=False
    )
    provider_config.sign_request.assert_called_once()
    assert kwargs.get("json") == {"model": "openai.gpt-5.5", "input": "hi"}
    assert "data" not in kwargs


def test_compact_handler_sends_signed_bytes_when_signed():
    """Signing provider on compact -> posts the signed bytes via data=, not json=.

    Regression for the adversarial-review finding that /responses/compact bypassed
    the SigV4 signing hook, so IAM-only Mantle callers sent unsigned bodies.
    """
    provider_config, kwargs = _make_compact_handler_call(
        signed_body=b'{"model": "openai.gpt-5.5", "input": "hi"}', is_async=False
    )
    assert kwargs.get("data") == b'{"model": "openai.gpt-5.5", "input": "hi"}'
    assert "json" not in kwargs
    assert kwargs["headers"] == {"X-Signed": "1"}
    # signing must use the compact endpoint as api_base, not the create URL
    assert provider_config.sign_request.call_args.kwargs["api_base"].endswith(
        "/openai/v1/responses/compact"
    )


def test_async_compact_handler_sends_signed_bytes_when_signed():
    """Async compact must sign identically to sync (same omission in the async twin)."""
    provider_config, kwargs = _make_compact_handler_call(
        signed_body=b'{"model": "openai.gpt-5.5", "input": "hi"}', is_async=True
    )
    assert kwargs.get("data") == b'{"model": "openai.gpt-5.5", "input": "hi"}'
    assert "json" not in kwargs
    assert kwargs["headers"] == {"X-Signed": "1"}


def test_async_compact_handler_sends_json_when_not_signed():
    """Async no-op provider on compact -> posts json=data, no data= bytes."""
    _provider_config, kwargs = _make_compact_handler_call(
        signed_body=None, is_async=True
    )
    assert kwargs.get("json") == {"model": "openai.gpt-5.5", "input": "hi"}
    assert "data" not in kwargs


async def _post_timeout_for_anthropic_messages(
    litellm_params: GenericLiteLLMParams,
    stream: bool,
):
    """Drive async_anthropic_messages_handler with the network mocked and return
    the timeout passed to the per-request .post call (the value that populates
    aiohttp's sock_read), or None if .post was never reached."""
    handler = BaseLLMHTTPHandler()

    mock_config = Mock()
    mock_config.validate_anthropic_messages_environment = Mock(
        return_value=({"x-api-key": "test-key"}, "https://api.anthropic.com")
    )
    mock_config.transform_anthropic_messages_request = Mock(
        return_value={"model": "claude-sonnet-4-20250514", "messages": []}
    )
    mock_config.sign_request = Mock(return_value=({"x-api-key": "test-key"}, None))
    mock_config.get_complete_url = Mock(
        return_value="https://api.anthropic.com/v1/messages"
    )
    mock_config.max_retry_on_anthropic_messages_http_error = 1
    mock_config.transform_anthropic_messages_response = Mock(return_value={"ok": True})
    mock_config.get_async_streaming_response_iterator = Mock(return_value=iter([]))

    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = Mock()

    captured = {"post_kwargs": None}

    def fake_get_client(*args, **kwargs):
        client = AsyncMock()

        async def fake_post(**post_kwargs):
            captured["post_kwargs"] = post_kwargs
            return mock_response

        client.post = fake_post
        return client

    mock_logging_obj = Mock()
    mock_logging_obj.update_from_kwargs = Mock()
    mock_logging_obj.pre_call = Mock()
    mock_logging_obj.model_call_details = {}
    mock_logging_obj.stream = stream
    mock_logging_obj.dynamic_success_callbacks = []

    with patch(
        "litellm.llms.custom_httpx.llm_http_handler.get_async_httpx_client",
        side_effect=fake_get_client,
    ):
        await handler.async_anthropic_messages_handler(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "Hello"}],
            anthropic_messages_provider_config=mock_config,
            anthropic_messages_optional_request_params={},
            custom_llm_provider="anthropic",
            litellm_params=litellm_params,
            logging_obj=mock_logging_obj,
            client=None,
            stream=stream,
            kwargs={},
        )

    post_kwargs = captured["post_kwargs"]
    return post_kwargs.get("timeout") if post_kwargs is not None else None


@pytest.mark.asyncio
async def test_anthropic_messages_honors_configured_timeout():
    """Regression: /v1/messages must honor litellm_params.timeout instead of the
    hardcoded 600s default. The per-request .post timeout is what reaches
    aiohttp's sock_read, so the configured value must land there."""
    timeout = await _post_timeout_for_anthropic_messages(
        litellm_params=GenericLiteLLMParams(timeout=1800),
        stream=False,
    )
    assert timeout == 1800.0


@pytest.mark.asyncio
async def test_anthropic_messages_uses_stream_timeout_when_streaming():
    """stream_timeout wins over timeout when stream=True."""
    timeout = await _post_timeout_for_anthropic_messages(
        litellm_params=GenericLiteLLMParams(timeout=1800, stream_timeout=300),
        stream=True,
    )
    assert timeout == 300.0


@pytest.mark.asyncio
async def test_anthropic_messages_ignores_stream_timeout_when_not_streaming():
    """stream_timeout must not apply to non-streaming calls; timeout is used."""
    timeout = await _post_timeout_for_anthropic_messages(
        litellm_params=GenericLiteLLMParams(timeout=1800, stream_timeout=300),
        stream=False,
    )
    assert timeout == 1800.0


@pytest.mark.asyncio
async def test_anthropic_messages_falls_back_to_timeout_when_streaming_without_stream_timeout():
    """stream=True but only timeout set -> timeout is used for the stream."""
    timeout = await _post_timeout_for_anthropic_messages(
        litellm_params=GenericLiteLLMParams(timeout=1800),
        stream=True,
    )
    assert timeout == 1800.0


@pytest.mark.asyncio
async def test_anthropic_messages_no_explicit_timeout_when_unset():
    """When neither timeout nor stream_timeout is configured, no explicit timeout
    is forced; .post gets None so the client default path is left untouched."""
    timeout = await _post_timeout_for_anthropic_messages(
        litellm_params=GenericLiteLLMParams(),
        stream=False,
    )
    assert timeout is None


@pytest.mark.asyncio
async def test_anthropic_messages_coerces_string_stream_timeout():
    """Numeric-string stream_timeout (e.g. from YAML) is coerced to float."""
    timeout = await _post_timeout_for_anthropic_messages(
        litellm_params=GenericLiteLLMParams(stream_timeout="300"),
        stream=True,
    )
    assert timeout == 300.0


@pytest.mark.asyncio
async def test_anthropic_messages_zero_timeout_treated_as_unset():
    """A non-positive timeout must not force an immediate-fail 0s read timeout;
    it falls back to None so the client default applies."""
    timeout = await _post_timeout_for_anthropic_messages(
        litellm_params=GenericLiteLLMParams(timeout=0),
        stream=False,
    )
    assert timeout is None


def test_resolve_anthropic_messages_timeout_selection_and_coercion():
    """Unit-level coverage of the resolver: stream selection, float coercion,
    non-positive -> None, and unset -> None."""
    resolve = BaseLLMHTTPHandler._resolve_anthropic_messages_timeout

    assert resolve(GenericLiteLLMParams(timeout=1800), stream=False) == 1800.0
    assert (
        resolve(GenericLiteLLMParams(timeout=1800, stream_timeout=300), stream=True)
        == 300.0
    )
    assert (
        resolve(GenericLiteLLMParams(timeout=1800, stream_timeout=300), stream=False)
        == 1800.0
    )
    assert resolve(GenericLiteLLMParams(stream_timeout="300"), stream=True) == 300.0
    assert resolve(GenericLiteLLMParams(timeout=0), stream=False) is None
    assert resolve(GenericLiteLLMParams(), stream=False) is None


def test_resolve_anthropic_messages_timeout_passes_httpx_timeout_through():
    """An httpx.Timeout is forwarded untouched, not collapsed to a float."""
    explicit = httpx.Timeout(timeout=1800.0, connect=5.0)
    resolved = BaseLLMHTTPHandler._resolve_anthropic_messages_timeout(
        GenericLiteLLMParams(timeout=explicit), stream=False
    )
    assert resolved is explicit


def test_resolve_anthropic_messages_timeout_does_not_read_env(monkeypatch):
    """Security regression: a client-supplied os.environ/ timeout string must NOT
    resolve a server environment variable. Request-time secret resolution would let
    a caller exfiltrate env-backed secrets via the float() error message; config-time
    resolution (router.set_model_list / proxy load_config) already handles the
    legitimate os.environ/ form before the request. The string is coerced as-is and
    raises with the literal input, never the secret value."""
    secret = "sk-ant-super-secret-value"
    monkeypatch.setenv("ANTHROPIC_API_KEY", secret)

    with pytest.raises(ValueError) as exc_info:
        BaseLLMHTTPHandler._resolve_anthropic_messages_timeout(
            GenericLiteLLMParams(timeout="os.environ/ANTHROPIC_API_KEY"),
            stream=False,
        )

    assert secret not in str(exc_info.value)
    assert "os.environ/ANTHROPIC_API_KEY" in str(exc_info.value)
