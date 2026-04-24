"""
Tests for the Rubrik LiteLLM plugin.

Covers initialization, apply_guardrail tool blocking (all allowed, all blocked,
partial blocking, fail-open), batch logging, and Anthropic format handling.
"""

import os
from typing import Any, Dict
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from litellm.integrations.custom_guardrail import ModifyResponseException
from litellm.integrations.rubrik import RubrikLogger
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

from tests.test_litellm.integrations.rubrik_test_helpers import (
    make_inputs_with_tools,
    make_tool_call_dict,
)


@pytest.fixture
def mock_env():
    """Set up environment variables for testing."""
    with patch.dict(
        os.environ,
        {
            "RUBRIK_WEBHOOK_URL": "http://localhost:8080",
            "RUBRIK_API_KEY": "test-api-key",
        },
    ):
        yield


@pytest.fixture
def handler(mock_env):
    """Create a RubrikLogger instance for testing."""
    with patch("asyncio.create_task", Mock()):
        return RubrikLogger()


# -- Initialization -----------------------------------------------------------


class TestInitialization:
    def test_init_success(self, mock_env):
        with patch("asyncio.create_task", Mock()):
            handler = RubrikLogger()
            assert (
                handler.tool_blocking_endpoint
                == "http://localhost:8080/v1/after_completion/openai/v1"
            )
            assert handler.logging_endpoint == "http://localhost:8080/v1/litellm/batch"
            assert handler.key == "test-api-key"
            assert isinstance(handler.tool_blocking_client, AsyncHTTPHandler)

    def test_init_with_constructor_params(self):
        with patch("asyncio.create_task", Mock()):
            handler = RubrikLogger(
                api_key="ctor-key", api_base="http://ctor-host:9090"
            )
            assert handler.key == "ctor-key"
            assert (
                handler.tool_blocking_endpoint
                == "http://ctor-host:9090/v1/after_completion/openai/v1"
            )

    def test_init_without_url(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Rubrik webhook URL not configured"):
                RubrikLogger()

    def test_init_without_api_key(self):
        with patch.dict(
            os.environ, {"RUBRIK_WEBHOOK_URL": "http://localhost:8080"}, clear=True
        ):
            with patch("asyncio.create_task", Mock()):
                assert RubrikLogger().key is None

    def test_trailing_slash_removed(self):
        with patch.dict(os.environ, {"RUBRIK_WEBHOOK_URL": "http://localhost:8080/"}):
            with patch("asyncio.create_task", Mock()):
                assert (
                    RubrikLogger().tool_blocking_endpoint
                    == "http://localhost:8080/v1/after_completion/openai/v1"
                )

    def test_v1_suffix_stripped_as_substring_not_charset(self):
        with patch("asyncio.create_task", Mock()):
            with patch.dict(os.environ, {"RUBRIK_WEBHOOK_URL": "http://host/v1"}):
                assert (
                    RubrikLogger().tool_blocking_endpoint
                    == "http://host/v1/after_completion/openai/v1"
                )

            with patch.dict(os.environ, {"RUBRIK_WEBHOOK_URL": "http://host/v11"}):
                assert (
                    RubrikLogger().tool_blocking_endpoint
                    == "http://host/v11/v1/after_completion/openai/v1"
                )

    def test_sampling_rate_fractional(self):
        with patch("asyncio.create_task", Mock()):
            with patch.dict(
                os.environ,
                {"RUBRIK_WEBHOOK_URL": "http://host", "RUBRIK_SAMPLING_RATE": "0.5"},
            ):
                assert RubrikLogger().sampling_rate == 0.5

    def test_sampling_rate_invalid_ignored(self):
        with patch("asyncio.create_task", Mock()):
            with patch.dict(
                os.environ,
                {"RUBRIK_WEBHOOK_URL": "http://host", "RUBRIK_SAMPLING_RATE": "abc"},
            ):
                assert RubrikLogger().sampling_rate == 1.0

    def test_sampling_rate_clamped(self):
        with patch("asyncio.create_task", Mock()):
            with patch.dict(
                os.environ,
                {"RUBRIK_WEBHOOK_URL": "http://host", "RUBRIK_SAMPLING_RATE": "2.0"},
            ):
                assert RubrikLogger().sampling_rate == 1.0
            with patch.dict(
                os.environ,
                {"RUBRIK_WEBHOOK_URL": "http://host", "RUBRIK_SAMPLING_RATE": "-0.5"},
            ):
                assert RubrikLogger().sampling_rate == 0.0

    def test_batch_size_invalid_ignored(self):
        with patch("asyncio.create_task", Mock()):
            with patch.dict(
                os.environ,
                {"RUBRIK_WEBHOOK_URL": "http://host", "RUBRIK_BATCH_SIZE": "abc"},
            ):
                # Should use default without crashing
                assert isinstance(RubrikLogger().batch_size, int)

    def test_batch_size_valid(self):
        with patch("asyncio.create_task", Mock()):
            with patch.dict(
                os.environ,
                {"RUBRIK_WEBHOOK_URL": "http://host", "RUBRIK_BATCH_SIZE": "256"},
            ):
                assert RubrikLogger().batch_size == 256

    def test_headers_with_api_key(self, handler):
        assert handler._headers["Authorization"] == "Bearer test-api-key"
        assert handler._headers["Content-Type"] == "application/json"

    def test_headers_without_api_key(self):
        with patch.dict(
            os.environ, {"RUBRIK_WEBHOOK_URL": "http://host"}, clear=True
        ):
            with patch("asyncio.create_task", Mock()):
                h = RubrikLogger()
                assert "Authorization" not in h._headers


# -- Batch Logging ------------------------------------------------------------


@pytest.mark.asyncio
class TestBatchLogging:
    async def test_log_success_event_appends_to_queue(self, handler):
        kwargs = {
            "standard_logging_object": {
                "messages": [{"role": "user", "content": "hi"}],
                "response": "hello",
            },
        }
        await handler.async_log_success_event(
            kwargs=kwargs, response_obj=None, start_time=None, end_time=None
        )
        assert len(handler.log_queue) == 1

    async def test_log_failure_event_appends_to_queue(self, handler):
        kwargs = {
            "standard_logging_object": {
                "messages": [{"role": "user", "content": "hi"}],
                "response": "error",
            },
        }
        await handler.async_log_failure_event(
            kwargs=kwargs, response_obj=None, start_time=None, end_time=None
        )
        assert len(handler.log_queue) == 1

    async def test_log_success_event_sampling_skips(self, handler):
        handler.sampling_rate = 0.0
        kwargs = {
            "standard_logging_object": {
                "messages": [{"role": "user", "content": "hi"}],
                "response": "hello",
            },
        }
        await handler.async_log_success_event(
            kwargs=kwargs, response_obj=None, start_time=None, end_time=None
        )
        assert len(handler.log_queue) == 0

    async def test_flush_queue_sends_batch(self, handler):
        handler.log_queue = [{"msg": "a"}, {"msg": "b"}]
        mock_response = Mock()
        mock_response.status_code = 200
        handler.async_httpx_client = AsyncMock()
        handler.async_httpx_client.post = AsyncMock(return_value=mock_response)
        await handler.flush_queue()
        handler.async_httpx_client.post.assert_called_once()
        assert len(handler.log_queue) == 0

    async def test_log_batch_error_does_not_crash(self, handler):
        handler.log_queue = [{"msg": "a"}]
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.raise_for_status = Mock(
            side_effect=httpx.HTTPStatusError(
                "err", request=Mock(), response=mock_response
            )
        )
        handler.async_httpx_client = AsyncMock()
        handler.async_httpx_client.post = AsyncMock(return_value=mock_response)
        await handler.flush_queue()
        assert len(handler.log_queue) == 0

    async def test_system_prompt_prepended_to_messages(self, handler):
        kwargs = {
            "standard_logging_object": {
                "messages": [{"role": "user", "content": "hi"}],
                "response": "hello",
            },
            "system": "You are a helpful assistant.",
        }
        await handler.async_log_success_event(
            kwargs=kwargs, response_obj=None, start_time=None, end_time=None
        )
        assert len(handler.log_queue) == 1
        msgs = handler.log_queue[0]["messages"]
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "You are a helpful assistant."

    async def test_system_prompt_with_dict_messages(self, handler):
        kwargs = {
            "standard_logging_object": {
                "messages": {"role": "user", "content": "hi"},
                "response": "hello",
            },
            "system": "Be concise.",
        }
        await handler.async_log_success_event(
            kwargs=kwargs, response_obj=None, start_time=None, end_time=None
        )
        assert len(handler.log_queue) == 1
        msgs = handler.log_queue[0]["messages"]
        assert isinstance(msgs, list)
        assert msgs[0]["role"] == "system"
        assert msgs[1] == {"role": "user", "content": "hi"}

    async def test_anthropic_id_normalization(self, handler):
        kwargs = {
            "standard_logging_object": {
                "id": "chatcmpl-original",
                "messages": [{"role": "user", "content": "hi"}],
                "response": "hello",
            },
            "litellm_params": {
                "proxy_server_request": {
                    "url": "http://proxy/v1/messages",
                },
            },
            "litellm_call_id": "litellm-call-123",
        }
        await handler.async_log_success_event(
            kwargs=kwargs, response_obj=None, start_time=None, end_time=None
        )
        assert handler.log_queue[0]["id"] == "litellm-call-123"

    async def test_non_anthropic_id_unchanged(self, handler):
        kwargs = {
            "standard_logging_object": {
                "id": "chatcmpl-original",
                "messages": [{"role": "user", "content": "hi"}],
                "response": "hello",
            },
            "litellm_params": {
                "proxy_server_request": {
                    "url": "http://proxy/v1/chat/completions",
                },
            },
            "litellm_call_id": "litellm-call-123",
        }
        await handler.async_log_success_event(
            kwargs=kwargs, response_obj=None, start_time=None, end_time=None
        )
        assert handler.log_queue[0]["id"] == "chatcmpl-original"

    async def test_payload_deep_copied_not_mutated(self, handler):
        """Verify the shared standard_logging_object is not mutated."""
        original_payload = {
            "id": "original-id",
            "messages": [{"role": "user", "content": "hi"}],
            "response": "hello",
        }
        kwargs = {
            "standard_logging_object": original_payload,
            "system": "System prompt.",
        }
        await handler.async_log_success_event(
            kwargs=kwargs, response_obj=None, start_time=None, end_time=None
        )
        # Original payload should NOT have been mutated
        assert original_payload["id"] == "original-id"
        assert len(original_payload["messages"]) == 1


# -- Tool Blocking (apply_guardrail) ------------------------------------------


def _mock_service_response(response_json):
    """Create a mock tool blocking client that returns the given JSON."""

    async def mock_post(*_args, **kwargs):
        mock_resp = Mock()
        mock_resp.json.return_value = response_json
        mock_resp.raise_for_status = Mock()
        return mock_resp

    mock_client = AsyncMock()
    mock_client.post = mock_post
    return mock_client


def _echo_service():
    """Create a mock tool blocking client that echoes the payload back."""

    async def mock_post(*_args, **kwargs):
        mock_resp = Mock()
        mock_resp.json.return_value = kwargs.get("json", {}).get("response", {})
        mock_resp.raise_for_status = Mock()
        return mock_resp

    mock_client = AsyncMock()
    mock_client.post = mock_post
    return mock_client


@pytest.mark.asyncio
class TestApplyGuardrail:
    async def test_skips_requests(self, handler):
        inputs = make_inputs_with_tools(
            [make_tool_call_dict("call_1", "test_tool")]
        )
        result = await handler.apply_guardrail(
            inputs=inputs, request_data={}, input_type="request"
        )
        assert result is inputs

    async def test_no_tool_calls(self, handler):
        from litellm.types.utils import GenericGuardrailAPIInputs

        inputs = GenericGuardrailAPIInputs(texts=["hello"])
        result = await handler.apply_guardrail(
            inputs=inputs, request_data={}, input_type="response"
        )
        assert result is inputs

    async def test_all_allowed(self, handler):
        tc1 = make_tool_call_dict("call_1", "get_weather")
        tc2 = make_tool_call_dict("call_2", "get_time")
        inputs = make_inputs_with_tools([tc1, tc2])

        handler.tool_blocking_client = _echo_service()

        result = await handler.apply_guardrail(
            inputs=inputs, request_data={}, input_type="response"
        )
        assert result is inputs

    async def test_all_blocked(self, handler):
        tc1 = make_tool_call_dict("call_1", "delete_table")
        tc2 = make_tool_call_dict("call_2", "drop_database")
        inputs = make_inputs_with_tools([tc1, tc2])

        handler.tool_blocking_client = _mock_service_response(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Tool blocked by policy",
                            "tool_calls": [],
                        }
                    }
                ],
            }
        )

        with pytest.raises(ModifyResponseException) as exc_info:
            await handler.apply_guardrail(
                inputs=inputs, request_data={}, input_type="response"
            )
        assert "Tool blocked by policy" in exc_info.value.message

    async def test_partial_blocking(self, handler):
        tc_blocked = make_tool_call_dict("call_A", "blocked_tool")
        tc_allowed = make_tool_call_dict("call_B", "allowed_tool")
        inputs = make_inputs_with_tools([tc_blocked, tc_allowed])

        async def mock_post(*_args, **kwargs):
            payload = kwargs.get("json", {}).get("response", {})
            all_tcs = payload["choices"][0]["message"]["tool_calls"]
            allowed = [tc for tc in all_tcs if tc.get("id") == "call_B"]
            mock_resp = Mock()
            mock_resp.json.return_value = {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "blocked",
                            "tool_calls": allowed,
                        }
                    }
                ],
            }
            mock_resp.raise_for_status = Mock()
            return mock_resp

        mock_client = AsyncMock()
        mock_client.post = mock_post
        handler.tool_blocking_client = mock_client

        with pytest.raises(ModifyResponseException):
            await handler.apply_guardrail(
                inputs=inputs, request_data={}, input_type="response"
            )

    async def test_service_failure_fail_open(self, handler):
        tc1 = make_tool_call_dict("call_1", "test_tool")
        inputs = make_inputs_with_tools([tc1])

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
        handler.tool_blocking_client = mock_client

        result = await handler.apply_guardrail(
            inputs=inputs, request_data={}, input_type="response"
        )
        assert result is inputs

    async def test_service_empty_choices_fail_open(self, handler):
        tc1 = make_tool_call_dict("call_1", "test_tool")
        inputs = make_inputs_with_tools([tc1])

        handler.tool_blocking_client = _mock_service_response({"choices": []})

        result = await handler.apply_guardrail(
            inputs=inputs, request_data={}, input_type="response"
        )
        assert result is inputs

    async def test_blocking_service_payload_format(self, handler):
        tc1 = make_tool_call_dict("call_1", "get_weather", '{"location": "SF"}')
        tc2 = make_tool_call_dict(
            "call_2", "send_email", '{"to": "user@example.com"}'
        )
        inputs = make_inputs_with_tools([tc1, tc2])

        captured_payload: Dict[str, Any] = {}

        async def mock_post(*_args, **kwargs):
            captured_payload.update(kwargs.get("json", {}))
            mock_resp = Mock()
            mock_resp.json.return_value = captured_payload.get("response", {})
            mock_resp.raise_for_status = Mock()
            return mock_resp

        mock_client = AsyncMock()
        mock_client.post = mock_post
        handler.tool_blocking_client = mock_client

        await handler.apply_guardrail(
            inputs=inputs, request_data={}, input_type="response"
        )

        # Verify envelope structure
        assert "request" in captured_payload
        assert "response" in captured_payload

        response_data = captured_payload["response"]
        message = response_data["choices"][0]["message"]
        assert message["role"] == "assistant"
        assert len(message["tool_calls"]) == 2
        assert message["tool_calls"][0]["id"] == "call_1"
        assert message["tool_calls"][0]["function"]["name"] == "get_weather"
        assert message["tool_calls"][1]["id"] == "call_2"
        assert message["tool_calls"][1]["function"]["name"] == "send_email"

    async def test_request_data_included_in_envelope(self, handler):
        tc = make_tool_call_dict("call_1", "test_tool")
        inputs = make_inputs_with_tools([tc])

        captured_payload: Dict[str, Any] = {}

        async def mock_post(*_args, **kwargs):
            captured_payload.update(kwargs.get("json", {}))
            mock_resp = Mock()
            mock_resp.json.return_value = captured_payload.get("response", {})
            mock_resp.raise_for_status = Mock()
            return mock_resp

        mock_client = AsyncMock()
        mock_client.post = mock_post
        handler.tool_blocking_client = mock_client

        logging_obj = Mock()
        logging_obj.model_call_details = {
            "messages": [{"role": "user", "content": "hi"}],
            "model": "gpt-4",
            "litellm_params": {
                "proxy_server_request": {"url": "/chat/completions"},
            },
        }

        await handler.apply_guardrail(
            inputs=inputs,
            request_data={},
            input_type="response",
            logging_obj=logging_obj,
        )

        req = captured_payload["request"]
        assert req["model"] == "gpt-4"
        assert req["messages"] == [{"role": "user", "content": "hi"}]


# -- Anthropic format ----------------------------------------------------------


@pytest.mark.asyncio
class TestApplyGuardrailAnthropicFormat:
    """Verify blocking works correctly regardless of original provider format.

    The framework converts Anthropic tool_use blocks to OpenAI-format
    tool_calls before calling apply_guardrail.
    """

    async def test_single_tool_allowed(self, handler):
        tc = make_tool_call_dict(
            "toolu_123", "get_weather", '{"location": "Portland, OR"}'
        )
        inputs = make_inputs_with_tools([tc], texts=["I'll check the weather."])

        handler.tool_blocking_client = _echo_service()

        result = await handler.apply_guardrail(
            inputs=inputs, request_data={}, input_type="response"
        )
        assert result is inputs

    async def test_single_tool_blocked(self, handler):
        tc = make_tool_call_dict("toolu_123", "dangerous_tool", '{"arg": "value"}')
        inputs = make_inputs_with_tools([tc])

        handler.tool_blocking_client = _mock_service_response(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "blocked",
                            "tool_calls": [],
                        }
                    }
                ],
            }
        )

        with pytest.raises(ModifyResponseException):
            await handler.apply_guardrail(
                inputs=inputs, request_data={}, input_type="response"
            )

    async def test_text_only_response_no_blocking(self, handler):
        from litellm.types.utils import GenericGuardrailAPIInputs

        inputs = GenericGuardrailAPIInputs(texts=["Hello! I'm Claude."])

        mock_client = AsyncMock()
        mock_client.post = AsyncMock()
        handler.tool_blocking_client = mock_client

        result = await handler.apply_guardrail(
            inputs=inputs, request_data={}, input_type="response"
        )

        assert result is inputs
        mock_client.post.assert_not_called()

    async def test_service_failure_preserves_tools(self, handler):
        tc = make_tool_call_dict("toolu_123", "get_weather", '{"location": "SF"}')
        inputs = make_inputs_with_tools([tc])

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            side_effect=httpx.TimeoutException("Timeout")
        )
        handler.tool_blocking_client = mock_client

        result = await handler.apply_guardrail(
            inputs=inputs, request_data={}, input_type="response"
        )
        assert result is inputs


# -- Normalize tool calls ------------------------------------------------------


class TestNormalizeToolCalls:
    def test_dict_input(self):
        tc = make_tool_call_dict("call_1", "test", '{"a": 1}')
        result = RubrikLogger._normalize_tool_calls([tc])
        assert len(result) == 1
        assert result[0].id == "call_1"
        assert result[0].function.name == "test"
        assert result[0].function.arguments == '{"a": 1}'

    def test_typed_object_input(self):
        from litellm.types.utils import ChatCompletionMessageToolCall, Function

        tc = ChatCompletionMessageToolCall(
            id="call_2",
            type="function",
            function=Function(name="fn", arguments="{}"),
        )
        result = RubrikLogger._normalize_tool_calls([tc])
        assert len(result) == 1
        assert result[0].id == "call_2"
        assert result[0].function.name == "fn"

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="Cannot normalize"):
            RubrikLogger._normalize_tool_calls(["not_a_tool_call"])


# -- Extract blocked tools -----------------------------------------------------


class TestExtractBlockedTools:
    def test_all_allowed_returns_none(self):
        from litellm.types.utils import ChatCompletionMessageToolCall, Function

        tc = ChatCompletionMessageToolCall(
            id="call_1", type="function", function=Function(name="fn", arguments="{}")
        )
        service_resp = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [{"id": "call_1"}],
                        "content": "",
                    }
                }
            ]
        }
        result = RubrikLogger._extract_blocked_tools(service_resp, [tc])
        assert result is None

    def test_some_blocked_returns_result(self):
        from litellm.types.utils import ChatCompletionMessageToolCall, Function

        tc1 = ChatCompletionMessageToolCall(
            id="call_1",
            type="function",
            function=Function(name="fn1", arguments="{}"),
        )
        tc2 = ChatCompletionMessageToolCall(
            id="call_2",
            type="function",
            function=Function(name="fn2", arguments="{}"),
        )
        service_resp = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [{"id": "call_1"}],
                        "content": "blocked fn2",
                    }
                }
            ]
        }
        result = RubrikLogger._extract_blocked_tools(service_resp, [tc1, tc2])
        assert result is not None
        assert len(result.allowed_tools) == 1
        assert result.allowed_tools[0].id == "call_1"
        assert "blocked fn2" in result.explanation

    def test_empty_choices_raises(self):
        with pytest.raises(Exception, match="empty response"):
            RubrikLogger._extract_blocked_tools({"choices": []}, [])


# -- Resolve model -------------------------------------------------------------


class TestResolveModel:
    def test_model_from_response(self):
        from unittest.mock import Mock

        response = Mock()
        response.model = "gpt-4"
        result = RubrikLogger._resolve_model({"response": response}, {})
        assert result == "gpt-4"

    def test_model_from_call_details(self):
        result = RubrikLogger._resolve_model({}, {"model": "claude-3"})
        assert result == "claude-3"

    def test_fallback_to_unknown(self):
        result = RubrikLogger._resolve_model({}, {})
        assert result == "unknown"

    def test_empty_model_on_response_returns_unknown(self):
        from unittest.mock import Mock

        response = Mock()
        response.model = ""
        result = RubrikLogger._resolve_model({"response": response}, {"model": "fallback"})
        assert result == "unknown"
