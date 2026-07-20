"""
Tests for the Rubrik LiteLLM plugin.

Covers initialization, apply_guardrail (prompt moderation + response/tool
blocking), batch logging, and Anthropic format handling.
"""

import os
from typing import Any, Dict
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from litellm.integrations.custom_guardrail import ModifyResponseException
from litellm.integrations.rubrik import (
    RubrikLogger,
    _MalformedToolBlockingResponseError,
)

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
                handler.response_moderation_endpoint
                == "http://localhost:8080/v1/after_completion/openai/v1"
            )
            assert handler.logging_endpoint == "http://localhost:8080/v1/litellm/batch"
            assert handler.key == "test-api-key"
            assert isinstance(handler.moderation_client, httpx.AsyncClient)

    def test_init_with_constructor_params(self):
        with patch("asyncio.create_task", Mock()):
            handler = RubrikLogger(api_key="ctor-key", api_base="http://ctor-host:9090")
            assert handler.key == "ctor-key"
            assert (
                handler.response_moderation_endpoint
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
                    RubrikLogger().response_moderation_endpoint
                    == "http://localhost:8080/v1/after_completion/openai/v1"
                )

    def test_v1_suffix_stripped_as_substring_not_charset(self):
        with patch("asyncio.create_task", Mock()):
            with patch.dict(os.environ, {"RUBRIK_WEBHOOK_URL": "http://host/v1"}):
                assert (
                    RubrikLogger().response_moderation_endpoint
                    == "http://host/v1/after_completion/openai/v1"
                )

            with patch.dict(os.environ, {"RUBRIK_WEBHOOK_URL": "http://host/v11"}):
                assert (
                    RubrikLogger().response_moderation_endpoint
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

    def test_init_outside_event_loop_does_not_raise(self):
        """Instantiation without a running event loop must not raise RuntimeError."""
        with patch.dict(
            os.environ,
            {"RUBRIK_WEBHOOK_URL": "http://localhost:8080", "RUBRIK_API_KEY": "k"},
        ):
            # Do NOT patch asyncio.create_task — the real call should be
            # guarded and fall back gracefully when there is no event loop.
            handler = RubrikLogger()
            assert handler.response_moderation_endpoint.startswith("http://localhost:8080")
            # Without a running loop at init, the periodic flush task should be
            # deferred so batches still get drained once a log event arrives.
            assert handler._periodic_flush_task is None

    @pytest.mark.asyncio
    async def test_periodic_flush_task_started_lazily_on_first_log(self, mock_env):
        """Loggers instantiated outside an event loop must still start the
        periodic flush task on first use to drain low-traffic batches."""
        # Simulate sync-init by hiding the running loop from the constructor.
        with patch(
            "litellm.integrations.rubrik.asyncio.get_running_loop",
            side_effect=RuntimeError("no running loop"),
        ):
            handler = RubrikLogger()
        assert handler._periodic_flush_task is None

        kwargs = {
            "standard_logging_object": {
                "messages": [{"role": "user", "content": "hi"}],
                "id": "litellm-id",
            },
            "litellm_call_id": "litellm-id",
            "litellm_params": {},
        }
        with patch.object(handler, "_log_batch_to_rubrik", AsyncMock()):
            await handler.async_log_success_event(kwargs, None, None, None)

        assert handler._periodic_flush_task is not None
        handler._periodic_flush_task.cancel()

    def test_event_hook_defaults_to_post_call_when_none_passed(self, mock_env):
        """`initialize_guardrail` always passes ``event_hook=litellm_params.mode``
        (which is ``None`` when the user omits ``mode``). The logger must coerce
        a None ``event_hook`` to ``post_call`` rather than leaving it as None,
        which would otherwise cause the guardrail to run on every event hook."""
        from litellm.types.guardrails import GuardrailEventHooks

        with patch("asyncio.create_task", Mock()):
            handler = RubrikLogger(event_hook=None)
            assert handler.event_hook == GuardrailEventHooks.post_call

    def test_explicit_event_hook_preserved(self, mock_env):
        from litellm.types.guardrails import GuardrailEventHooks

        with patch("asyncio.create_task", Mock()):
            handler = RubrikLogger(event_hook=GuardrailEventHooks.pre_call)
            assert handler.event_hook == GuardrailEventHooks.pre_call

    def test_default_on_defaults_to_true_when_none_passed(self, mock_env):
        """`initialize_guardrail` always passes ``default_on=litellm_params.default_on``
        (which is ``None`` when the user omits ``default_on``). The logger must
        coerce a None ``default_on`` to True, otherwise ``should_run_guardrail``
        (which checks ``self.default_on is True``) silently skips the guardrail."""
        with patch("asyncio.create_task", Mock()):
            handler = RubrikLogger(default_on=None)
            assert handler.default_on is True

    def test_explicit_default_on_false_preserved(self, mock_env):
        """A user explicitly setting ``default_on: false`` in their guardrail
        config must NOT be silently overridden to True."""
        with patch("asyncio.create_task", Mock()):
            handler = RubrikLogger(default_on=False)
            assert handler.default_on is False

    def test_explicit_default_on_true_preserved(self, mock_env):
        with patch("asyncio.create_task", Mock()):
            handler = RubrikLogger(default_on=True)
            assert handler.default_on is True

    def test_headers_with_api_key(self, handler):
        assert handler._headers["Authorization"] == "Bearer test-api-key"
        assert handler._headers["Content-Type"] == "application/json"

    def test_headers_without_api_key(self):
        with patch.dict(os.environ, {"RUBRIK_WEBHOOK_URL": "http://host"}, clear=True):
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

    async def test_flush_queue_preserves_events_added_during_send(self, handler):
        handler.log_queue = [{"msg": "a"}, {"msg": "b"}]

        async def mock_post(*_args, **_kwargs):
            handler.log_queue.append({"msg": "c"})
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            return mock_response

        handler.async_httpx_client = AsyncMock()
        handler.async_httpx_client.post = mock_post

        await handler.flush_queue()

        assert handler.log_queue == [{"msg": "c"}]

    async def test_async_send_batch_does_not_drain_events(self, handler):
        handler.log_queue = [{"msg": "a"}, {"msg": "b"}]

        async def mock_post(*_args, **_kwargs):
            handler.log_queue.append({"msg": "c"})
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            return mock_response

        handler.async_httpx_client = AsyncMock()
        handler.async_httpx_client.post = mock_post

        await handler.async_send_batch()

        assert handler.log_queue == [{"msg": "a"}, {"msg": "b"}, {"msg": "c"}]

    async def test_log_batch_error_does_not_crash_and_preserves_events(self, handler):
        """A failed batch send must not crash the caller AND must preserve the
        original events in the queue so they can be retried on the next flush.
        Previously the events were silently dropped on HTTP 5xx / network errors.
        """
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
        assert handler.log_queue == [{"msg": "a"}]

    async def test_log_batch_network_error_preserves_events(self, handler):
        """Network/timeout errors must also preserve the in-flight events."""
        handler.log_queue = [{"msg": "a"}, {"msg": "b"}]
        handler.async_httpx_client = AsyncMock()
        handler.async_httpx_client.post = AsyncMock(
            side_effect=httpx.TimeoutException("timeout")
        )
        await handler.flush_queue()
        assert handler.log_queue == [{"msg": "a"}, {"msg": "b"}]

    async def test_enqueue_drops_oldest_when_queue_exceeds_max_size(self, handler):
        """A sustained Rubrik webhook outage must not let the in-memory retry
        queue grow without bound. Once max_queue_size is exceeded, the oldest
        events are dropped to make room for new ones."""
        handler.max_queue_size = 3
        handler.batch_size = 10**6  # disable size-triggered flush
        handler.flush_queue = AsyncMock()
        for i in range(5):
            await handler._enqueue_log_event(
                kwargs={
                    "standard_logging_object": {
                        "messages": [{"role": "user", "content": f"hi-{i}"}],
                        "response": "hello",
                    },
                },
                event_type="success",
            )
        assert len(handler.log_queue) == 3
        retained = [item["messages"][0]["content"] for item in handler.log_queue]
        assert retained == ["hi-2", "hi-3", "hi-4"]

    async def test_log_batch_failure_preserves_events_added_during_send(self, handler):
        """Failure must preserve both the snapshot AND events appended mid-flush."""
        handler.log_queue = [{"msg": "a"}, {"msg": "b"}]

        async def mock_post(*_args, **_kwargs):
            handler.log_queue.append({"msg": "c"})
            mock_response = Mock()
            mock_response.status_code = 500
            mock_response.text = "boom"
            mock_response.raise_for_status = Mock(
                side_effect=httpx.HTTPStatusError(
                    "err", request=Mock(), response=mock_response
                )
            )
            return mock_response

        handler.async_httpx_client = AsyncMock()
        handler.async_httpx_client.post = mock_post

        await handler.flush_queue()
        assert handler.log_queue == [{"msg": "a"}, {"msg": "b"}, {"msg": "c"}]

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

    async def test_litellm_call_id_always_used_as_correlation_key(self, handler):
        """The merged plugin always uses litellm_call_id as the log ID for all
        providers (not just Anthropic) so that logs correlate with the
        moderation (_blocking) and failure logs for the same request."""
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
        assert handler.log_queue[0]["id"] == "litellm-call-123"

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
        inputs = make_inputs_with_tools([make_tool_call_dict("call_1", "test_tool")])
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

        handler.moderation_client = _echo_service()

        result = await handler.apply_guardrail(
            inputs=inputs, request_data={}, input_type="response"
        )
        assert result is inputs

    async def test_all_blocked(self, handler):
        tc1 = make_tool_call_dict("call_1", "delete_table")
        tc2 = make_tool_call_dict("call_2", "drop_database")
        inputs = make_inputs_with_tools([tc1, tc2])

        handler.moderation_client = _mock_service_response(
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
        handler.moderation_client = mock_client

        with pytest.raises(ModifyResponseException):
            await handler.apply_guardrail(
                inputs=inputs, request_data={}, input_type="response"
            )

    async def test_service_failure_fail_open(self, handler):
        tc1 = make_tool_call_dict("call_1", "test_tool")
        inputs = make_inputs_with_tools([tc1])

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
        handler.moderation_client = mock_client

        result = await handler.apply_guardrail(
            inputs=inputs, request_data={}, input_type="response"
        )
        assert result is inputs

    async def test_service_empty_choices_fail_open(self, handler):
        tc1 = make_tool_call_dict("call_1", "test_tool")
        inputs = make_inputs_with_tools([tc1])

        handler.moderation_client = _mock_service_response({"choices": []})

        result = await handler.apply_guardrail(
            inputs=inputs, request_data={}, input_type="response"
        )
        assert result is inputs

    async def test_blocking_service_payload_format(self, handler):
        tc1 = make_tool_call_dict("call_1", "get_weather", '{"location": "SF"}')
        tc2 = make_tool_call_dict("call_2", "send_email", '{"to": "user@example.com"}')
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
        handler.moderation_client = mock_client

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
        handler.moderation_client = mock_client

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

    async def test_proxy_server_request_not_forwarded(self, handler):
        """proxy_server_request is intentionally NOT included in the request
        envelope: in litellm >=1.83 its ``body`` carries a UserAPIKeyAuth
        instance that breaks json.dumps, silently fail-opening the guardrail."""
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
        handler.moderation_client = mock_client

        logging_obj = Mock()
        logging_obj.model_call_details = {
            "messages": [{"role": "user", "content": "hi"}],
            "model": "gpt-4",
            "litellm_params": {
                "proxy_server_request": {
                    "url": "/chat/completions",
                    "method": "POST",
                    "headers": {
                        "authorization": "Bearer sk-litellm-secret",
                        "cookie": "session=abc",
                        "x-api-key": "leaked-key",
                    },
                    "body": {"api_key": "sk-upstream-secret"},
                },
            },
        }

        await handler.apply_guardrail(
            inputs=inputs,
            request_data={},
            input_type="response",
            logging_obj=logging_obj,
        )

        # proxy_server_request is deliberately excluded from the forwarded envelope
        assert "proxy_server_request" not in captured_payload["request"]


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

        handler.moderation_client = _echo_service()

        result = await handler.apply_guardrail(
            inputs=inputs, request_data={}, input_type="response"
        )
        assert result is inputs

    async def test_single_tool_blocked(self, handler):
        tc = make_tool_call_dict("toolu_123", "dangerous_tool", '{"arg": "value"}')
        inputs = make_inputs_with_tools([tc])

        handler.moderation_client = _mock_service_response(
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

    async def test_text_only_response_sent_to_moderation(self, handler):
        """Text-only responses (no tool calls) are sent to the response
        moderation service to check the assistant's text content."""
        from litellm.types.utils import GenericGuardrailAPIInputs

        inputs = GenericGuardrailAPIInputs(texts=["Hello! I'm Claude."])

        # Service allows the response (returns the content unchanged)
        handler.moderation_client = _echo_service()

        result = await handler.apply_guardrail(
            inputs=inputs, request_data={}, input_type="response"
        )

        assert result is inputs

    async def test_service_failure_preserves_tools(self, handler):
        tc = make_tool_call_dict("toolu_123", "get_weather", '{"location": "SF"}')
        inputs = make_inputs_with_tools([tc])

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
        handler.moderation_client = mock_client

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


# -- Extract response block ----------------------------------------------------


class TestExtractResponseBlock:
    """Tests for _extract_response_block, which replaces the upstream
    _extract_blocked_tools and handles both text blocks and tool blocks."""

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
        result = RubrikLogger._extract_response_block(service_resp, [tc], "")
        assert result is None

    def test_some_blocked_returns_explanation(self):
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
        result = RubrikLogger._extract_response_block(service_resp, [tc1, tc2], "")
        assert result is not None
        assert "blocked fn2" in result.explanation

    def test_empty_choices_raises(self):
        with pytest.raises(_MalformedToolBlockingResponseError):
            RubrikLogger._extract_response_block({"choices": []}, [], "")

    def test_null_tool_calls_treated_as_all_blocked(self):
        from litellm.types.utils import ChatCompletionMessageToolCall, Function

        tc = ChatCompletionMessageToolCall(
            id="call_1", type="function", function=Function(name="fn", arguments="{}")
        )
        service_resp = {
            "choices": [
                {
                    "message": {
                        "tool_calls": None,
                        "content": "blocked everything",
                    }
                }
            ]
        }
        result = RubrikLogger._extract_response_block(service_resp, [tc], "")
        assert result is not None
        assert "blocked everything" in result.explanation

    def test_text_block_detected(self):
        """When the service replaces the response text wholesale, it's a text block."""
        from litellm.types.utils import ChatCompletionMessageToolCall, Function

        service_resp = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [],
                        "content": "This content violates policy.",
                    }
                }
            ]
        }
        result = RubrikLogger._extract_response_block(
            service_resp, [], "Original assistant text."
        )
        assert result is not None
        assert "violates policy" in result.explanation

    def test_tool_block_with_appended_explanation(self):
        """When the service appends an explanation to the original text, only the
        appended part is returned as the explanation."""
        from litellm.types.utils import ChatCompletionMessageToolCall, Function

        tc = ChatCompletionMessageToolCall(
            id="call_1", type="function", function=Function(name="fn", arguments="{}")
        )
        original_text = "Here is my response."
        appended_explanation = "Tool call was blocked."
        service_resp = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [],
                        "content": original_text + "\n\n" + appended_explanation,
                    }
                }
            ]
        }
        result = RubrikLogger._extract_response_block(
            service_resp, [tc], original_text
        )
        assert result is not None
        assert appended_explanation in result.explanation


# -- Sanitize proxy server request -------------------------------------------


class TestSanitizeProxyServerRequest:
    def test_drops_headers_and_body(self):
        proxy_request = {
            "url": "/chat/completions",
            "method": "POST",
            "headers": {
                "authorization": "Bearer sk-litellm-secret",
                "cookie": "session=abc",
                "content-type": "application/json",
            },
            "body": {"api_key": "sk-upstream-secret", "model": "gpt-4"},
        }
        result = RubrikLogger._sanitize_proxy_server_request(proxy_request)
        assert result == {"url": "/chat/completions", "method": "POST"}

    def test_none_passthrough(self):
        assert RubrikLogger._sanitize_proxy_server_request(None) is None

    def test_non_dict_passthrough(self):
        assert RubrikLogger._sanitize_proxy_server_request("not a dict") == "not a dict"

    def test_partial_dict(self):
        result = RubrikLogger._sanitize_proxy_server_request({"url": "/v1/messages"})
        assert result == {"url": "/v1/messages"}


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
        result = RubrikLogger._resolve_model(
            {"response": response}, {"model": "fallback"}
        )
        assert result == "unknown"
