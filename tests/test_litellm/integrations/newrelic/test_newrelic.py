import os
import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

# newrelic is a proxy-runtime dependency (pyproject.toml) and is not installed
# in the CI Python environment. Mock it in sys.modules before importing the
# integration so that deferred `import newrelic.agent` calls inside NewRelicLogger
# methods resolve to these mocks rather than failing with ModuleNotFoundError.
_mock_newrelic = MagicMock()
_mock_newrelic_agent = MagicMock()
# Explicitly link so _mock_newrelic.agent IS _mock_newrelic_agent. Without this,
# the first getattr(_mock_newrelic, 'agent') auto-creates a different child mock,
# causing patch("newrelic.agent.xxx") to patch the wrong object.
_mock_newrelic.agent = _mock_newrelic_agent
sys.modules["newrelic"] = _mock_newrelic
sys.modules["newrelic.agent"] = _mock_newrelic_agent

import litellm.integrations.newrelic.newrelic as nr_module
from litellm.integrations.newrelic.newrelic import NewRelicLogger

# The module may have been imported before sys.modules was patched (e.g. via
# litellm's own startup imports), leaving _newrelic_agent=None. Point it at
# the mock agent so all tests see a non-None agent.
nr_module._newrelic_agent = _mock_newrelic_agent


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

NR_ENV = {
    "NEW_RELIC_LICENSE_KEY": "test-license-key",
    "NEW_RELIC_APP_NAME": "test-app",
}


def make_logger(**kwargs) -> NewRelicLogger:
    """Instantiate NewRelicLogger with NR agent calls mocked out."""
    with patch.dict(os.environ, NR_ENV):
        return NewRelicLogger(**kwargs)


def make_kwargs(
    model="gpt-4",
    provider="openai",
    messages=None,
    optional_params=None,
    traceparent=None,
) -> dict:
    """Build a minimal kwargs dict representative of a litellm callback invocation."""
    headers = {}
    if traceparent:
        headers["traceparent"] = traceparent

    return {
        "model": model,
        "messages": messages or [{"role": "user", "content": "Hello"}],
        "optional_params": optional_params or {},
        "litellm_params": {
            "custom_llm_provider": provider,
            "metadata": {"headers": headers},
        },
        "start_time": 1_000_000.0,
        "end_time": 1_000_001.5,
        "llm_api_duration_ms": 1500.0,
    }


def make_response(
    model="gpt-4",
    response_id="chatcmpl-abc123",
    content="Hello there!",
    finish_reason="stop",
    prompt_tokens=10,
    completion_tokens=20,
):
    """Build a minimal ModelResponse-like dict."""
    return {
        "id": response_id,
        "model": model,
        "choices": [
            {
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def make_slo(**overrides):
    """Build a StandardLoggingPayload-like dict with sentinel values distinct from
    make_kwargs/make_response defaults, so tests can prove the SLO branch won."""
    base = {
        "trace_id": "slo-trace-abc",
        "custom_llm_provider": "slo-provider",
        "model": "slo-model",
        "prompt_tokens": 100,
        "completion_tokens": 200,
        "total_tokens": 300,
        "response_time": 1.5,  # seconds; converted to ms by _get_duration
        "model_parameters": {"temperature": 0.7, "max_tokens": 500},
        "startTime": 2_000_000.0,
        "endTime": 2_000_001.5,
        "messages": [{"role": "user", "content": "from-slo"}],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Init / configuration
# ---------------------------------------------------------------------------


class TestNewRelicLoggerInit:
    def test_disabled_when_license_key_missing(self):
        with patch("newrelic.agent.register_application"):
            with patch.dict(os.environ, {"NEW_RELIC_APP_NAME": "app"}, clear=True):
                logger = NewRelicLogger()
        assert logger.enabled is False

    def test_disabled_when_app_name_missing(self):
        with patch("newrelic.agent.register_application"):
            with patch.dict(os.environ, {"NEW_RELIC_LICENSE_KEY": "key"}, clear=True):
                logger = NewRelicLogger()
        assert logger.enabled is False

    def test_enabled_with_valid_env_vars(self):
        with patch("newrelic.agent.register_application"):
            with patch.dict(os.environ, NR_ENV):
                logger = NewRelicLogger()
        assert logger.enabled is True

    def test_disabled_on_import_error(self):
        with patch.object(
            _mock_newrelic_agent, "register_application", side_effect=ImportError
        ):
            with patch.dict(os.environ, NR_ENV):
                logger = NewRelicLogger()
        assert logger.enabled is False

    def test_disabled_on_agent_startup_error(self):
        with patch.object(
            _mock_newrelic_agent,
            "register_application",
            side_effect=RuntimeError("agent startup failed"),
        ):
            with patch.dict(os.environ, NR_ENV):
                logger = NewRelicLogger()
        assert logger.enabled is False

    def test_disabled_when_agent_package_missing(self):
        with patch.object(nr_module, "_newrelic_agent", None):
            with patch.dict(os.environ, NR_ENV):
                logger = NewRelicLogger()
        assert logger.enabled is False

    def test_record_content_default_true(self):
        logger = make_logger()
        assert logger.record_content is True

    def test_record_content_disabled_by_param(self):
        logger = make_logger(turn_off_message_logging=True)
        assert logger.record_content is False

    def test_record_content_disabled_by_env_var(self):
        with patch("newrelic.agent.register_application"):
            with patch.dict(
                os.environ,
                {**NR_ENV, "NEW_RELIC_AI_MONITORING_RECORD_CONTENT_ENABLED": "false"},
            ):
                logger = NewRelicLogger()
                assert logger.record_content is False

    def test_record_content_requires_both_enabled(self):
        """param says record, but env var says no — result is False."""
        with patch("newrelic.agent.register_application"):
            with patch.dict(
                os.environ,
                {**NR_ENV, "NEW_RELIC_AI_MONITORING_RECORD_CONTENT_ENABLED": "false"},
            ):
                logger = NewRelicLogger(turn_off_message_logging=False)
                assert logger.record_content is False

    def test_constructor_kwargs_take_priority_over_global_params(self):
        """Constructor turn_off_message_logging=True must not be overwritten by
        litellm.newrelic_params which defaults turn_off_message_logging to False."""
        from litellm.types.integrations.newrelic import NewRelicInitParams

        with patch("newrelic.agent.register_application"):
            with patch.dict(os.environ, NR_ENV):
                with patch(
                    "litellm.newrelic_params",
                    NewRelicInitParams(turn_off_message_logging=False),
                ):
                    logger = NewRelicLogger(turn_off_message_logging=True)
        assert logger.record_content is False

    def test_newrelic_params_plain_dict_branch(self):
        """litellm.newrelic_params can be a plain dict; it should be validated
        through NewRelicInitParams and its values applied to the logger."""
        with patch("newrelic.agent.register_application"):
            with patch.dict(os.environ, NR_ENV):
                with patch(
                    "litellm.newrelic_params",
                    {"turn_off_message_logging": True},
                ):
                    logger = NewRelicLogger()
        assert logger.turn_off_message_logging is True


# ---------------------------------------------------------------------------
# _parse_bool_env
# ---------------------------------------------------------------------------


class TestParseBoolEnv:
    def setup_method(self):
        self.logger = make_logger()

    @pytest.mark.parametrize("raw", ["true", "TRUE", "True", "1", "yes", "on", "ON"])
    def test_truthy_values(self, raw):
        with patch.dict(os.environ, {"MY_VAR": raw}):
            assert self.logger._parse_bool_env("MY_VAR") is True

    @pytest.mark.parametrize("raw", ["false", "FALSE", "0", "no", "off", "Off"])
    def test_falsy_values(self, raw):
        with patch.dict(os.environ, {"MY_VAR": raw}):
            assert self.logger._parse_bool_env("MY_VAR") is False

    @pytest.mark.parametrize("raw", [" true ", "  1\t", "\nyes"])
    def test_whitespace_tolerance_truthy(self, raw):
        with patch.dict(os.environ, {"MY_VAR": raw}):
            assert self.logger._parse_bool_env("MY_VAR") is True

    @pytest.mark.parametrize("raw", [" false ", "  0\t", "\nno"])
    def test_whitespace_tolerance_falsy(self, raw):
        with patch.dict(os.environ, {"MY_VAR": raw}):
            assert self.logger._parse_bool_env("MY_VAR") is False

    def test_missing_uses_default(self):
        with patch.dict(os.environ, {}, clear=True):
            assert self.logger._parse_bool_env("MY_VAR", default=True) is True
            assert self.logger._parse_bool_env("MY_VAR", default=False) is False

    def test_empty_string_uses_default(self):
        with patch.dict(os.environ, {"MY_VAR": ""}):
            assert self.logger._parse_bool_env("MY_VAR", default=True) is True
            assert self.logger._parse_bool_env("MY_VAR", default=False) is False

    @pytest.mark.parametrize("raw", ["maybe", "2", "enabled", "tru"])
    def test_unrecognised_value_falls_back_to_default_with_warning(self, raw):
        with (
            patch.dict(os.environ, {"MY_VAR": raw}),
            patch.object(nr_module.verbose_logger, "warning") as mock_warn,
        ):
            assert self.logger._parse_bool_env("MY_VAR", default=True) is True
            assert self.logger._parse_bool_env("MY_VAR", default=False) is False
            assert mock_warn.call_count == 2
            # Warning should mention the variable name and the raw value
            for call in mock_warn.call_args_list:
                assert "MY_VAR" in call.args[0]
                assert repr(raw) in call.args[0]


# ---------------------------------------------------------------------------
# _get_trace_context
# ---------------------------------------------------------------------------


class TestGetTraceContext:
    def setup_method(self):
        self.logger = make_logger()

    def test_extracts_trace_id_from_traceparent(self):
        kwargs = make_kwargs(
            traceparent="00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-00"
        )
        trace_id = self.logger._get_trace_context(kwargs)
        assert trace_id == "4bf92f3577b34da6a3ce929d0e0e4736"

    def test_generates_uuid_when_no_headers(self):
        kwargs = make_kwargs()
        trace_id = self.logger._get_trace_context(kwargs)
        assert trace_id is not None
        assert (
            len(trace_id) == 32
        )  # 32-char lowercase hex, matches W3C traceparent format

    def test_generates_uuid_when_traceparent_malformed(self):
        kwargs = make_kwargs(traceparent="not-valid")
        trace_id = self.logger._get_trace_context(kwargs)
        # Falls back to a 32-char lowercase hex, matching W3C traceparent format
        assert trace_id is not None
        assert len(trace_id) == 32

    def test_extracts_trace_id_from_mixed_case_traceparent_header(self):
        # Callers passing headers directly may not normalise case; per W3C spec
        # header names are case-insensitive, so "Traceparent" must work too.
        kwargs = make_kwargs()
        kwargs["litellm_params"]["metadata"]["headers"] = {
            "Traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-00"
        }
        trace_id = self.logger._get_trace_context(kwargs)
        assert trace_id == "4bf92f3577b34da6a3ce929d0e0e4736"

    def test_parse_failure_falls_through_to_synthetic_uuid(self):
        """When parsing upstream sources raises, emit a synthetic UUID rather
        than dropping the event. NR schema requires every AIM event carry a
        trace_id; this method's contract is to always return a valid string.
        """
        # Non-dict headers value forces .items() to raise inside the try
        kwargs = {"litellm_params": {"metadata": {"headers": "not-a-dict"}}}
        trace_id = self.logger._get_trace_context(kwargs)
        assert trace_id is not None
        assert len(trace_id) == 32  # 32-char lowercase hex fallback


# ---------------------------------------------------------------------------
# _extract_message_content edge cases
# ---------------------------------------------------------------------------


class TestExtractMessageContent:
    def setup_method(self):
        self.logger = make_logger()

    def test_plain_text(self):
        assert self.logger._extract_message_content({"content": "hello"}) == "hello"

    def test_none_content_returns_empty_string(self):
        assert self.logger._extract_message_content({"content": None}) == ""

    def test_missing_content_returns_empty_string(self):
        assert self.logger._extract_message_content({}) == ""

    def test_tool_calls_serialized_as_json(self):
        msg = {
            "content": None,
            "tool_calls": [{"id": "call_1", "function": {"name": "get_weather"}}],
        }
        result = self.logger._extract_message_content(msg)
        assert "get_weather" in result
        assert "call_1" in result

    def test_multimodal_list_serialized_as_json(self):
        msg = {
            "content": [
                {"type": "text", "text": "describe this"},
                {"type": "image_url"},
            ]
        }
        result = self.logger._extract_message_content(msg)
        assert "describe this" in result
        assert "image_url" in result

    def test_non_string_content_coerced_to_str(self):
        """Numeric/bool content passes the None and list guards; final branch coerces to str."""
        assert self.logger._extract_message_content({"content": 123}) == "123"
        assert self.logger._extract_message_content({"content": True}) == "True"


# ---------------------------------------------------------------------------
# _extract_all_messages — record_content=False path
# ---------------------------------------------------------------------------


class TestExtractAllMessagesContentDisabled:
    def test_no_content_key_when_recording_disabled(self):
        logger = make_logger(turn_off_message_logging=True)
        kwargs = make_kwargs(messages=[{"role": "user", "content": "secret"}])
        response = make_response(content="also secret")

        messages = logger._extract_all_messages(
            kwargs, response, response_model="gpt-4", vendor="openai"
        )

        for msg in messages:
            assert "content" not in msg


class TestExtractAllMessagesTimestamps:
    def setup_method(self):
        self.logger = make_logger()

    def test_input_messages_get_start_time_timestamp(self):
        kwargs = make_kwargs(messages=[{"role": "user", "content": "Hi"}])
        # make_kwargs sets start_time=1_000_000.0 and end_time=1_000_001.5
        response = make_response()

        messages = self.logger._extract_all_messages(
            kwargs, response, response_model="gpt-4", vendor="openai"
        )

        input_msg = next(m for m in messages if not m.get("is_response"))
        assert input_msg["timestamp"] == int(1_000_000.0 * 1000.0)

    def test_output_messages_get_end_time_timestamp(self):
        kwargs = make_kwargs(messages=[{"role": "user", "content": "Hi"}])
        response = make_response()

        messages = self.logger._extract_all_messages(
            kwargs, response, response_model="gpt-4", vendor="openai"
        )

        output_msg = next(m for m in messages if m.get("is_response"))
        assert output_msg["timestamp"] == int(1_000_001.5 * 1000.0)

    def test_timestamp_forwarded_to_event_data(self):
        logger = make_logger()
        mock_app = MagicMock()
        mock_app.enabled = True

        kwargs = make_kwargs(
            traceparent="00-aabbccddeeff00112233445566778899-0011223344556677-01",
            messages=[{"role": "user", "content": "Hi"}],
        )
        response = make_response()

        with patch("newrelic.agent.application", return_value=mock_app):
            logger._process_success(kwargs, response, start_time=1.0, end_time=2.5)

        calls = mock_app.record_custom_event.call_args_list
        message_events = [
            c[0][1] for c in calls if c[0][0] == "LlmChatCompletionMessage"
        ]
        for event in message_events:
            assert "timestamp" in event


# ---------------------------------------------------------------------------
# Streaming response handling
# ---------------------------------------------------------------------------


def make_streaming_response(
    model="gpt-4",
    response_id="chatcmpl-stream123",
    content="Hello from streaming!",
    finish_reason="stop",
    prompt_tokens=8,
    completion_tokens=15,
):
    """Build a streaming-assembled response dict using 'delta' instead of 'message'."""
    return {
        "id": response_id,
        "model": model,
        "choices": [
            {
                "delta": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


class TestStreamingResponse:
    """Verify graceful handling of streaming-assembled responses.

    When LiteLLM assembles a streaming response, some providers produce a
    final choice dict with a 'delta' key instead of 'message'. The integration
    must extract content from either key without raising.
    """

    def setup_method(self):
        self.logger = make_logger()

    def test_extracts_content_from_delta_key(self):
        kwargs = make_kwargs(messages=[{"role": "user", "content": "Hi"}])
        response = make_streaming_response(content="Streamed reply")

        messages = self.logger._extract_all_messages(
            kwargs, response, response_model="gpt-4", vendor="openai"
        )

        response_msgs = [m for m in messages if m.get("is_response")]
        assert len(response_msgs) == 1
        assert response_msgs[0]["content"] == "Streamed reply"
        assert response_msgs[0]["role"] == "assistant"

    def test_streaming_response_records_summary_and_message_events(self):
        mock_app = MagicMock()
        mock_app.enabled = True

        kwargs = make_kwargs(
            traceparent="00-aabbccddeeff00112233445566778899-0011223344556677-01",
            messages=[{"role": "user", "content": "Hi"}],
        )
        response = make_streaming_response(
            response_id="chatcmpl-stream123",
            content="Streamed reply",
            finish_reason="stop",
            prompt_tokens=8,
            completion_tokens=15,
        )

        with patch("newrelic.agent.application", return_value=mock_app):
            self.logger._process_success(kwargs, response, start_time=1.0, end_time=2.0)

        calls = mock_app.record_custom_event.call_args_list
        event_types = [c[0][0] for c in calls]
        assert "LlmChatCompletionSummary" in event_types
        assert "LlmChatCompletionMessage" in event_types

        message_events = [
            c[0][1] for c in calls if c[0][0] == "LlmChatCompletionMessage"
        ]
        response_msg = next((e for e in message_events if e.get("is_response")), None)
        assert response_msg is not None
        assert response_msg["content"] == "Streamed reply"

    @pytest.mark.asyncio
    async def test_async_log_success_event_streaming(self):
        """async_log_success_event is the primary entry point for streaming calls."""
        mock_app = MagicMock()
        mock_app.enabled = True

        kwargs = make_kwargs(messages=[{"role": "user", "content": "Hi"}])
        response = make_streaming_response()

        with patch("newrelic.agent.application", return_value=mock_app):
            await self.logger.async_log_success_event(
                kwargs, response, start_time=1.0, end_time=2.0
            )

        calls = mock_app.record_custom_event.call_args_list
        event_types = [c[0][0] for c in calls]
        assert "LlmChatCompletionSummary" in event_types
        assert "LlmChatCompletionMessage" in event_types

    def test_no_content_when_recording_disabled_streaming(self):
        logger = make_logger(turn_off_message_logging=True)
        kwargs = make_kwargs(messages=[{"role": "user", "content": "secret"}])
        response = make_streaming_response(content="also secret")

        messages = logger._extract_all_messages(
            kwargs, response, response_model="gpt-4", vendor="openai"
        )

        for msg in messages:
            assert "content" not in msg


# ---------------------------------------------------------------------------
# Explicit-None defensive tests
# ---------------------------------------------------------------------------


class TestExplicitNoneValues:
    """Verify that explicitly None values in kwargs/response don't raise or silently drop events."""

    def setup_method(self):
        self.logger = make_logger()

    # _get_trace_context — chained dict lookups
    def test_trace_context_litellm_params_none(self):
        kwargs = make_kwargs()
        kwargs["litellm_params"] = None
        trace_id = self.logger._get_trace_context(kwargs)
        assert trace_id is not None  # falls back to UUID

    def test_trace_context_metadata_none(self):
        kwargs = make_kwargs()
        kwargs["litellm_params"] = {"metadata": None}
        trace_id = self.logger._get_trace_context(kwargs)
        assert trace_id is not None

    def test_trace_context_headers_none(self):
        kwargs = make_kwargs()
        kwargs["litellm_params"] = {"metadata": {"headers": None}}
        trace_id = self.logger._get_trace_context(kwargs)
        assert trace_id is not None

    # _get_request_params
    def test_request_params_optional_params_none(self):
        assert self.logger._get_request_params({"optional_params": None}) == {}

    # _get_model_names
    def test_model_names_model_none_in_kwargs(self):
        request_model, _ = self.logger._get_model_names(
            {"model": None}, make_response()
        )
        assert request_model == "unknown"

    def test_model_names_model_none_in_response(self):
        response = make_response()
        response["model"] = None
        _, response_model = self.logger._get_model_names(make_kwargs(), response)
        assert response_model == "gpt-4"  # falls back to request_model from kwargs

    # _extract_all_messages
    def test_extract_messages_messages_none(self):
        kwargs = make_kwargs()
        kwargs["messages"] = None
        response = make_response()
        messages = self.logger._extract_all_messages(
            kwargs, response, response_model="gpt-4", vendor="openai"
        )
        # No request messages, but response message should still be extracted
        assert any(m.get("is_response") for m in messages)

    def test_extract_messages_choices_none(self):
        kwargs = make_kwargs(messages=[{"role": "user", "content": "Hi"}])
        response = make_response()
        response["choices"] = None
        messages = self.logger._extract_all_messages(
            kwargs, response, response_model="gpt-4", vendor="openai"
        )
        # No response messages, but request message should still be extracted
        assert any(not m.get("is_response") for m in messages)


# ---------------------------------------------------------------------------
# Helper edge cases
# ---------------------------------------------------------------------------


class TestExtractUsage:
    def setup_method(self):
        self.logger = make_logger()

    def test_missing_usage_returns_zeros(self):
        response = {"id": "r1", "model": "gpt-4", "choices": []}
        usage = self.logger._extract_usage(response)
        assert usage == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def test_explicit_none_token_fields_return_zeros(self):
        response = {
            "usage": {
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
            }
        }
        usage = self.logger._extract_usage(response)
        assert usage == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


class TestGetFinishReason:
    def setup_method(self):
        self.logger = make_logger()

    def test_returns_unknown_when_no_choices(self):
        response = {"choices": []}
        assert self.logger._get_finish_reason(response) == "unknown"

    def test_returns_unknown_when_choices_missing(self):
        assert self.logger._get_finish_reason({}) == "unknown"

    def test_returns_unknown_when_finish_reason_explicitly_none(self):
        response = {"choices": [{"finish_reason": None}]}
        assert self.logger._get_finish_reason(response) == "unknown"


class TestToEpochMs:
    def setup_method(self):
        self.logger = make_logger()

    def test_float_passthrough(self):
        assert self.logger._to_epoch_ms(1.0) == pytest.approx(1000.0)

    def test_datetime_converted(self):
        dt = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        assert self.logger._to_epoch_ms(dt) == pytest.approx(dt.timestamp() * 1000.0)


class TestGetDuration:
    def setup_method(self):
        self.logger = make_logger()

    def test_uses_kwargs_value_when_present(self):
        kwargs = {"llm_api_duration_ms": 750.0}
        assert self.logger._get_duration(kwargs, 0.0, 1.0) == 750.0

    def test_calculates_from_float_timestamps(self):
        kwargs = {}
        result = self.logger._get_duration(kwargs, 1.0, 2.5)
        assert result == pytest.approx(1500.0)

    def test_calculates_from_datetime_timestamps(self):
        kwargs = {}
        start = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, 0, 0, 1, 500000, tzinfo=timezone.utc)  # +1.5s
        result = self.logger._get_duration(kwargs, start, end)
        assert result == pytest.approx(1500.0)

    def test_returns_none_when_nothing_available(self):
        assert self.logger._get_duration({}, None, None) is None


class TestGetRequestParams:
    def setup_method(self):
        self.logger = make_logger()

    def test_includes_only_present_params(self):
        kwargs = {"optional_params": {"temperature": 0.7}}
        params = self.logger._get_request_params(kwargs)
        assert params == {"temperature": 0.7}
        assert "max_tokens" not in params

    def test_empty_when_no_optional_params(self):
        assert self.logger._get_request_params({}) == {}


# ---------------------------------------------------------------------------
# _process_success — comprehensive happy-path
# ---------------------------------------------------------------------------


class TestProcessSuccess:
    def test_records_summary_and_message_events(self):
        logger = make_logger()
        mock_app = MagicMock()
        mock_app.enabled = True

        kwargs = make_kwargs(
            traceparent="00-aabbccddeeff00112233445566778899-0011223344556677-01",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={"temperature": 0.5, "max_tokens": 100},
        )
        response = make_response(
            response_id="chatcmpl-xyz",
            content="Hi there!",
            finish_reason="stop",
            prompt_tokens=5,
            completion_tokens=10,
        )

        with patch("newrelic.agent.application", return_value=mock_app):
            logger._process_success(kwargs, response, start_time=1.0, end_time=2.5)

        calls = mock_app.record_custom_event.call_args_list
        event_types = [c[0][0] for c in calls]

        assert "LlmChatCompletionSummary" in event_types
        assert "LlmChatCompletionMessage" in event_types

        # Verify summary event fields
        summary_data = next(
            c[0][1] for c in calls if c[0][0] == "LlmChatCompletionSummary"
        )
        assert summary_data["vendor"] == "openai"
        assert summary_data["request.model"] == "gpt-4"
        assert summary_data["response.model"] == "gpt-4"
        assert summary_data["response.choices.finish_reason"] == "stop"
        assert summary_data["response.usage.prompt_tokens"] == 5
        assert summary_data["response.usage.completion_tokens"] == 10
        assert summary_data["response.usage.total_tokens"] == 15
        assert summary_data["request.temperature"] == 0.5
        assert summary_data["request.max_tokens"] == 100
        assert summary_data["ingest_source"] == "litellm"
        assert summary_data["trace_id"] == "aabbccddeeff00112233445566778899"

        # Verify message event id format: "{llm_response_id}-{sequence}"
        message_events = [
            c[0][1] for c in calls if c[0][0] == "LlmChatCompletionMessage"
        ]
        assert any(e["id"].startswith("chatcmpl-xyz-") for e in message_events)
        response_msg = next(e for e in message_events if e.get("is_response"))
        assert response_msg["content"] == "Hi there!"
        assert response_msg["role"] == "assistant"

    def test_skips_when_disabled(self):
        logger = make_logger()
        logger.enabled = False

        with patch("newrelic.agent.application") as mock_app:
            logger._process_success(make_kwargs(), make_response())

        mock_app.assert_not_called()


# ---------------------------------------------------------------------------
# _record_error_metric
# ---------------------------------------------------------------------------


class TestRecordErrorMetric:
    def setup_method(self):
        self.logger = make_logger()

    def test_calls_record_custom_metric(self):
        mock_app = MagicMock()
        mock_app.enabled = True

        with patch.object(self.logger, "_check_and_emit_periodic_metric"):
            with patch("newrelic.agent.application", return_value=mock_app):
                self.logger._record_error_metric()

        mock_app.record_custom_metric.assert_called_once_with("LLM/LiteLLM/Error", 1)

    def test_skips_when_app_disabled(self):
        mock_app = MagicMock()
        mock_app.enabled = False

        with patch.object(self.logger, "_check_and_emit_periodic_metric"):
            with patch("newrelic.agent.application", return_value=mock_app):
                self.logger._record_error_metric()

        mock_app.record_custom_metric.assert_not_called()

    def test_calls_check_and_emit_periodic_metric(self):
        with patch.object(
            self.logger, "_check_and_emit_periodic_metric"
        ) as mock_periodic:
            with patch("newrelic.agent.application", return_value=MagicMock()):
                self.logger._record_error_metric()

        mock_periodic.assert_called_once()

    def test_skips_when_logger_disabled(self):
        self.logger.enabled = False
        with patch("newrelic.agent.application") as mock_app:
            self.logger._record_error_metric()
        mock_app.assert_not_called()

    def test_handles_exception(self):
        with patch(
            "newrelic.agent.application", side_effect=RuntimeError("agent down")
        ):
            self.logger._record_error_metric()  # must not raise


# ---------------------------------------------------------------------------
# _emit_supportability_metric
# ---------------------------------------------------------------------------


class TestEmitSupportabilityMetric:
    def setup_method(self):
        self.logger = make_logger()
        NewRelicLogger._last_metric_emission_time = 0.0

    def test_records_metric_with_correct_name_and_value(self):
        mock_app = MagicMock()
        mock_app.enabled = True
        with patch("newrelic.agent.application", return_value=mock_app):
            with patch.object(
                self.logger, "_get_litellm_version", return_value="1.80.0"
            ):
                self.logger._emit_supportability_metric()
        mock_app.record_custom_metric.assert_called_once_with(
            "Supportability/Python/ML/LiteLLM/1.80.0", 1
        )

    def test_updates_last_emission_time(self):
        mock_app = MagicMock()
        mock_app.enabled = True
        fake_now = 9_999_999.0
        with patch("newrelic.agent.application", return_value=mock_app):
            with patch(
                "litellm.integrations.newrelic.newrelic.time.time",
                return_value=fake_now,
            ):
                self.logger._emit_supportability_metric()
        assert NewRelicLogger._last_metric_emission_time == fake_now

    def test_skips_when_app_disabled(self):
        mock_app = MagicMock()
        mock_app.enabled = False
        with patch("newrelic.agent.application", return_value=mock_app):
            self.logger._emit_supportability_metric()
        mock_app.record_custom_metric.assert_not_called()
        # Timestamp is still updated to back off lock contention during registration.
        assert NewRelicLogger._last_metric_emission_time != 0.0

    def test_skips_when_no_app(self):
        with patch("newrelic.agent.application", return_value=None):
            self.logger._emit_supportability_metric()
        # Timestamp is updated even when app is None to back off lock contention
        # if the agent never starts or is slow to initialise.
        assert NewRelicLogger._last_metric_emission_time != 0.0

    def test_handles_exception(self):
        with patch(
            "newrelic.agent.application", side_effect=RuntimeError("agent down")
        ):
            self.logger._emit_supportability_metric()  # must not raise


# ---------------------------------------------------------------------------
# _check_and_emit_periodic_metric
# ---------------------------------------------------------------------------


class TestCheckAndEmitPeriodicMetric:
    def setup_method(self):
        self.logger = make_logger()
        NewRelicLogger._last_metric_emission_time = 0.0

    def test_emits_on_first_call(self):
        """_last_metric_emission_time starts at 0.0; any real time satisfies 27-hour window."""
        with patch.object(self.logger, "_emit_supportability_metric") as mock_emit:
            with patch(
                "litellm.integrations.newrelic.newrelic.time.time",
                return_value=100_000.0,
            ):
                self.logger._check_and_emit_periodic_metric()
        mock_emit.assert_called_once()

    def test_does_not_re_emit_within_27_hours(self):
        recent = 1_000_000.0
        NewRelicLogger._last_metric_emission_time = recent
        with patch.object(self.logger, "_emit_supportability_metric") as mock_emit:
            with patch(
                "litellm.integrations.newrelic.newrelic.time.time",
                return_value=recent + 3600,  # 1 hour later
            ):
                self.logger._check_and_emit_periodic_metric()
        mock_emit.assert_not_called()

    def test_re_emits_after_27_hours(self):
        old = 1_000_000.0
        NewRelicLogger._last_metric_emission_time = old
        with patch.object(self.logger, "_emit_supportability_metric") as mock_emit:
            with patch(
                "litellm.integrations.newrelic.newrelic.time.time",
                return_value=old + 97201,  # 27 hours + 1 second
            ):
                self.logger._check_and_emit_periodic_metric()
        mock_emit.assert_called_once()

    def test_boundary_exactly_27_hours_triggers_emission(self):
        old = 1_000_000.0
        NewRelicLogger._last_metric_emission_time = old
        with patch.object(self.logger, "_emit_supportability_metric") as mock_emit:
            with patch(
                "litellm.integrations.newrelic.newrelic.time.time",
                return_value=old + 97200,
            ):
                self.logger._check_and_emit_periodic_metric()
        mock_emit.assert_called_once()


# ---------------------------------------------------------------------------
# _get_litellm_version
# ---------------------------------------------------------------------------


class TestGetLitellmVersion:
    def setup_method(self):
        self.logger = make_logger()

    def test_returns_unknown_on_exception(self):
        with patch("importlib.metadata.version", side_effect=Exception("no package")):
            result = self.logger._get_litellm_version()
        assert result == "unknown"


# ---------------------------------------------------------------------------
# _record_summary_event — disabled-app and exception paths
# ---------------------------------------------------------------------------

_USAGE = {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}


class TestRecordSummaryEvent:
    def setup_method(self):
        self.logger = make_logger()

    def _call(self, **kwargs):
        self.logger._record_summary_event(
            request_id="req-1",
            trace_id="trace-abc",
            request_model="gpt-4",
            response_model="gpt-4",
            vendor="openai",
            finish_reason="stop",
            num_messages=2,
            usage=_USAGE,
            **kwargs,
        )

    def test_skips_when_app_disabled(self):
        mock_app = MagicMock()
        mock_app.enabled = False
        with patch("newrelic.agent.application", return_value=mock_app):
            self._call()
        mock_app.record_custom_event.assert_not_called()

    def test_handles_exception(self):
        with patch(
            "newrelic.agent.application", side_effect=RuntimeError("agent down")
        ):
            self._call()  # must not raise


# ---------------------------------------------------------------------------
# _record_message_events — disabled-app and exception paths
# ---------------------------------------------------------------------------

_MESSAGES = [
    {"role": "user", "sequence": 0, "response.model": "gpt-4", "vendor": "openai"}
]


class TestRecordMessageEvents:
    def setup_method(self):
        self.logger = make_logger()

    def _call(self):
        self.logger._record_message_events(
            request_id="req-1",
            llm_response_id="resp-1",
            trace_id="trace-abc",
            messages=_MESSAGES,
        )

    def test_skips_when_app_disabled(self):
        mock_app = MagicMock()
        mock_app.enabled = False
        with patch("newrelic.agent.application", return_value=mock_app):
            self._call()
        mock_app.record_custom_event.assert_not_called()

    def test_handles_exception(self):
        with patch(
            "newrelic.agent.application", side_effect=RuntimeError("agent down")
        ):
            self._call()  # must not raise


# ---------------------------------------------------------------------------
# CustomLogger interface entry points
# ---------------------------------------------------------------------------


class TestLogSuccessEvent:
    def test_delegates_to_process_success(self):
        logger = make_logger()
        with patch.object(logger, "_process_success") as mock_process:
            logger.log_success_event(make_kwargs(), make_response(), 1.0, 2.0)
        mock_process.assert_called_once()

    def test_exception_is_handled(self):
        logger = make_logger()
        with patch.object(logger, "_process_success", side_effect=RuntimeError("boom")):
            logger.log_success_event(make_kwargs(), make_response(), 1.0, 2.0)

    @pytest.mark.asyncio
    async def test_async_delegates_to_process_success(self):
        logger = make_logger()
        with patch.object(logger, "_process_success") as mock_process:
            await logger.async_log_success_event(
                make_kwargs(), make_response(), 1.0, 2.0
            )
        mock_process.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_exception_is_handled(self):
        logger = make_logger()
        with patch.object(logger, "_process_success", side_effect=RuntimeError("boom")):
            await logger.async_log_success_event(
                make_kwargs(), make_response(), 1.0, 2.0
            )


class TestLogFailureEvent:
    def test_sync_records_error_metric(self):
        logger = make_logger()
        with patch.object(logger, "_record_error_metric") as mock_metric:
            logger.log_failure_event(make_kwargs(), None, 1.0, 2.0)
        mock_metric.assert_called_once()

    def test_sync_exception_is_handled(self):
        logger = make_logger()
        with patch.object(
            logger, "_record_error_metric", side_effect=RuntimeError("boom")
        ):
            logger.log_failure_event(make_kwargs(), None, 1.0, 2.0)

    @pytest.mark.asyncio
    async def test_async_records_error_metric(self):
        logger = make_logger()
        with patch.object(logger, "_record_error_metric") as mock_metric:
            await logger.async_log_failure_event(make_kwargs(), None, 1.0, 2.0)
        mock_metric.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_exception_is_handled(self):
        logger = make_logger()
        with patch.object(
            logger, "_record_error_metric", side_effect=RuntimeError("boom")
        ):
            await logger.async_log_failure_event(make_kwargs(), None, 1.0, 2.0)


# ---------------------------------------------------------------------------
# async_health_check
# ---------------------------------------------------------------------------


class TestAsyncHealthCheck:
    @pytest.mark.asyncio
    async def test_unhealthy_when_disabled(self):
        logger = make_logger()
        logger.enabled = False
        result = await logger.async_health_check()
        assert result["status"] == "unhealthy"
        assert result["error_message"] is not None

    @pytest.mark.asyncio
    async def test_healthy_when_app_enabled(self):
        logger = make_logger()
        mock_app = MagicMock()
        mock_app.enabled = True
        with patch("newrelic.agent.application", return_value=mock_app):
            result = await logger.async_health_check()
        assert result["status"] == "healthy"
        assert result["error_message"] is None

    @pytest.mark.asyncio
    async def test_unhealthy_when_app_disabled(self):
        logger = make_logger()
        mock_app = MagicMock()
        mock_app.enabled = False
        with patch("newrelic.agent.application", return_value=mock_app):
            result = await logger.async_health_check()
        assert result["status"] == "unhealthy"
        assert result["error_message"] is not None

    @pytest.mark.asyncio
    async def test_exception_returns_unhealthy(self):
        logger = make_logger()
        with patch(
            "newrelic.agent.application", side_effect=RuntimeError("agent down")
        ):
            result = await logger.async_health_check()
        assert result["status"] == "unhealthy"
        assert "agent down" in result["error_message"]


# ---------------------------------------------------------------------------
# _extract_completion_id fallback chain
# ---------------------------------------------------------------------------


class TestExtractCompletionId:
    def setup_method(self):
        self.logger = make_logger()

    def test_uses_litellm_call_id_when_response_has_no_id(self):
        result = self.logger._extract_completion_id(
            kwargs={"litellm_call_id": "call-abc-123"},
            response_obj={},
        )
        assert result == "call-abc-123"

    def test_generates_uuid_when_neither_id_present(self):
        result = self.logger._extract_completion_id(kwargs={}, response_obj={})
        # UUID4 hex-with-dashes is 36 chars; just confirm shape and uniqueness
        assert isinstance(result, str)
        assert len(result) == 36
        second = self.logger._extract_completion_id(kwargs={}, response_obj={})
        assert result != second


# ---------------------------------------------------------------------------
# StandardLoggingPayload preference across extractors
# ---------------------------------------------------------------------------


class TestStandardLoggingPayloadPreference:
    """Each extractor that accepts a StandardLoggingPayload must prefer its
    values over the raw kwargs/response fallbacks."""

    def setup_method(self):
        self.logger = make_logger()

    def test_trace_context_uses_slo_trace_id_when_no_traceparent(self):
        kwargs = {"litellm_params": {"metadata": {"headers": {}}}}
        trace_id = self.logger._get_trace_context(
            kwargs, standard_logging_object=make_slo()
        )
        assert trace_id == "slo-trace-abc"

    def test_vendor_from_slo(self):
        # kwargs carries a different provider; SLO must win.
        kwargs = {"litellm_params": {"custom_llm_provider": "kwargs-provider"}}
        assert (
            self.logger._get_vendor(kwargs, standard_logging_object=make_slo())
            == "slo-provider"
        )

    def test_model_names_uses_slo_model(self):
        request_model, _ = self.logger._get_model_names(
            {"model": "kwargs-model"},
            make_response(model="response-model"),
            standard_logging_object=make_slo(),
        )
        assert request_model == "slo-model"

    def test_usage_from_slo_when_any_token_field_present(self):
        # make_response defaults to 10/20/30 tokens; SLO sentinels are 100/200/300.
        usage = self.logger._extract_usage(
            make_response(), standard_logging_object=make_slo()
        )
        assert usage == {
            "prompt_tokens": 100,
            "completion_tokens": 200,
            "total_tokens": 300,
        }

    def test_duration_from_slo_response_time_converted_to_ms(self):
        # SLO response_time is 1.5 seconds; expected 1500.0 ms.
        # Pass start/end that would compute a different value to prove SLO won.
        duration = self.logger._get_duration(
            kwargs={"llm_api_duration_ms": 9999.0},
            start_time=1.0,
            end_time=2.0,
            standard_logging_object=make_slo(),
        )
        assert duration == 1500.0

    def test_request_params_from_slo_model_parameters(self):
        params = self.logger._get_request_params(
            {"optional_params": {"temperature": 0.1}},
            standard_logging_object=make_slo(),
        )
        assert params == {"temperature": 0.7, "max_tokens": 500}

    def test_extract_all_messages_sources_timestamps_and_messages_from_slo(self):
        """Covers three SLO branches at once: startTime, endTime, and messages list."""
        kwargs = make_kwargs(messages=[{"role": "user", "content": "from-kwargs"}])
        messages = self.logger._extract_all_messages(
            kwargs,
            make_response(),
            response_model="gpt-4",
            vendor="openai",
            standard_logging_object=make_slo(),
        )

        request = next(m for m in messages if not m.get("is_response"))
        assert request["content"] == "from-slo"  # SLO messages list wins
        assert request["timestamp"] == int(2_000_000.0 * 1000.0)  # SLO startTime

        response = next(m for m in messages if m.get("is_response"))
        assert response["timestamp"] == int(2_000_001.5 * 1000.0)  # SLO endTime
