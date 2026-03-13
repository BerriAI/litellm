import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# newrelic is a container-only dependency (requirements.txt) and is not installed
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

sys.path.insert(0, os.path.abspath("../.."))

from litellm.integrations.newrelic.newrelic import NewRelicLogger


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


# ---------------------------------------------------------------------------
# 1. Init / configuration
# ---------------------------------------------------------------------------


class TestNewRelicLoggerInit:
    def test_disabled_when_license_key_missing(self):
        with patch("newrelic.agent.register_application"):
            with patch.dict(os.environ, {"NEW_RELIC_APP_NAME": "app"}, clear=True):
                logger = NewRelicLogger()
        assert logger.enabled is False

    def test_disabled_when_app_name_missing(self):
        with patch("newrelic.agent.register_application"):
            with patch.dict(
                os.environ, {"NEW_RELIC_LICENSE_KEY": "key"}, clear=True
            ):
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


# ---------------------------------------------------------------------------
# 2. _parse_bool_env
# ---------------------------------------------------------------------------


class TestParseBoolEnv:
    def setup_method(self):
        self.logger = make_logger()

    def test_true_string(self):
        with patch.dict(os.environ, {"MY_VAR": "true"}):
            assert self.logger._parse_bool_env("MY_VAR") is True

    def test_true_uppercase(self):
        with patch.dict(os.environ, {"MY_VAR": "TRUE"}):
            assert self.logger._parse_bool_env("MY_VAR") is True

    def test_false_string(self):
        with patch.dict(os.environ, {"MY_VAR": "false"}):
            assert self.logger._parse_bool_env("MY_VAR") is False

    def test_missing_uses_default(self):
        with patch.dict(os.environ, {}, clear=True):
            assert self.logger._parse_bool_env("MY_VAR", default=True) is True
            assert self.logger._parse_bool_env("MY_VAR", default=False) is False


# ---------------------------------------------------------------------------
# 3. _get_trace_context
# ---------------------------------------------------------------------------


class TestGetTraceContext:
    def setup_method(self):
        self.logger = make_logger()

    def test_extracts_trace_id_from_traceparent(self):
        kwargs = make_kwargs(
            traceparent="00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-00"
        )
        trace_id, span_id = self.logger._get_trace_context(kwargs)
        assert trace_id == "4bf92f3577b34da6a3ce929d0e0e4736"

    def test_generates_uuid_when_no_headers(self):
        kwargs = make_kwargs()
        trace_id, span_id = self.logger._get_trace_context(kwargs)
        assert trace_id is not None
        assert len(trace_id) == 36  # UUID format

    def test_generates_uuid_when_traceparent_malformed(self):
        kwargs = make_kwargs(traceparent="not-valid")
        trace_id, span_id = self.logger._get_trace_context(kwargs)
        # Falls back to a generated UUID
        assert trace_id is not None
        assert len(trace_id) == 36


# ---------------------------------------------------------------------------
# 4. _extract_message_content edge cases
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
        msg = {"content": [{"type": "text", "text": "describe this"}, {"type": "image_url"}]}
        result = self.logger._extract_message_content(msg)
        assert "describe this" in result
        assert "image_url" in result


# ---------------------------------------------------------------------------
# 5. _extract_all_messages — record_content=False path
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


# ---------------------------------------------------------------------------
# 6. Helper edge cases
# ---------------------------------------------------------------------------


class TestExtractUsage:
    def setup_method(self):
        self.logger = make_logger()

    def test_missing_usage_returns_zeros(self):
        response = {"id": "r1", "model": "gpt-4", "choices": []}
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


class TestGetDuration:
    def setup_method(self):
        self.logger = make_logger()

    def test_uses_kwargs_value_when_present(self):
        kwargs = {"llm_api_duration_ms": 750.0}
        assert self.logger._get_duration(kwargs, 0.0, 1.0) == 750.0

    def test_calculates_from_timestamps_when_kwarg_absent(self):
        kwargs = {}
        result = self.logger._get_duration(kwargs, 1.0, 2.5)
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
# 7. _process_success — comprehensive happy-path
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
# 8. _record_error_metric
# ---------------------------------------------------------------------------


class TestRecordErrorMetric:
    def test_calls_record_custom_metric(self):
        logger = make_logger()
        mock_app = MagicMock()

        with patch("newrelic.agent.application", return_value=mock_app):
            logger._record_error_metric()

        mock_app.record_custom_metric.assert_called_once_with("LLM/LiteLLM/Error", 1)
