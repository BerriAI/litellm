import os
from copy import deepcopy
from datetime import datetime
from typing import Final
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import respx
from pydantic import TypeAdapter

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.generic_api.generic_api_callback import GenericAPILogger
from litellm.integrations.langfuse.langfuse_prompt_management import LangfusePromptManagement
from litellm.integrations.opentelemetry import OpenTelemetry
from litellm.litellm_core_utils.logging_callback_manager import LoggingCallbackManager
from litellm.litellm_core_utils import logging_callback_manager as callback_manager_module


@pytest.fixture
def callback_manager():
    manager = LoggingCallbackManager()

    manager._reset_all_callbacks()
    return manager


@pytest.fixture
def mock_custom_logger():
    class TestLogger(CustomLogger):
        def log_success_event(self, kwargs, response_obj, start_time, end_time):
            pass

    return TestLogger()


def test_add_string_callback():
    """
    Test adding a string callback to litellm.callbacks - only 1 instance of the string callback should be added
    """
    manager = LoggingCallbackManager()
    test_callback = "test_callback"

    manager.add_litellm_callback(test_callback)
    assert test_callback in litellm.callbacks

    manager.add_litellm_callback(test_callback)
    assert litellm.callbacks.count(test_callback) == 1


def test_add_function_callback():
    manager = LoggingCallbackManager()

    def test_func(kwargs):
        pass

    manager.add_litellm_callback(test_func)
    assert test_func in litellm.callbacks

    manager.add_litellm_callback(test_func)
    assert litellm.callbacks.count(test_func) == 1


def test_add_custom_logger(mock_custom_logger):
    manager = LoggingCallbackManager()

    manager.add_litellm_callback(mock_custom_logger)
    assert mock_custom_logger in litellm.callbacks


def test_add_multiple_callback_types(mock_custom_logger):
    manager = LoggingCallbackManager()

    def test_func(kwargs):
        pass

    string_callback = "test_callback"

    manager.add_litellm_callback(string_callback)
    manager.add_litellm_callback(test_func)
    manager.add_litellm_callback(mock_custom_logger)

    assert string_callback in litellm.callbacks
    assert test_func in litellm.callbacks
    assert mock_custom_logger in litellm.callbacks
    assert len(litellm.callbacks) == 3


def test_success_failure_callbacks():
    manager = LoggingCallbackManager()

    success_callback = "success_callback"
    failure_callback = "failure_callback"

    manager.add_litellm_success_callback(success_callback)
    manager.add_litellm_failure_callback(failure_callback)

    assert success_callback in litellm.success_callback
    assert failure_callback in litellm.failure_callback


def test_async_callbacks():
    manager = LoggingCallbackManager()

    async_success = "async_success"
    async_failure = "async_failure"

    manager.add_litellm_async_success_callback(async_success)
    manager.add_litellm_async_failure_callback(async_failure)

    assert async_success in litellm._async_success_callback
    assert async_failure in litellm._async_failure_callback


def test_remove_callback_from_list_by_object():
    manager = LoggingCallbackManager()

    manager._reset_all_callbacks()

    def TestObject():
        def __init__(self):
            manager.add_litellm_callback(self.callback)
            manager.add_litellm_success_callback(self.callback)
            manager.add_litellm_failure_callback(self.callback)
            manager.add_litellm_async_success_callback(self.callback)
            manager.add_litellm_async_failure_callback(self.callback)

        def callback(self):
            pass

    obj = TestObject()

    manager.remove_callback_from_list_by_object(litellm.callbacks, obj)
    manager.remove_callback_from_list_by_object(litellm.success_callback, obj)
    manager.remove_callback_from_list_by_object(litellm.failure_callback, obj)
    manager.remove_callback_from_list_by_object(litellm._async_success_callback, obj)
    manager.remove_callback_from_list_by_object(litellm._async_failure_callback, obj)

    assert len(litellm.callbacks) == 0
    assert len(litellm.success_callback) == 0
    assert len(litellm.failure_callback) == 0
    assert len(litellm._async_success_callback) == 0
    assert len(litellm._async_failure_callback) == 0


def test_remove_callback_from_all_lists():
    manager = LoggingCallbackManager()
    manager._reset_all_callbacks()

    class TestLogger(CustomLogger):
        pass

    obj = TestLogger()
    manager.add_litellm_callback(obj)
    manager.add_litellm_success_callback(obj)
    manager.add_litellm_failure_callback(obj)
    manager.add_litellm_async_success_callback(obj)
    manager.add_litellm_async_failure_callback(obj)

    manager.remove_callback_from_all_lists(obj)

    assert obj not in litellm.callbacks
    assert obj not in litellm.success_callback
    assert obj not in litellm.failure_callback
    assert obj not in litellm._async_success_callback
    assert obj not in litellm._async_failure_callback


def test_reset_callbacks(callback_manager):

    callback_manager.add_litellm_callback("test")
    callback_manager.add_litellm_success_callback("success")
    callback_manager.add_litellm_failure_callback("failure")
    callback_manager.add_litellm_async_success_callback("async_success")
    callback_manager.add_litellm_async_failure_callback("async_failure")

    callback_manager._reset_all_callbacks()

    assert len(litellm.callbacks) == 0
    assert len(litellm.success_callback) == 0
    assert len(litellm.failure_callback) == 0
    assert len(litellm._async_success_callback) == 0
    assert len(litellm._async_failure_callback) == 0


@pytest.mark.asyncio
async def test_slack_alerting_callback_registration(callback_manager):
    """
    Test that litellm callbacks are correctly registered for slack alerting
    when outage_alerts or region_outage_alerts are enabled
    """
    from litellm.caching.caching import DualCache
    from litellm.proxy.utils import ProxyLogging
    from litellm.integrations.SlackAlerting.slack_alerting import SlackAlerting
    from unittest.mock import patch

    with patch("litellm.integrations.SlackAlerting.slack_alerting.get_async_httpx_client") as mock_http:
        mock_http.return_value = AsyncMock()

        proxy_logging = ProxyLogging(user_api_key_cache=DualCache())

        proxy_logging.update_values(alerting=None, alert_types=["outage_alerts", "region_outage_alerts"])
        assert len(litellm.callbacks) == 0

        proxy_logging.update_values(alerting=["slack"], alert_types=["outage_alerts"])
        assert len(litellm.callbacks) == 1
        assert isinstance(litellm.callbacks[0], SlackAlerting)

        callback_manager._reset_all_callbacks()
        proxy_logging.update_values(alerting=["slack"], alert_types=["region_outage_alerts"])
        assert len(litellm.callbacks) == 1
        assert isinstance(litellm.callbacks[0], SlackAlerting)

        callback_manager._reset_all_callbacks()
        proxy_logging.update_values(alerting=["slack"], alert_types=["budget_alerts"])
        assert len(litellm.callbacks) == 0

        callback_manager._reset_all_callbacks()
        proxy_logging.update_values(alerting=["slack"], alert_types=["outage_alerts"])
        assert len(litellm.callbacks) == 1
        assert isinstance(litellm.callbacks[0], SlackAlerting)

        response_taking_too_long_callback = proxy_logging.slack_alerting_instance.response_taking_too_long_callback
        assert len(litellm._async_success_callback) == 1
        assert litellm._async_success_callback[0] == response_taking_too_long_callback

        callback_manager._reset_all_callbacks()


@pytest.mark.asyncio
async def test_generic_api_compatible_callbacks_json():
    """
    Test that callbacks defined in generic_api_compatible_callbacks.json
    are properly loaded and initialized by _add_custom_callback_generic_api_str
    """
    from litellm.integrations.generic_api.generic_api_callback import GenericAPILogger

    test_sumologic_url = "https://collectors.sumologic.com/receiver/v1/http/test123"

    with patch.dict(os.environ, {"SUMOLOGIC_WEBHOOK_URL": test_sumologic_url}):
        result = LoggingCallbackManager.add_custom_callback_generic_api_str("sumologic")

        assert isinstance(result, GenericAPILogger), "Should return GenericAPILogger instance for sumologic callback"

        assert result.endpoint == test_sumologic_url, f"Endpoint should be {test_sumologic_url}"

        assert "Content-Type" in result.headers, "Should have Content-Type header"
        assert result.headers["Content-Type"] == "application/json", "Content-Type should be application/json"
        assert "Authorization" not in result.headers, "Should not have Authorization header for SumoLogic"


@pytest.mark.asyncio
async def test_generic_api_compatible_callbacks_json_rubrik():
    """
    Test the rubrik callback from generic_api_compatible_callbacks.json
    which requires both API key and webhook URL
    """
    from litellm.integrations.generic_api.generic_api_callback import GenericAPILogger

    test_rubrik_url = "https://webhook.site/test-rubrik"
    test_rubrik_api_key = "sk-rubrik-test-key"

    with patch.dict(
        os.environ,
        {"RUBRIK_WEBHOOK_URL": test_rubrik_url, "RUBRIK_API_KEY": test_rubrik_api_key},
    ):
        result = LoggingCallbackManager.add_custom_callback_generic_api_str("rubrik")

        assert isinstance(result, GenericAPILogger), "Should return GenericAPILogger instance for rubrik callback"

        assert result.endpoint == test_rubrik_url, f"Endpoint should be {test_rubrik_url}"

        assert "Content-Type" in result.headers, "Should have Content-Type header"
        assert "Authorization" in result.headers, "Should have Authorization header for Rubrik"
        assert result.headers["Authorization"] == f"Bearer {test_rubrik_api_key}", (
            "Authorization should have correct API key"
        )

        assert result.event_types == ["llm_api_success"], "Rubrik should only log success events"


def test_generic_api_compatible_callbacks_json_unknown_callback():
    """
    Test that unknown callbacks (not in JSON or callback_settings) are returned unchanged
    """

    result = LoggingCallbackManager.add_custom_callback_generic_api_str("unknown_callback")

    assert result == "unknown_callback", "Unknown callback should be returned as-is"
    assert isinstance(result, str), "Unknown callback should remain a string"


@pytest.mark.asyncio
async def test_generic_api_callback_settings_retry_config():
    """
    Test that generic_api callback_settings are passed to GenericAPILogger.
    """
    from litellm.integrations.generic_api.generic_api_callback import GenericAPILogger
    from litellm.litellm_core_utils.logging_callback_manager import (
        _generic_api_logger_cache,
    )

    callback_name = "test_generic_api_retry_config"
    _generic_api_logger_cache.pop(callback_name, None)
    litellm.callback_settings[callback_name] = {
        "callback_type": "generic_api",
        "endpoint": "https://example.com/api/logs",
        "headers": {"Content-Type": "application/json"},
        "max_retries": 2,
        "retry_delay": 0.5,
        "timeout": 3,
    }

    try:
        result = LoggingCallbackManager.add_custom_callback_generic_api_str(callback_name)

        assert isinstance(result, GenericAPILogger)
        assert result.endpoint == "https://example.com/api/logs"
        assert result.headers == {"Content-Type": "application/json"}
        assert result.max_retries == 2
        assert result.retry_delay == 0.5
        assert result.timeout == 3
    finally:
        litellm.callback_settings.pop(callback_name, None)
        _generic_api_logger_cache.pop(callback_name, None)


def test_duplicate_langfuse_logger_test():
    manager = LoggingCallbackManager()
    for _ in range(10):
        langfuse_logger = LangfusePromptManagement()
        manager.add_litellm_success_callback(langfuse_logger)
    assert len(litellm.success_callback) == 1


def test_duplicate_multiple_loggers_test():
    manager = LoggingCallbackManager()
    for _ in range(10):
        langfuse_logger = LangfusePromptManagement()
        otel_logger = OpenTelemetry()
        manager.add_litellm_success_callback(langfuse_logger)
        manager.add_litellm_success_callback(otel_logger)
    assert len(litellm.success_callback) == 2

    langfuse_count = sum(1 for callback in litellm.success_callback if isinstance(callback, LangfusePromptManagement))
    otel_count = sum(1 for callback in litellm.success_callback if isinstance(callback, OpenTelemetry))

    assert langfuse_count == 1, "Should have exactly one LangfusePromptManagement instance"
    assert otel_count == 1, "Should have exactly one OpenTelemetry instance"


@pytest.fixture
def generic_api_redaction_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(callback_manager_module, "_generic_api_logger_cache", {})
    monkeypatch.setattr(litellm, "callback_settings", {})
    monkeypatch.setattr(litellm, "turn_off_message_logging", False)
    monkeypatch.setattr(litellm, "generic_api_use_v1", False)
    monkeypatch.setattr(litellm, "standard_logging_payload_excluded_fields", None)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setenv("RAMP_API_KEY", "test-ramp-key")


async def _assert_generic_api_export(logger: GenericAPILogger, redact: bool) -> None:
    payload: Final = {
        "messages": [{"role": "user", "content": "private-prompt-canary"}],
        "response": "private-response-canary",
        "model": "accounting-model",
        "prompt_tokens": 20,
        "completion_tokens": 5,
        "total_tokens": 25,
        "response_cost": 0.03,
    }
    original: Final = deepcopy(payload)
    full_logger: Final = CustomLogger()
    full_details: Final = {"standard_logging_object": payload}
    ramp_details: Final = logger.redact_standard_logging_payload_from_model_call_details(full_details)
    now: Final = datetime(2026, 1, 1)
    with respx.mock as http:
        destination: Final = http.post(logger.endpoint).mock(return_value=httpx.Response(200))
        await logger.async_log_success_event(ramp_details, None, now, now)
        await logger.flush_queue()
        assert destination.call_count == 1
        request_body: Final = destination.calls.last.request.content
        exported: Final = TypeAdapter(list[dict[str, object]]).validate_json(request_body)

    assert len(exported) == 1
    assert exported[0]["model"] == payload["model"]
    assert exported[0]["prompt_tokens"] == payload["prompt_tokens"]
    assert exported[0]["completion_tokens"] == payload["completion_tokens"]
    assert exported[0]["total_tokens"] == payload["total_tokens"]
    assert exported[0]["response_cost"] == payload["response_cost"]
    assert (b"private-prompt-canary" not in request_body) is redact
    assert (b"private-response-canary" not in request_body) is redact
    assert full_logger.redact_standard_logging_payload_from_model_call_details(full_details) == full_details
    assert payload == original


@pytest.mark.asyncio
@pytest.mark.parametrize("callback_name", ["ramp", "accounting_sink"])
@pytest.mark.parametrize("redaction", [None, False, True])
async def test_generic_api_redaction_keeps_original_logging_payload(
    callback_name: str,
    redaction: bool | None,
    generic_api_redaction_state: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings: Final = (
        {
            "callback_type": "generic_api",
            "endpoint": "https://example.com/logs",
            "headers": {"Content-Type": "application/json"},
            "log_format": "json_array",
        }
        if callback_name == "accounting_sink"
        else {}
    )
    redaction_setting: Final = {} if redaction is None else {"turn_off_message_logging": redaction}
    monkeypatch.setattr(litellm, "callback_settings", {callback_name: {**settings, **redaction_setting}})
    logger: Final = LoggingCallbackManager.add_custom_callback_generic_api_str(callback_name)
    assert isinstance(logger, GenericAPILogger)
    await _assert_generic_api_export(logger, redact=redaction is True)


@pytest.mark.asyncio
@pytest.mark.parametrize("callback_name", ["ramp", "accounting_sink"])
@pytest.mark.parametrize("before,after", [(False, True), (True, False), (True, True)])
async def test_generic_api_cached_logger_tracks_redaction_changes(
    callback_name: str,
    before: bool,
    after: bool,
    generic_api_redaction_state: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings: Final = (
        {
            "callback_type": "generic_api",
            "endpoint": "https://example.com/logs",
            "headers": {"Content-Type": "application/json"},
            "log_format": "json_array",
        }
        if callback_name == "accounting_sink"
        else {}
    )
    monkeypatch.setattr(litellm, "callback_settings", {callback_name: {**settings, "turn_off_message_logging": before}})
    initial: Final = LoggingCallbackManager.add_custom_callback_generic_api_str(callback_name)
    assert isinstance(initial, GenericAPILogger)
    await _assert_generic_api_export(initial, redact=before)

    monkeypatch.setattr(litellm, "callback_settings", {callback_name: {**settings, "turn_off_message_logging": after}})
    updated: Final = LoggingCallbackManager.add_custom_callback_generic_api_str(callback_name)
    assert isinstance(updated, GenericAPILogger)
    await _assert_generic_api_export(updated, redact=after)
    assert (updated is initial) is (before == after)
