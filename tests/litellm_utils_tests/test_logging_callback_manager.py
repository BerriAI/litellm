import json
import os
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.langfuse.langfuse_prompt_management import (
    LangfusePromptManagement,
)
from litellm.integrations.opentelemetry import OpenTelemetry
from litellm.litellm_core_utils.logging_callback_manager import LoggingCallbackManager


# Test fixtures
@pytest.fixture
def callback_manager():
    manager = LoggingCallbackManager()
    # Reset callbacks before each test
    manager._reset_all_callbacks()
    return manager


@pytest.fixture
def mock_custom_logger():
    class TestLogger(CustomLogger):
        def log_success_event(self, kwargs, response_obj, start_time, end_time):
            pass

    return TestLogger()


# Test cases
def test_dashboard_callback_names_cover_callback_config_catalogue():
    from pathlib import Path

    from litellm.litellm_core_utils.logging_callback_manager import (
        get_dashboard_callback_name,
    )

    callback_config_path = Path(__file__).parents[2] / "litellm/integrations/callback_configs.json"
    with callback_config_path.open() as config_file:
        callback_configs = json.load(config_file)

    assert all(get_dashboard_callback_name(callback_config["id"]) is not None for callback_config in callback_configs)


def test_dashboard_callback_names_cover_dashboard_callback_catalogue():
    from litellm.litellm_core_utils.logging_callback_manager import (
        get_dashboard_callback_name,
    )
    from litellm.proxy._types import AllCallbacks

    callback_names = {callback["litellm_callback_name"] for callback in AllCallbacks().model_dump().values()}

    assert all(get_dashboard_callback_name(callback_name) is not None for callback_name in callback_names)


def test_dashboard_callback_inventory_includes_only_public_callbacks(callback_manager, monkeypatch):
    class UnhashableCustomLogger(CustomLogger):
        __hash__ = None

        def __eq__(self, other):
            return self is other

    def custom_callback(*args, **kwargs):
        pass

    unhashable_callback = UnhashableCustomLogger()
    monkeypatch.setattr(
        litellm,
        "success_callback",
        ["opentelemetry", "s3_v2", "custom_callback_api", unhashable_callback, custom_callback],
    )
    monkeypatch.setattr(litellm, "_async_success_callback", ["langsmith"])
    monkeypatch.setattr(litellm, "failure_callback", ["aws_sqs", "langsmith"])
    monkeypatch.setattr(litellm, "_async_failure_callback", [])
    monkeypatch.setattr(litellm, "callbacks", ["langfuse_otel", "_PROXY_VirtualKeyModelMaxBudgetLimiter"])

    assert callback_manager.get_dashboard_callback_registrations() == (
        ("otel", "success"),
        ("s3", "success"),
        ("generic_api", "success"),
        ("langsmith", "success_and_failure"),
        ("sqs", "failure"),
        ("langfuse_otel", "success_and_failure"),
    )


def test_dashboard_callback_inventory_recognizes_azure_sentinel_and_traceloop(
    callback_manager,
    monkeypatch,
):
    from litellm.integrations.azure_sentinel.azure_sentinel import AzureSentinelLogger

    # azure_sentinel is promoted to a logger object by the factory; traceloop is a
    # legacy string callback the success handler dispatches by name.
    monkeypatch.setattr(litellm, "success_callback", [object.__new__(AzureSentinelLogger), "traceloop"])

    assert callback_manager.get_dashboard_callback_registrations() == (
        ("azure_sentinel", "success"),
        ("traceloop", "success"),
    )


def test_dashboard_callback_inventory_rejects_callback_name_spoof(callback_manager, monkeypatch):
    class SpoofedCallback(CustomLogger):
        callback_name = "s3_v2"

    monkeypatch.setattr(litellm, "success_callback", [SpoofedCallback()])

    assert callback_manager.get_dashboard_callback_registrations() == ()


def test_dashboard_callback_inventory_rejects_spoofed_trusted_subclasses(
    callback_manager,
    monkeypatch,
):
    from litellm.integrations.arize.arize import ArizeLogger
    from litellm.integrations.generic_api.generic_api_callback import (
        GenericAPILogger,
    )
    from litellm.integrations.langfuse.langfuse_otel import LangfuseOtelLogger
    from litellm.integrations.otel.logger import OpenTelemetryV2

    spoofed_callback_types = tuple(
        type(
            callback_name,
            (callback_type,),
            {},
        )
        for callback_name, callback_type in (
            ("SpoofedArizeLogger", ArizeLogger),
            ("SpoofedGenericAPILogger", GenericAPILogger),
            ("SpoofedLangfuseOtelLogger", LangfuseOtelLogger),
            ("SpoofedOpenTelemetry", OpenTelemetry),
            ("SpoofedOpenTelemetryV2", OpenTelemetryV2),
        )
    )
    monkeypatch.setattr(
        litellm, "success_callback", [object.__new__(callback_type) for callback_type in spoofed_callback_types]
    )
    for callback in litellm.success_callback:
        callback.callback_name = "generic_api"

    assert callback_manager.get_dashboard_callback_registrations() == ()


def test_dashboard_callback_inventory_rejects_unregistered_otel_callback(
    callback_manager,
    monkeypatch,
):
    otel_callback = object.__new__(OpenTelemetry)
    otel_callback.callback_name = "langfuse"
    monkeypatch.setattr(litellm, "success_callback", [otel_callback])

    assert callback_manager.get_dashboard_callback_registrations() == ()


@pytest.mark.asyncio
async def test_dashboard_callback_inventory_recognizes_supported_logger_types(
    callback_manager,
    monkeypatch,
):
    from litellm.integrations.datadog.datadog_cost_management import (
        DatadogCostManagementLogger,
    )
    from litellm.integrations.datadog.datadog_metrics import DatadogMetricsLogger
    from litellm.integrations.lago import LagoLogger
    from litellm.integrations.langfuse.langfuse_prompt_management import (
        LangfusePromptManagement,
    )
    from litellm.integrations.langsmith import LangsmithLogger
    from litellm.integrations.openmeter import OpenMeterLogger
    from litellm.integrations.s3_v2 import S3Logger
    from litellm.integrations.sqs import SQSLogger

    monkeypatch.setenv("LAGO_API_KEY", "test-key")
    monkeypatch.setenv("LAGO_API_BASE", "https://example.com")
    monkeypatch.setenv("LAGO_API_EVENT_CODE", "test-event")
    monkeypatch.setenv("OPENMETER_API_KEY", "test-key")
    monkeypatch.setattr(
        litellm,
        "success_callback",
        [
            LagoLogger(),
            OpenMeterLogger(),
            LangfusePromptManagement(),
            LangsmithLogger(),
            DatadogMetricsLogger(),
            DatadogCostManagementLogger(),
            S3Logger(),
            SQSLogger(),
        ],
    )

    assert callback_manager.get_dashboard_callback_registrations() == (
        ("lago", "success"),
        ("openmeter", "success"),
        ("langfuse", "success"),
        ("langsmith", "success"),
        ("datadog_metrics", "success"),
        ("datadog_cost_management", "success"),
        ("s3", "success"),
        ("sqs", "success"),
    )


@pytest.mark.asyncio
async def test_dashboard_callback_inventory_includes_builtin_generic_api_callback(
    callback_manager,
    monkeypatch,
):
    from litellm.litellm_core_utils import litellm_logging as logging_module

    monkeypatch.setenv("GENERIC_LOGGER_ENDPOINT", "https://example.com/logs")
    logging_module._in_memory_loggers.clear()
    try:
        generic_api_callback = logging_module._init_custom_logger_compatible_class(
            logging_integration="generic_api",
            internal_usage_cache=None,
            llm_router=None,
            custom_logger_init_args={},
        )

        assert generic_api_callback is not None
        assert generic_api_callback.callback_name is None

        monkeypatch.setattr(litellm, "success_callback", [generic_api_callback])

        assert callback_manager.get_dashboard_callback_registrations() == (("generic_api", "success"),)
    finally:
        logging_module._in_memory_loggers.clear()


@pytest.mark.asyncio
@pytest.mark.parametrize("callback_name", [None, "logfire"])
async def test_dashboard_callback_inventory_includes_factory_created_otel_callback(
    callback_manager,
    monkeypatch,
    callback_name,
):
    from litellm.litellm_core_utils import litellm_logging as logging_module

    monkeypatch.setattr(
        litellm,
        "callback_settings",
        {"otel": {"callback_name": callback_name}},
    )
    logging_module._in_memory_loggers.clear()
    try:
        otel_callback = logging_module._init_custom_logger_compatible_class(
            logging_integration="otel",
            internal_usage_cache=None,
            llm_router=None,
            custom_logger_init_args={},
        )
        assert otel_callback is not None
        assert otel_callback.callback_name == callback_name
        monkeypatch.setattr(litellm, "success_callback", [otel_callback])

        assert callback_manager.get_dashboard_callback_registrations() == (("otel", "success"),)
    finally:
        logging_module._in_memory_loggers.clear()


@pytest.mark.asyncio
async def test_dashboard_otel_provenance_keeps_legacy_otel_separate_from_preset(
    callback_manager,
    monkeypatch,
):
    from litellm.integrations.otel.model.config import is_otel_v2_enabled
    from litellm.litellm_core_utils import litellm_logging as logging_module

    monkeypatch.setenv("LANGTRACE_API_KEY", "test-key")
    monkeypatch.setenv("LITELLM_OTEL_V2", "false")
    is_otel_v2_enabled.cache_clear()
    logging_module._in_memory_loggers.clear()
    try:
        langtrace = logging_module._init_custom_logger_compatible_class("langtrace", None, None, {})
        otel = logging_module._init_custom_logger_compatible_class("otel", None, None, {})

        assert otel is not langtrace
        monkeypatch.setattr(litellm, "success_callback", [otel])
        assert callback_manager.get_dashboard_callback_registrations() == (("otel", "success"),)
    finally:
        logging_module._in_memory_loggers.clear()
        is_otel_v2_enabled.cache_clear()


@pytest.mark.asyncio
async def test_dashboard_otel_provenance_registered_on_v2_preset_cache_hit_after_reset(
    callback_manager,
    monkeypatch,
):
    from litellm.integrations.otel.model.config import is_otel_v2_enabled
    from litellm.litellm_core_utils import litellm_logging as logging_module

    monkeypatch.setenv("LITELLM_OTEL_V2", "true")
    is_otel_v2_enabled.cache_clear()
    logging_module._in_memory_loggers.clear()
    try:
        arize = logging_module._init_custom_logger_compatible_class("arize", None, None, {})
        callback_manager._reset_all_callbacks()
        cached_arize = logging_module._init_custom_logger_compatible_class("arize", None, None, {})
        assert cached_arize is arize
        monkeypatch.setattr(litellm, "success_callback", [cached_arize])

        assert callback_manager.get_dashboard_callback_registrations() == (("arize", "success"),)
    finally:
        logging_module._in_memory_loggers.clear()
        is_otel_v2_enabled.cache_clear()


@pytest.mark.asyncio
async def test_dashboard_otel_provenance_registered_on_v2_otel_cache_hit(
    callback_manager,
    monkeypatch,
):
    from litellm.integrations.otel.model.config import is_otel_v2_enabled
    from litellm.litellm_core_utils import litellm_logging as logging_module

    monkeypatch.setenv("LITELLM_OTEL_V2", "true")
    is_otel_v2_enabled.cache_clear()
    logging_module._in_memory_loggers.clear()
    try:
        arize = logging_module._init_custom_logger_compatible_class("arize", None, None, {})
        otel = logging_module._init_custom_logger_compatible_class("otel", None, None, {})
        assert otel is not arize
        monkeypatch.setattr(litellm, "success_callback", [otel])
        monkeypatch.setattr(litellm, "failure_callback", [])
        monkeypatch.setattr(litellm, "callbacks", [])
        monkeypatch.setattr(litellm, "_async_success_callback", [])
        monkeypatch.setattr(litellm, "_async_failure_callback", [])

        assert callback_manager.get_dashboard_callback_registrations() == (("otel", "success"),)
    finally:
        logging_module._in_memory_loggers.clear()
        is_otel_v2_enabled.cache_clear()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("otel_v2_enabled", "expected_type"),
    ((True, "OpenTelemetryV2"), (False, "OpenTelemetry")),
)
async def test_dynamic_otel_callback_resolution_is_request_local(
    callback_manager,
    monkeypatch,
    otel_v2_enabled,
    expected_type,
):
    from litellm.integrations.opentelemetry import OpenTelemetry
    from litellm.integrations.otel.logger import OpenTelemetryV2
    from litellm.integrations.otel.model.config import is_otel_v2_enabled
    from litellm.litellm_core_utils import litellm_logging as logging_module

    monkeypatch.setenv("LITELLM_OTEL_V2", str(otel_v2_enabled).lower())
    is_otel_v2_enabled.cache_clear()
    monkeypatch.setattr(logging_module, "_in_memory_loggers", [])
    monkeypatch.setattr(litellm, "input_callback", [])
    monkeypatch.setattr(litellm, "service_callback", [])
    monkeypatch.setattr(litellm, "success_callback", [])
    monkeypatch.setattr(litellm, "failure_callback", [])
    monkeypatch.setattr(litellm, "_async_success_callback", [])
    monkeypatch.setattr(litellm, "_async_failure_callback", [])
    monkeypatch.setattr(litellm, "callbacks", [])

    request_logging = logging_module.Logging(
        model="test-model",
        messages=[],
        stream=False,
        call_type="completion",
        start_time=None,
        litellm_call_id="request-id",
        function_id="function-id",
        dynamic_success_callbacks=["otel"],
    )

    dynamic_callback = request_logging.dynamic_success_callbacks[0]
    expected_callback_type = OpenTelemetryV2 if expected_type == "OpenTelemetryV2" else OpenTelemetry
    assert type(dynamic_callback) is expected_callback_type
    assert dynamic_callback.callback_name == "otel"
    assert logging_module._in_memory_loggers == []
    assert all(
        dynamic_callback is not registered_callback
        for callback_registry in (
            litellm.input_callback,
            litellm.service_callback,
            litellm.success_callback,
            litellm.failure_callback,
            litellm._async_success_callback,
            litellm._async_failure_callback,
            litellm.callbacks,
        )
        for registered_callback in callback_registry
    )
    assert callback_manager.get_dashboard_callback_registrations() == ()
    is_otel_v2_enabled.cache_clear()


@pytest.mark.parametrize(
    ("otel_v2_enabled", "expected_type"),
    ((True, "OpenTelemetryV2"), (False, "OpenTelemetry")),
)
def test_prompt_management_otel_resolution_is_request_local(
    callback_manager,
    monkeypatch,
    otel_v2_enabled,
    expected_type,
):
    from litellm.integrations.opentelemetry import OpenTelemetry
    from litellm.integrations.otel.logger import OpenTelemetryV2
    from litellm.integrations.otel.model.config import is_otel_v2_enabled
    from litellm.litellm_core_utils import litellm_logging as logging_module

    monkeypatch.setenv("LITELLM_OTEL_V2", str(otel_v2_enabled).lower())
    is_otel_v2_enabled.cache_clear()
    monkeypatch.setattr(logging_module, "_in_memory_loggers", [])
    monkeypatch.setattr(litellm, "input_callback", [])
    monkeypatch.setattr(litellm, "service_callback", [])
    monkeypatch.setattr(litellm, "success_callback", [])
    monkeypatch.setattr(litellm, "failure_callback", [])
    monkeypatch.setattr(litellm, "_async_success_callback", [])
    monkeypatch.setattr(litellm, "_async_failure_callback", [])
    monkeypatch.setattr(litellm, "callbacks", [])

    request_logging = logging_module.Logging(
        model="otel/prompt",
        messages=[],
        stream=False,
        call_type="completion",
        start_time=None,
        litellm_call_id="request-id",
        function_id="function-id",
    )
    prompt_logger = request_logging.get_custom_logger_for_prompt_management(
        model="otel/prompt",
        non_default_params={},
    )

    expected_callback_type = OpenTelemetryV2 if expected_type == "OpenTelemetryV2" else OpenTelemetry
    assert type(prompt_logger) is expected_callback_type
    assert prompt_logger.callback_name == "otel"
    assert logging_module._in_memory_loggers == []
    assert all(
        prompt_logger is not registered_callback
        for callback_registry in (
            litellm.input_callback,
            litellm.service_callback,
            litellm.success_callback,
            litellm.failure_callback,
            litellm._async_success_callback,
            litellm._async_failure_callback,
            litellm.callbacks,
        )
        for registered_callback in callback_registry
    )
    assert callback_manager.get_dashboard_callback_registrations() == ()
    is_otel_v2_enabled.cache_clear()


@pytest.mark.parametrize(
    ("otel_v2_enabled", "expected_type"),
    ((True, "OpenTelemetryV2"), (False, "OpenTelemetry")),
)
def test_function_setup_promotes_cold_otel_only_after_duplicate_check(
    callback_manager,
    monkeypatch,
    otel_v2_enabled,
    expected_type,
):
    from litellm.integrations.opentelemetry import OpenTelemetry
    from litellm.integrations.otel.logger import OpenTelemetryV2
    from litellm.integrations.otel.model.config import is_otel_v2_enabled
    from litellm.litellm_core_utils import litellm_logging as logging_module
    from litellm.proxy import proxy_server

    checker = MagicMock()
    checker.is_async_callable.return_value = False
    logging_object = MagicMock()
    monkeypatch.setenv("LITELLM_OTEL_V2", str(otel_v2_enabled).lower())
    monkeypatch.setattr(litellm.utils.litellm_utils, "get_coroutine_checker", lambda: checker)
    monkeypatch.setattr(litellm.utils, "get_litellm_logging_class", lambda: logging_object)
    monkeypatch.setattr(litellm.utils, "callback_list", ["already-initialized"])
    monkeypatch.setattr(logging_module, "_in_memory_loggers", [])
    monkeypatch.setattr(litellm, "input_callback", [])
    monkeypatch.setattr(litellm, "service_callback", [])
    monkeypatch.setattr(litellm, "success_callback", [])
    monkeypatch.setattr(litellm, "failure_callback", [])
    monkeypatch.setattr(litellm, "_async_success_callback", [])
    monkeypatch.setattr(litellm, "_async_failure_callback", [])
    monkeypatch.setattr(litellm, "callbacks", [])
    monkeypatch.setattr(proxy_server, "open_telemetry_logger", None)
    is_otel_v2_enabled.cache_clear()
    try:
        litellm.utils.function_setup(
            original_function="acompletion",
            rules_obj=litellm.utils.Rules(),
            start_time=datetime.now(),
            model="gpt-4",
            messages=[],
            litellm_call_id="cold-global-promotion",
            callbacks=["otel"],
        )

        otel_callback = litellm._async_success_callback[0]
        expected_callback_type = OpenTelemetryV2 if expected_type == "OpenTelemetryV2" else OpenTelemetry
        assert type(otel_callback) is expected_callback_type
        assert otel_callback.callback_name == "otel"
        assert otel_callback in logging_module._in_memory_loggers
        assert otel_callback in litellm.input_callback
        assert otel_callback in litellm.service_callback
        assert otel_callback in litellm._async_failure_callback
        assert proxy_server.open_telemetry_logger is otel_callback
        assert callback_manager.get_dashboard_callback_registrations() == (("otel", "success_and_failure"),)
    finally:
        is_otel_v2_enabled.cache_clear()


def test_add_string_callback():
    """
    Test adding a string callback to litellm.callbacks - only 1 instance of the string callback should be added
    """
    manager = LoggingCallbackManager()
    test_callback = "test_callback"

    # Add string callback
    manager.add_litellm_callback(test_callback)
    assert test_callback in litellm.callbacks

    # Test duplicate prevention
    manager.add_litellm_callback(test_callback)
    assert litellm.callbacks.count(test_callback) == 1


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

    # Check exactly one instance of each logger type
    langfuse_count = sum(1 for callback in litellm.success_callback if isinstance(callback, LangfusePromptManagement))
    otel_count = sum(1 for callback in litellm.success_callback if isinstance(callback, OpenTelemetry))

    assert langfuse_count == 1, "Should have exactly one LangfusePromptManagement instance"
    assert otel_count == 1, "Should have exactly one OpenTelemetry instance"


def test_add_function_callback():
    manager = LoggingCallbackManager()

    def test_func(kwargs):
        pass

    # Add function callback
    manager.add_litellm_callback(test_func)
    assert test_func in litellm.callbacks

    # Test duplicate prevention
    manager.add_litellm_callback(test_func)
    assert litellm.callbacks.count(test_func) == 1


def test_add_custom_logger(mock_custom_logger):
    manager = LoggingCallbackManager()

    # Add custom logger
    manager.add_litellm_callback(mock_custom_logger)
    assert mock_custom_logger in litellm.callbacks


def test_add_multiple_callback_types(mock_custom_logger):
    manager = LoggingCallbackManager()

    def test_func(kwargs):
        pass

    string_callback = "test_callback"

    # Add different types of callbacks
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

    # Add callbacks
    manager.add_litellm_success_callback(success_callback)
    manager.add_litellm_failure_callback(failure_callback)

    assert success_callback in litellm.success_callback
    assert failure_callback in litellm.failure_callback


def test_async_callbacks():
    manager = LoggingCallbackManager()

    async_success = "async_success"
    async_failure = "async_failure"

    # Add async callbacks
    manager.add_litellm_async_success_callback(async_success)
    manager.add_litellm_async_failure_callback(async_failure)

    assert async_success in litellm._async_success_callback
    assert async_failure in litellm._async_failure_callback


def test_remove_callback_from_list_by_object():
    manager = LoggingCallbackManager()
    # Reset all callbacks
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

    # Verify all callback lists are empty
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
    # Add various callbacks
    callback_manager.add_litellm_callback("test")
    callback_manager.add_litellm_success_callback("success")
    callback_manager.add_litellm_failure_callback("failure")
    callback_manager.add_litellm_async_success_callback("async_success")
    callback_manager.add_litellm_async_failure_callback("async_failure")

    # Reset all callbacks
    callback_manager._reset_all_callbacks()

    # Verify all callback lists are empty
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
    from litellm.integrations.SlackAlerting.slack_alerting import SlackAlerting
    from litellm.proxy.utils import ProxyLogging

    # Mock the async HTTP handler
    with patch("litellm.integrations.SlackAlerting.slack_alerting.get_async_httpx_client") as mock_http:
        mock_http.return_value = AsyncMock()

        # Create a fresh ProxyLogging instance
        proxy_logging = ProxyLogging(user_api_key_cache=DualCache())

        # Test 1: No callbacks should be added when alerting is None
        proxy_logging.update_values(alerting=None, alert_types=["outage_alerts", "region_outage_alerts"])
        assert len(litellm.callbacks) == 0

        # Test 2: Callbacks should be added when slack alerting is enabled with outage alerts
        proxy_logging.update_values(alerting=["slack"], alert_types=["outage_alerts"])
        assert len(litellm.callbacks) == 1
        assert isinstance(litellm.callbacks[0], SlackAlerting)

        # Test 3: Callbacks should be added when slack alerting is enabled with region outage alerts
        callback_manager._reset_all_callbacks()  # Reset callbacks
        proxy_logging.update_values(alerting=["slack"], alert_types=["region_outage_alerts"])
        assert len(litellm.callbacks) == 1
        assert isinstance(litellm.callbacks[0], SlackAlerting)

        # Test 4: No callbacks should be added for other alert types
        callback_manager._reset_all_callbacks()  # Reset callbacks
        proxy_logging.update_values(
            alerting=["slack"],
            alert_types=["budget_alerts"],  # Some other alert type
        )
        assert len(litellm.callbacks) == 0

        # Test 5: Both success and regular callbacks should be added
        callback_manager._reset_all_callbacks()  # Reset callbacks
        proxy_logging.update_values(alerting=["slack"], alert_types=["outage_alerts"])
        assert len(litellm.callbacks) == 1  # Regular callback for outage alerts
        assert isinstance(litellm.callbacks[0], SlackAlerting)
        # response_taking_too_long_callback is async, so it should be in the async success callback list
        response_taking_too_long_callback = proxy_logging.slack_alerting_instance.response_taking_too_long_callback
        assert len(litellm._async_success_callback) == 1
        assert litellm._async_success_callback[0] == response_taking_too_long_callback

        # Cleanup
        callback_manager._reset_all_callbacks()


@pytest.mark.asyncio
async def test_generic_api_compatible_callbacks_json():
    """
    Test that callbacks defined in generic_api_compatible_callbacks.json
    are properly loaded and initialized by _add_custom_callback_generic_api_str
    """
    from litellm.integrations.generic_api.generic_api_callback import GenericAPILogger

    # Mock environment variable for SumoLogic webhook URL
    test_sumologic_url = "https://collectors.sumologic.com/receiver/v1/http/test123"

    with patch.dict(os.environ, {"SUMOLOGIC_WEBHOOK_URL": test_sumologic_url}):
        # Test that sumologic callback is recognized from JSON file
        result = LoggingCallbackManager._add_custom_callback_generic_api_str("sumologic")

        # Verify a GenericAPILogger instance is returned
        assert isinstance(result, GenericAPILogger), "Should return GenericAPILogger instance for sumologic callback"

        # Verify the endpoint is correctly loaded from environment variable
        assert result.endpoint == test_sumologic_url, f"Endpoint should be {test_sumologic_url}"

        # Verify headers only contain Content-Type (no Authorization for SumoLogic)
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

    # Mock environment variables for Rubrik
    test_rubrik_url = "https://webhook.site/test-rubrik"
    test_rubrik_api_key = "sk-rubrik-test-key"

    with patch.dict(
        os.environ,
        {"RUBRIK_WEBHOOK_URL": test_rubrik_url, "RUBRIK_API_KEY": test_rubrik_api_key},
    ):
        # Test that rubrik callback is recognized from JSON file
        result = LoggingCallbackManager._add_custom_callback_generic_api_str("rubrik")

        # Verify a GenericAPILogger instance is returned
        assert isinstance(result, GenericAPILogger), "Should return GenericAPILogger instance for rubrik callback"

        # Verify the endpoint is correctly loaded
        assert result.endpoint == test_rubrik_url, f"Endpoint should be {test_rubrik_url}"

        # Verify headers include Authorization with Bearer token
        assert "Content-Type" in result.headers, "Should have Content-Type header"
        assert "Authorization" in result.headers, "Should have Authorization header for Rubrik"
        assert result.headers["Authorization"] == f"Bearer {test_rubrik_api_key}", (
            "Authorization should have correct API key"
        )

        # Verify event_types filter (rubrik only logs success events)
        assert result.event_types == ["llm_api_success"], "Rubrik should only log success events"


def test_generic_api_compatible_callbacks_json_unknown_callback():
    """
    Test that unknown callbacks (not in JSON or callback_settings) are returned unchanged
    """
    # Test with a callback that doesn't exist in the JSON file
    result = LoggingCallbackManager._add_custom_callback_generic_api_str("unknown_callback")

    # Should return the string unchanged
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
        result = LoggingCallbackManager._add_custom_callback_generic_api_str(callback_name)

        assert isinstance(result, GenericAPILogger)
        assert result.endpoint == "https://example.com/api/logs"
        assert result.headers == {"Content-Type": "application/json"}
        assert result.max_retries == 2
        assert result.retry_delay == 0.5
        assert result.timeout == 3
    finally:
        litellm.callback_settings.pop(callback_name, None)
        _generic_api_logger_cache.pop(callback_name, None)
