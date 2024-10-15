import json
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock

from pydantic.main import Model
import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.litellm_core_utils.litellm_logging import (
    get_custom_logger_compatible_class,
    _in_memory_loggers,
)
from litellm.integrations.lago import LagoLogger
from litellm.integrations.openmeter import OpenMeterLogger
from litellm.integrations.braintrust_logging import BraintrustLogger
from litellm.integrations.galileo import GalileoObserve
from litellm.integrations.langsmith import LangsmithLogger
from litellm.integrations.literal_ai import LiteralAILogger
from litellm.integrations.prometheus import PrometheusLogger
from litellm.integrations.s3 import S3Logger
from litellm.integrations.datadog.datadog import DataDogLogger
from litellm.integrations.gcs_bucket.gcs_bucket import GCSBucketLogger
from litellm.integrations.opik.opik import OpikLogger
from litellm.integrations.opentelemetry import OpenTelemetry
from litellm.proxy.hooks.dynamic_rate_limiter import _PROXY_DynamicRateLimitHandler


@pytest.fixture(autouse=True)
def clear_in_memory_loggers():
    _in_memory_loggers.clear()
    yield
    _in_memory_loggers.clear()


all_tested_callbacks = []


@pytest.mark.parametrize(
    "logging_integration,logger_class,env_vars",
    [
        (
            "lago",
            LagoLogger,
            {
                "LAGO_API_KEY": "mock_lago_api_key",
                "LAGO_API_BASE": "mock_lago_base",
                "LAGO_API_EVENT_CODE": "mock_lago_event_code",
            },
        ),
        ("openmeter", OpenMeterLogger, {"OPENMETER_API_KEY": "mock_openmeter_api_key"}),
        (
            "braintrust",
            BraintrustLogger,
            {"BRAINTRUST_API_KEY": "mock_braintrust_api_key"},
        ),
        ("galileo", GalileoObserve, {"GALILEO_API_KEY": "mock_galileo_api_key"}),
        ("langsmith", LangsmithLogger, {"LANGCHAIN_API_KEY": "mock_langchain_api_key"}),
        ("literalai", LiteralAILogger, {"LITERAL_API_KEY": "mock_literal_api_key"}),
        ("prometheus", PrometheusLogger, {}),  # Prometheus might not need an env var
        (
            "s3",
            S3Logger,
            {
                "AWS_ACCESS_KEY_ID": "mock_aws_key",
                "AWS_SECRET_ACCESS_KEY": "mock_aws_secret",
            },
        ),
        (
            "datadog",
            DataDogLogger,
            {
                "DD_API_KEY": "mock_datadog_api_key",
                "DD_SITE": "https://us5.datadoghq.com",
            },
        ),
        (
            "gcs_bucket",
            GCSBucketLogger,
            {"GOOGLE_APPLICATION_CREDENTIALS": "mock_gcp_credentials.json"},
        ),
        ("opik", OpikLogger, {"OPIK_API_KEY": "mock_opik_api_key"}),
        (
            "otel",
            OpenTelemetry,
            {"OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:4317"},
        ),
        # ("dynamic_rate_limiter", _PROXY_DynamicRateLimitHandler, {}),  # This might not need env vars
    ],
)
def test_get_custom_logger_compatible_class(
    logging_integration, logger_class, env_vars
):
    """
    Tests that if the logger_class is within _in_memory_loggers, then get_custom_logger_compatible_class returns the logger_class
    """
    all_tested_callbacks.append(logger_class)
    # Set environment variables

    _set_env_vars = []
    for key, value in env_vars.items():
        if key not in os.environ:
            os.environ[key] = value
            _set_env_vars.append(key)
    # Initialize the logger
    logger = logger_class()
    _in_memory_loggers.append(logger)

    # Test the function
    result = get_custom_logger_compatible_class(logging_integration)

    assert result is not None
    assert isinstance(result, logger_class)

    # Clean up environment variables
    for key in _set_env_vars:
        del os.environ[key]

    all_tested_callbacks.append(logger_class)


# @pytest.mark.dependency(depends=["test_get_custom_logger_compatible_class"])
# def test_all_callbacks_tested():
#     """
#     Asserts that all callbacks in litellm._known_custom_logger_compatible_callbacks were tested
#     """
#     untested_callbacks = set(litellm._known_custom_logger_compatible_callbacks) - set(all_tested_callbacks)
#     assert not untested_callbacks, f"The following CustomLogger callbacks were not tested: {untested_callbacks}"


@pytest.mark.parametrize(
    "logging_integration,logger_class,env_vars",
    [
        (
            "arize",
            OpenTelemetry,
            {"ARIZE_SPACE_KEY": "mock_space_key", "ARIZE_API_KEY": "mock_api_key"},
        ),
        ("logfire", OpenTelemetry, {"LOGFIRE_TOKEN": "mock_token"}),
        ("langtrace", OpenTelemetry, {"LANGTRACE_API_KEY": "mock_api_key"}),
    ],
)
def test_get_custom_logger_compatible_class_with_env_vars(
    logging_integration, logger_class, env_vars
):
    # Set environment variables
    for key, value in env_vars.items():
        os.environ[key] = value

    # Initialize the logger
    logger = logger_class()
    if logging_integration in ["arize", "langtrace"]:
        logger.callback_name = logging_integration
    _in_memory_loggers.append(logger)

    # Test the function
    result = get_custom_logger_compatible_class(logging_integration)

    assert result is not None
    assert isinstance(result, logger_class)
    if logging_integration in ["arize", "langtrace"]:
        assert result.callback_name == logging_integration

    # Clean up environment variables
    for key in env_vars:
        del os.environ[key]


def test_get_custom_logger_compatible_class_non_existent():
    """
    Tests that if the logger_class is not within _in_memory_loggers and is unknown then get_custom_logger_compatible_class returns None

    """
    result = get_custom_logger_compatible_class("non_existent_integration")
    assert result is None
