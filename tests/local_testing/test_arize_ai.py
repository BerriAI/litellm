import asyncio
import json
import logging
import os
import time
from unittest.mock import patch, Mock
import opentelemetry.exporter.otlp.proto.grpc.trace_exporter
from litellm import Choices
import pytest
from dotenv import load_dotenv

import litellm
from litellm._logging import verbose_logger, verbose_proxy_logger
from litellm.integrations.arize.arize import ArizeConfig, ArizeLogger

load_dotenv()


@pytest.mark.asyncio()
async def test_async_otel_callback():
    litellm.set_verbose = True

    verbose_proxy_logger.setLevel(logging.DEBUG)
    verbose_logger.setLevel(logging.DEBUG)
    litellm.success_callback = ["arize"]

    await litellm.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "hi test from local arize"}],
        mock_response="hello",
        temperature=0.1,
        user="OTEL_USER",
    )

    await asyncio.sleep(2)


@pytest.mark.asyncio()
async def test_async_dynamic_arize_config():
    litellm.set_verbose = True

    verbose_proxy_logger.setLevel(logging.DEBUG)
    verbose_logger.setLevel(logging.DEBUG)
    litellm.success_callback = ["arize"]

    await litellm.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "hi test from arize dynamic config"}],
        temperature=0.1,
        user="OTEL_USER",
        arize_api_key=os.getenv("ARIZE_SPACE_2_API_KEY"),
        arize_space_key=os.getenv("ARIZE_SPACE_2_KEY"),
    )

    await asyncio.sleep(2)


@pytest.fixture
def mock_env_vars(monkeypatch):
    monkeypatch.setenv("ARIZE_SPACE_KEY", "test_space_key")
    monkeypatch.setenv("ARIZE_API_KEY", "test_api_key")


def test_get_arize_config(mock_env_vars):
    """
    Use Arize default endpoint when no endpoints are provided
    """
    config = ArizeLogger.get_arize_config()
    assert isinstance(config, ArizeConfig)
    assert config.space_key == "test_space_key"
    assert config.api_key == "test_api_key"
    assert config.endpoint == "https://otlp.arize.com/v1"
    assert config.protocol == "otlp_grpc"
    assert config.project_name is None


def test_get_arize_config_with_endpoints(mock_env_vars, monkeypatch):
    """
    Use provided endpoints when they are set
    """
    monkeypatch.setenv("ARIZE_ENDPOINT", "grpc://test.endpoint")
    monkeypatch.setenv("ARIZE_HTTP_ENDPOINT", "http://test.endpoint")
    monkeypatch.setenv("ARIZE_PROJECT_NAME", "custom-project")

    config = ArizeLogger.get_arize_config()
    assert config.endpoint == "grpc://test.endpoint"
    assert config.protocol == "otlp_grpc"
    assert config.project_name == "custom-project"


@pytest.mark.skip(
    reason="Works locally but not in CI/CD. We'll need a better way to test Arize on CI/CD"
)
def test_arize_callback():
    litellm.callbacks = ["arize"]
    os.environ["ARIZE_SPACE_KEY"] = "test_space_key"
    os.environ["ARIZE_API_KEY"] = "test_api_key"
    os.environ["ARIZE_ENDPOINT"] = "https://otlp.arize.com/v1"

    # Set the batch span processor to quickly flush after a span has been added
    # This is to ensure that the span is exported before the test ends
    os.environ["OTEL_BSP_MAX_QUEUE_SIZE"] = "1"
    os.environ["OTEL_BSP_MAX_EXPORT_BATCH_SIZE"] = "1"
    os.environ["OTEL_BSP_SCHEDULE_DELAY_MILLIS"] = "1"
    os.environ["OTEL_BSP_EXPORT_TIMEOUT_MILLIS"] = "5"

    try:
        with patch.object(
            opentelemetry.exporter.otlp.proto.grpc.trace_exporter.OTLPSpanExporter,
            "export",
            new=Mock(),
        ) as patched_export:
            litellm.completion(
                model="openai/test-model",
                messages=[{"role": "user", "content": "arize test content"}],
                stream=False,
                mock_response="hello there!",
            )

            time.sleep(1)  # Wait for the batch span processor to flush
            assert patched_export.called
    finally:
        # Clean up environment variables
        for key in [
            "ARIZE_SPACE_KEY",
            "ARIZE_API_KEY",
            "ARIZE_ENDPOINT",
            "OTEL_BSP_MAX_QUEUE_SIZE",
            "OTEL_BSP_MAX_EXPORT_BATCH_SIZE",
            "OTEL_BSP_SCHEDULE_DELAY_MILLIS",
            "OTEL_BSP_EXPORT_TIMEOUT_MILLIS",
        ]:
            if key in os.environ:
                del os.environ[key]

        # Reset callbacks
        litellm.callbacks = []
