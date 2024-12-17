import asyncio
import logging
import os
import time

import pytest
from dotenv import load_dotenv
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import litellm
from litellm._logging import verbose_logger, verbose_proxy_logger
from litellm.integrations.opentelemetry import OpenTelemetry, OpenTelemetryConfig
from litellm.integrations.arize_ai import ArizeConfig, ArizeLogger

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


@pytest.fixture
def mock_env_vars(monkeypatch):
    monkeypatch.setenv("ARIZE_SPACE_KEY", "test_space_key")
    monkeypatch.setenv("ARIZE_API_KEY", "test_api_key")


def test_get_arize_config(mock_env_vars):
    """
    Use Arize default endpoint when no endpoints are provided
    """
    config = ArizeLogger._get_arize_config()
    assert isinstance(config, ArizeConfig)
    assert config.space_key == "test_space_key"
    assert config.api_key == "test_api_key"
    assert config.grpc_endpoint == "https://otlp.arize.com/v1"
    assert config.http_endpoint is None


def test_get_arize_config_with_endpoints(mock_env_vars, monkeypatch):
    """
    Use provided endpoints when they are set
    """
    monkeypatch.setenv("ARIZE_ENDPOINT", "grpc://test.endpoint")
    monkeypatch.setenv("ARIZE_HTTP_ENDPOINT", "http://test.endpoint")

    config = ArizeLogger._get_arize_config()
    assert config.grpc_endpoint == "grpc://test.endpoint"
    assert config.http_endpoint == "http://test.endpoint"


def test_get_arize_opentelemetry_config_grpc(mock_env_vars, monkeypatch):
    """
    Use provided GRPC endpoint when it is set
    """
    monkeypatch.setenv("ARIZE_ENDPOINT", "grpc://test.endpoint")

    config = ArizeLogger.get_arize_opentelemetry_config()
    assert isinstance(config, OpenTelemetryConfig)
    assert config.exporter == "otlp_grpc"
    assert config.endpoint == "grpc://test.endpoint"


def test_get_arize_opentelemetry_config_http(mock_env_vars, monkeypatch):
    """
    Use provided HTTP endpoint when it is set
    """
    monkeypatch.setenv("ARIZE_HTTP_ENDPOINT", "http://test.endpoint")

    config = ArizeLogger.get_arize_opentelemetry_config()
    assert isinstance(config, OpenTelemetryConfig)
    assert config.exporter == "otlp_http"
    assert config.endpoint == "http://test.endpoint"
