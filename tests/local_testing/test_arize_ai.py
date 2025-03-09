import asyncio
import logging

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


def test_get_arize_config_with_endpoints(mock_env_vars, monkeypatch):
    """
    Use provided endpoints when they are set
    """
    monkeypatch.setenv("ARIZE_ENDPOINT", "grpc://test.endpoint")
    monkeypatch.setenv("ARIZE_HTTP_ENDPOINT", "http://test.endpoint")

    config = ArizeLogger.get_arize_config()
    assert config.endpoint == "grpc://test.endpoint"
    assert config.protocol == "otlp_grpc"
