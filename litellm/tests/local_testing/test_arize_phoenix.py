import asyncio
import logging
import pytest
from dotenv import load_dotenv

import litellm
from litellm._logging import verbose_logger, verbose_proxy_logger
from litellm.integrations.arize.arize_phoenix import ArizePhoenixConfig, ArizePhoenixLogger

load_dotenv()


@pytest.mark.asyncio()
async def test_async_otel_callback():
    litellm.set_verbose = True

    verbose_proxy_logger.setLevel(logging.DEBUG)
    verbose_logger.setLevel(logging.DEBUG)
    litellm.success_callback = ["arize_phoenix"]

    await litellm.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "this is arize phoenix"}],
        mock_response="hello",
        temperature=0.1,
        user="OTEL_USER",
    )

    await asyncio.sleep(2)


@pytest.mark.parametrize(
    "env_vars, expected_headers, expected_endpoint, expected_protocol",
    [
        pytest.param(
            {"PHOENIX_API_KEY": "test_api_key"},
            "api_key=test_api_key",
            "https://app.phoenix.arize.com/v1/traces",
            "otlp_http",
            id="default to http protocol and Arize hosted Phoenix endpoint",
        ),
        pytest.param(
            {"PHOENIX_COLLECTOR_HTTP_ENDPOINT": "", "PHOENIX_API_KEY": "test_api_key"},
            "api_key=test_api_key",
            "https://app.phoenix.arize.com/v1/traces",
            "otlp_http",
            id="empty string/unset endpoint will default to http protocol and Arize hosted Phoenix endpoint",
        ),
        pytest.param(
            {"PHOENIX_COLLECTOR_HTTP_ENDPOINT": "http://localhost:4318", "PHOENIX_COLLECTOR_ENDPOINT": "http://localhost:4317", "PHOENIX_API_KEY": "test_api_key"},
            "Authorization=Bearer%20test_api_key",
            "http://localhost:4318",
            "otlp_http",
            id="prioritize http if both endpoints are set",
        ),
        pytest.param(
            {"PHOENIX_COLLECTOR_ENDPOINT": "https://localhost:6006", "PHOENIX_API_KEY": "test_api_key"},
            "Authorization=Bearer%20test_api_key",
            "https://localhost:6006",
            "otlp_grpc",
            id="custom grpc endpoint",
        ),
        pytest.param(
            {"PHOENIX_COLLECTOR_ENDPOINT": "https://localhost:6006"},
            None,
            "https://localhost:6006",
            "otlp_grpc",
            id="custom grpc endpoint with no auth",
        ),
        pytest.param(
            {"PHOENIX_COLLECTOR_HTTP_ENDPOINT": "https://localhost:6006", "PHOENIX_API_KEY": "test_api_key"},
            "Authorization=Bearer%20test_api_key",
            "https://localhost:6006",
            "otlp_http",
            id="custom http endpoint",
        ),
    ],
)
def test_get_arize_phoenix_config(monkeypatch, env_vars, expected_headers, expected_endpoint, expected_protocol):
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)

    config = ArizePhoenixLogger.get_arize_phoenix_config()

    assert isinstance(config, ArizePhoenixConfig)
    assert config.otlp_auth_headers == expected_headers
    assert config.endpoint == expected_endpoint
    assert config.protocol == expected_protocol

@pytest.mark.parametrize(
    "env_vars",
    [
        pytest.param(
            {"PHOENIX_COLLECTOR_ENDPOINT": "https://app.phoenix.arize.com/v1/traces"},
            id="missing api_key with explicit Arize Phoenix endpoint"
        ),
        pytest.param(
            {},
            id="missing api_key with no endpoint (defaults to Arize Phoenix)"
        ),
    ],
)
def test_get_arize_phoenix_config_expection_on_missing_api_key(monkeypatch, env_vars):
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)

    with pytest.raises(ValueError, match=f"PHOENIX_API_KEY must be set when the Arize hosted Phoenix endpoint is used."):
        ArizePhoenixLogger.get_arize_phoenix_config()
