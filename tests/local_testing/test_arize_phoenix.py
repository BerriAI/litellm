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
    "test_name, env_vars, expected_headers, expected_endpoint, expected_protocol",
    [
        ("default to grpc protocol and arize hosted phoenix endpoint", {"PHOENIX_API_KEY": "test_api_key"}, "api_key=test_api_key", "https://app.phoenix.arize.com/v1/traces", "otlp_grpc"),
        ("custom grpc endpoint", {"PHOENIX_COLLECTOR_ENDPOINT": "https://localhost:6006","PHOENIX_API_KEY": "test_api_key"}, "Authorization=Bearer test_api_key", "https://localhost:6006", "otlp_grpc"),
        ("custom grpc endpoint with no auth", {"PHOENIX_COLLECTOR_ENDPOINT": "https://localhost:6006"}, None, "https://localhost:6006", "otlp_grpc"),
        ("custom http endpoint", {"PHOENIX_COLLECTOR_HTTP_ENDPOINT": "https://localhost:6006", "PHOENIX_API_KEY": "test_api_key"}, "Authorization=Bearer test_api_key", "https://localhost:6006", "otlp_http"),
    ],
)
def test_get_arize_phoenix_config(test_name, monkeypatch, env_vars, expected_headers, expected_endpoint, expected_protocol):
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)

    config = ArizePhoenixLogger.get_arize_phoenix_config()
    assert isinstance(config, ArizePhoenixConfig), f"Failed for {test_name}"
    assert config.otlp_auth_headers == expected_headers, f"Failed for {test_name}"
    assert config.endpoint == expected_endpoint, f"Failed for  {test_name}"
    assert config.protocol == expected_protocol, f"Failed for  {test_name}"
