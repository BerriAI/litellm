from unittest.mock import patch

import pytest

from litellm.llms.litellm_proxy.responses.transformation import (
    LiteLLMProxyResponsesAPIConfig,
)
from litellm.types.router import GenericLiteLLMParams


@pytest.mark.parametrize(
    "litellm_params_api_key, env_api_key, expected_bearer",
    [
        ("user-provided-key", "secret-key", "Bearer user-provided-key"),
        (None, "secret-key", "Bearer secret-key"),
        (None, None, "Bearer fake-api-key"),
        ("", "secret-key", "Bearer secret-key"),
        ("", None, "Bearer fake-api-key"),
    ],
)
def test_validate_environment(litellm_params_api_key, env_api_key, expected_bearer):
    config = LiteLLMProxyResponsesAPIConfig()
    env = {}
    if env_api_key is not None:
        env["LITELLM_PROXY_API_KEY"] = env_api_key

    litellm_params = GenericLiteLLMParams(api_key=litellm_params_api_key)

    with patch.dict("os.environ", env, clear=True):
        headers = config.validate_environment(
            headers={}, model="gpt-4o", litellm_params=litellm_params
        )
        assert headers.get("Authorization") == expected_bearer


def test_validate_environment_does_not_use_openai_key():
    """
    OPENAI_API_KEY should NOT be used for litellm_proxy requests.
    The proxy should use LITELLM_PROXY_API_KEY or fall back to fake-api-key.
    """
    config = LiteLLMProxyResponsesAPIConfig()
    env = {"OPENAI_API_KEY": "sk-real-openai-key"}
    litellm_params = GenericLiteLLMParams()

    with patch.dict("os.environ", env, clear=True):
        headers = config.validate_environment(
            headers={}, model="gpt-4o", litellm_params=litellm_params
        )
        assert headers.get("Authorization") == "Bearer fake-api-key"
