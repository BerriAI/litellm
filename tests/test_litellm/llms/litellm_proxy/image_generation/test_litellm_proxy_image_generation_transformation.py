from unittest.mock import patch

import pytest

from litellm.llms.litellm_proxy.image_generation.transformation import (
    LiteLLMProxyImageGenerationConfig,
)


@pytest.mark.parametrize(
    "input_api_key, env_api_key, expected_bearer",
    [
        ("user-provided-key", "secret-key", "Bearer user-provided-key"),
        (None, "secret-key", "Bearer secret-key"),
        (None, None, "Bearer fake-api-key"),
        ("", "secret-key", "Bearer secret-key"),
        ("", None, "Bearer fake-api-key"),
    ],
)
def test_validate_environment(input_api_key, env_api_key, expected_bearer):
    config = LiteLLMProxyImageGenerationConfig()
    env = {}
    if env_api_key is not None:
        env["LITELLM_PROXY_API_KEY"] = env_api_key

    with patch.dict("os.environ", env, clear=True):
        headers = config.validate_environment(
            headers={},
            model="dall-e-3",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key=input_api_key,
        )
        assert headers.get("Authorization") == expected_bearer
