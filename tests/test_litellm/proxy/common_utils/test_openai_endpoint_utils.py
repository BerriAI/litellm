import pytest

from litellm.proxy.common_utils.openai_endpoint_utils import (
    remove_sensitive_info_from_deployment,
)


@pytest.mark.parametrize(
    "model_config, expected_config",
    [
        # Test case 1: Empty litellm_params
        (
            {"model_name": "test-model", "litellm_params": {}},
            {"model_name": "test-model", "litellm_params": {}},
        ),
        # Test case 2: Full sensitive data removal, mixed secrets of azure, aws, gcp, and typical api_key
        (
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "openai/gpt-4",
                    "api_key": "sk-sensitive-key-123",
                    "client_secret": "~v8Q4W:Zp9gJ-3sTqX5aB@LkR2mNfYdC",
                    "vertex_credentials": {"type": "service_account"},
                    "aws_access_key_id": "AKIA123456789",
                    "aws_secret_access_key": "secret-access-key",
                    "api_base": "https://api.openai.com/v1",
                    "temperature": 0.7,
                },
                "model_info": {"id": "test-id"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "openai/gpt-4",
                    "api_base": "https://api.openai.com/v1",
                    "temperature": 0.7,
                },
                "model_info": {"id": "test-id"},
            },
        ),
        # Test case 3: Partial sensitive data, api_key
        (
            {
                "model_name": "claude-3",
                "litellm_params": {
                    "model": "anthropic/claude-3",
                    "api_key": "sk-anthropic-key",
                    "temperature": 0.5,
                },
            },
            {
                "model_name": "claude-3",
                "litellm_params": {"model": "anthropic/claude-3", "temperature": 0.5},
            },
        ),
        # Test case 4: No sensitive data
        (
            {
                "model_name": "local-model",
                "litellm_params": {
                    "model": "local/model",
                    "temperature": 0.8,
                    "max_tokens": 100,
                },
            },
            {
                "model_name": "local-model",
                "litellm_params": {
                    "model": "local/model",
                    "temperature": 0.8,
                    "max_tokens": 100,
                },
            },
        ),
    ],
)
def test_remove_sensitive_info_from_deployment(
    model_config: dict, expected_config: dict
):
    sanitized_config = remove_sensitive_info_from_deployment(model_config)
    assert sanitized_config == expected_config
