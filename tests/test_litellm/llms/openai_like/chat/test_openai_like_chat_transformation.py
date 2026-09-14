from unittest.mock import MagicMock, patch

import litellm
import pytest
from litellm.llms.openai_like.chat.transformation import (
    CustomOpenAIChatConfig,
    OpenAILikeChatConfig,
)
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager, get_valid_models


def test_sanitize_usage_obj_handles_null_tokens():
    """
    Tests that _sanitize_usage_obj correctly converts None values for token counts to 0.
    """
    response_json = {
        "choices": [],
        "usage": {"prompt_tokens": None, "completion_tokens": 50, "total_tokens": None},
    }

    sanitized_json = OpenAILikeChatConfig._sanitize_usage_obj(response_json)

    # Assert
    assert sanitized_json["usage"]["prompt_tokens"] == 0
    assert sanitized_json["usage"]["completion_tokens"] == 50  # Should remain unchanged
    assert sanitized_json["usage"]["total_tokens"] == 0


def test_sanitize_usage_obj_no_usage():
    """
    Tests that the sanitizer handles cases where the 'usage' object is missing.
    """
    response_json = {"choices": []}

    sanitized_json = OpenAILikeChatConfig._sanitize_usage_obj(response_json)

    # Assert
    assert "usage" not in sanitized_json  # Should not add a usage key


def test_sanitize_usage_obj_valid_usage():
    """
    Tests that the sanitizer does not modify a valid usage object.
    """
    response_json = {
        "choices": [],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }

    # Create a copy to compare against
    original_json = response_json.copy()

    sanitized_json = OpenAILikeChatConfig._sanitize_usage_obj(response_json)

    # Assert
    assert sanitized_json == original_json  # The object should be unchanged


def test_get_valid_models_custom_openai():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "object": "list",
        "data": [
            {
                "id": "my-custom-model",
                "object": "model",
                "created": 1686935002,
                "owned_by": "organization-owner",
            },
        ],
    }

    with patch.object(litellm.module_level_client, "get", return_value=mock_response) as mock_get:
        valid_models = get_valid_models(
            check_provider_endpoint=True,
            custom_llm_provider="custom_openai",
            api_key="sk-1234",
            api_base="https://my-openai-compatible-endpoint/v1",
        )

    assert valid_models == ["custom_openai/my-custom-model"]
    mock_get.assert_called_once_with(
        url="https://my-openai-compatible-endpoint/v1/models",
        headers={"Authorization": "Bearer sk-1234"},
    )


def test_custom_openai_get_models_requires_api_base():
    with pytest.raises(
        ValueError,
        match="api_base must be set to discover models for the custom_openai provider",
    ):
        CustomOpenAIChatConfig().get_models(api_base=None)


def test_custom_openai_get_models_does_not_use_openai_api_key(monkeypatch):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"data": [{"id": "my-custom-model"}]}
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-openai-api-key")

    with patch.object(litellm.module_level_client, "get", return_value=mock_response) as mock_get:
        models = CustomOpenAIChatConfig().get_models(
            api_base="https://my-openai-compatible-endpoint/v1"
        )

    assert models == ["custom_openai/my-custom-model"]
    mock_get.assert_called_once_with(
        url="https://my-openai-compatible-endpoint/v1/models",
        headers={"Authorization": "Bearer "},
    )


def test_openai_model_info_uses_openai_config():
    config = ProviderConfigManager.get_provider_model_info(
        model=None,
        provider=LlmProviders.OPENAI,
    )

    assert type(config) is OpenAIGPTConfig
