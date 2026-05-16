from unittest.mock import MagicMock, patch

import pytest

from litellm.llms.ollama.common_utils import OllamaModelInfo
from litellm.llms.deepseek.chat.transformation import DeepSeekChatConfig
from litellm.llms.openrouter.chat.transformation import OpenrouterConfig


def test_openrouter_get_models_uses_openrouter_endpoint():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": [
            {"id": "openai/gpt-5"},
            {"id": "anthropic/claude-sonnet-4"},
        ]
    }

    with patch(
        "litellm.module_level_client.get", return_value=mock_response
    ) as mock_get:
        models = OpenrouterConfig().get_models(api_key="sk-or-test")

    assert models == ["openai/gpt-5", "anthropic/claude-sonnet-4"]
    mock_get.assert_called_once_with(
        url="https://openrouter.ai/api/v1/models",
        headers={"Authorization": "Bearer sk-or-test"},
    )


def test_openrouter_get_models_uses_custom_base_and_secret_key():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"data": [{"id": "openrouter/custom-model"}]}

    with (
        patch("litellm.get_secret", return_value="sk-or-secret"),
        patch(
            "litellm.module_level_client.get", return_value=mock_response
        ) as mock_get,
    ):
        models = OpenrouterConfig().get_models(api_base="https://example.com/v1/")

    assert models == ["openrouter/custom-model"]
    mock_get.assert_called_once_with(
        url="https://example.com/v1/models",
        headers={"Authorization": "Bearer sk-or-secret"},
    )


def test_openrouter_get_models_raises_for_non_200_response():
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.text = "unauthorized"

    with patch("litellm.module_level_client.get", return_value=mock_response):
        with pytest.raises(Exception, match="Failed to get models: unauthorized"):
            OpenrouterConfig().get_models(api_key="sk-or-test")


def test_deepseek_get_models_uses_deepseek_endpoint():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": [
            {"id": "deepseek-chat"},
            {"id": "deepseek-reasoner"},
        ]
    }

    with patch(
        "litellm.module_level_client.get", return_value=mock_response
    ) as mock_get:
        models = DeepSeekChatConfig().get_models(api_key="sk-deepseek-test")

    assert models == ["deepseek-chat", "deepseek-reasoner"]
    mock_get.assert_called_once_with(
        url="https://api.deepseek.com/models",
        headers={"Authorization": "Bearer sk-deepseek-test"},
    )


def test_deepseek_get_models_uses_custom_base_and_secret_key():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"data": [{"id": "deepseek-custom"}]}

    with (
        patch(
            "litellm.llms.deepseek.chat.transformation.get_secret_str",
            return_value="sk-deepseek-secret",
        ),
        patch(
            "litellm.module_level_client.get", return_value=mock_response
        ) as mock_get,
    ):
        models = DeepSeekChatConfig().get_models(api_base="https://example.com/")

    assert models == ["deepseek-custom"]
    mock_get.assert_called_once_with(
        url="https://example.com/models",
        headers={"Authorization": "Bearer sk-deepseek-secret"},
    )


def test_deepseek_get_models_raises_for_non_200_response():
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "server error"

    with patch("litellm.module_level_client.get", return_value=mock_response):
        with pytest.raises(Exception, match="Failed to get models: server error"):
            DeepSeekChatConfig().get_models(api_key="sk-deepseek-test")


def test_ollama_get_models_uses_passed_api_key():
    mock_response = MagicMock()
    mock_response.json.return_value = {"models": [{"name": "llama3.1"}]}

    with patch("httpx.get", return_value=mock_response) as mock_get:
        models = OllamaModelInfo().get_models(
            api_key="ollama-key", api_base="https://ollama.example.com"
        )

    assert models == ["llama3.1"]
    mock_response.raise_for_status.assert_called_once()
    mock_get.assert_called_once_with(
        "https://ollama.example.com/api/tags",
        headers={"Authorization": "Bearer ollama-key"},
    )
