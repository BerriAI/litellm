from unittest.mock import MagicMock, patch

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

    with patch("litellm.module_level_client.get", return_value=mock_response) as mock_get:
        models = OpenrouterConfig().get_models(api_key="sk-or-test")

    assert models == ["openai/gpt-5", "anthropic/claude-sonnet-4"]
    mock_get.assert_called_once_with(
        url="https://openrouter.ai/api/v1/models",
        headers={"Authorization": "Bearer sk-or-test"},
    )


def test_deepseek_get_models_uses_deepseek_endpoint():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": [
            {"id": "deepseek-chat"},
            {"id": "deepseek-reasoner"},
        ]
    }

    with patch("litellm.module_level_client.get", return_value=mock_response) as mock_get:
        models = DeepSeekChatConfig().get_models(api_key="sk-deepseek-test")

    assert models == ["deepseek-chat", "deepseek-reasoner"]
    mock_get.assert_called_once_with(
        url="https://api.deepseek.com/models",
        headers={"Authorization": "Bearer sk-deepseek-test"},
    )
