"""
Tests for the discovery URL built by OpenAIGPTConfig.get_models
(litellm/llms/openai/chat/gpt_transformation.py)

Regression coverage: api_base values carrying a path (e.g. merge's
https://api-gateway.merge.dev/v1/openai) must probe {api_base}/models
instead of being stripped to {host}/v1/models.
"""

from unittest.mock import MagicMock, patch

import pytest

from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig


@pytest.mark.parametrize(
    "api_base,expected_url",
    [
        (None, "https://api.openai.com/v1/models"),
        ("https://api.openai.com", "https://api.openai.com/v1/models"),
        ("https://api.openai.com/v1", "https://api.openai.com/v1/models"),
        (
            "https://api-gateway.merge.dev/v1/openai",
            "https://api-gateway.merge.dev/v1/openai/models",
        ),
        ("http://localhost:8080", "http://localhost:8080/v1/models"),
        ("http://localhost:8080/v1", "http://localhost:8080/v1/models"),
    ],
)
def test_get_models_probes_path_preserving_url(api_base, expected_url):
    config = OpenAIGPTConfig()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"data": [{"id": "m1"}]}

    with patch("litellm.module_level_client.get", return_value=mock_response) as mock_get:
        models = config.get_models(api_key="sk-test", api_base=api_base)

    assert mock_get.call_args.kwargs["url"] == expected_url
    assert models == ["m1"]
