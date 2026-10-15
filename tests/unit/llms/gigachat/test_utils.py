"""
Tests for litellm.llms.gigachat.utils
"""

import pytest
from litellm.llms.gigachat.chat.transformation import GigaChatConfig
from litellm.llms.gigachat.embedding.transformation import GigaChatEmbeddingConfig
from litellm.llms.gigachat.utils import convert_usage
from litellm.types.utils import PromptTokensDetailsWrapper, Usage


class TestConvertUsage:
    def test_basic_usage_without_precached(self):
        """Test convert_usage with standard tokens, no precached prompt tokens."""
        result = convert_usage(
            {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            }
        )

        assert result == Usage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            prompt_tokens_details=None,
        )

    def test_usage_with_precached_prompt_tokens(self):
        """GigaChat's prompt_tokens and total_tokens exclude cached tokens (docs example:
        prompt_tokens=1, precached_prompt_tokens=37, total_tokens=5), so OpenAI-convention
        usage adds precached back in and surfaces it as cached_tokens."""
        result = convert_usage(
            {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "precached_prompt_tokens": 3,
                "total_tokens": 15,
            }
        )

        assert result == Usage(
            prompt_tokens=13,
            completion_tokens=5,
            total_tokens=18,
            prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=3),
        )

    def test_zero_precached_prompt_tokens(self):
        """Test convert_usage with zero precached_prompt_tokens does not create details wrapper."""
        result = convert_usage(
            {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "precached_prompt_tokens": 0,
                "total_tokens": 15,
            }
        )

        assert result == Usage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            prompt_tokens_details=None,
        )

    def test_missing_optional_fields(self):
        """Test convert_usage with missing optional fields defaults to zero."""
        result = convert_usage(
            {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            }
        )

        assert result.prompt_tokens == 10
        assert result.completion_tokens == 5
        assert result.total_tokens == 15
        assert result.prompt_tokens_details is None


@pytest.mark.parametrize(
    "api_base", ["https://gateway.example.com/gigachat/api/v1", "https://gateway.example.com/gigachat/api/v1/"]
)
@pytest.mark.parametrize(
    ("config", "route"),
    [(GigaChatConfig(), "chat/completions"), (GigaChatEmbeddingConfig(), "embeddings")],
)
def test_get_complete_url_ignores_a_trailing_slash_on_api_base(
    api_base: str, config: GigaChatConfig | GigaChatEmbeddingConfig, route: str
) -> None:
    assert (
        config.get_complete_url(
            api_base=api_base, api_key=None, model="GigaChat", optional_params={}, litellm_params={}
        )
        == f"https://gateway.example.com/gigachat/api/v1/{route}"
    )
