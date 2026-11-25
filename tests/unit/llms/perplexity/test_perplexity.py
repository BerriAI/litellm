

import pytest
import httpx
import respx

import litellm


class TestPerplexityWebSearch:
    """Test suite for Perplexity web search functionality."""

    @pytest.mark.parametrize("model", ["perplexity/sonar", "perplexity/sonar-pro"])
    def test_web_search_options_in_supported_params(self, model):
        """
        Test that web_search_options is in the list of supported parameters for Perplexity sonar models
        """
        from litellm.llms.perplexity.chat.transformation import PerplexityChatConfig

        config = PerplexityChatConfig()
        supported_params = config.get_supported_openai_params(model=model)

        assert (
            "web_search_options" in supported_params
        ), f"web_search_options should be supported for {model}"


def test_perplexity_401_maps_to_authentication_error(respx_mock: respx.MockRouter) -> None:
    route = respx_mock.post("https://api.perplexity.ai/chat/completions").mock(
        return_value=httpx.Response(401, json={"error": {"message": "invalid API key"}})
    )

    with pytest.raises(litellm.AuthenticationError) as exc_info:
        litellm.completion(
            model="perplexity/mistral-7b-instruct",
            messages=[{"role": "user", "content": "hello"}],
            api_key="bad-perplexity-key",
            max_retries=0,
        )

    assert exc_info.value.status_code == 401
    assert "PerplexityException" in str(exc_info.value)
    assert "invalid API key" in str(exc_info.value)
    assert route.calls.last.request.headers["Authorization"] == "Bearer bad-perplexity-key"
