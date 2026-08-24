
import pytest


import litellm
from litellm.llms.azure_ai.rerank.transformation import AzureAIRerankConfig


class TestAzureAIRerankConfigGetCompleteUrl:
    def setup_method(self):
        self.config = AzureAIRerankConfig()
        self.model = "azure_ai/cohere-rerank-v3-english"

    def test_api_base_required(self):
        with pytest.raises(ValueError, match='Azure AI API Base is required\\. api_base=None\\. Set in') as exc_info:
            self.config.get_complete_url(api_base=None, model=self.model)

        assert "api_base=None" in str(exc_info.value)

    @pytest.mark.parametrize(
        "api_base",
        [
            "example.com",
            "example.com/v1",
            "//example.com/v1",
            "/v1/rerank",
        ],
    )
    def test_api_base_requires_scheme(self, api_base):
        with pytest.raises(ValueError, match='Azure AI API Base must be an absolute URL including scheme') as exc_info:
            self.config.get_complete_url(api_base=api_base, model=self.model)

        error_message = str(exc_info.value).lower()
        assert "absolute url" in error_message
        assert "scheme" in error_message

    @pytest.mark.parametrize(
        "api_base, expected_url",
        [
            (
                "https://my-resource.services.ai.azure.com/v1/rerank/",
                "https://my-resource.services.ai.azure.com/v1/rerank",
            ),
            (
                "https://my-resource.services.ai.azure.com/providers/cohere/v2/rerank/",
                "https://my-resource.services.ai.azure.com/providers/cohere/v2/rerank",
            ),
        ],
    )
    def test_preserves_full_rerank_endpoint(self, api_base, expected_url):
        url = self.config.get_complete_url(api_base=api_base, model=self.model)
        assert url == expected_url

    @pytest.mark.parametrize(
        "api_base, expected_url",
        [
            (
                "https://my-resource.services.ai.azure.com/v1",
                "https://my-resource.services.ai.azure.com/v1/rerank",
            ),
            (
                "https://my-resource.services.ai.azure.com/v2/",
                "https://my-resource.services.ai.azure.com/v2/rerank",
            ),
            (
                "https://my-resource.services.ai.azure.com/providers/cohere/v2",
                "https://my-resource.services.ai.azure.com/providers/cohere/v2/rerank",
            ),
            (
                "https://my-resource.services.ai.azure.com/providers/cohere/v2/",
                "https://my-resource.services.ai.azure.com/providers/cohere/v2/rerank",
            ),
        ],
    )
    def test_appends_rerank_for_version_paths(self, api_base, expected_url):
        url = self.config.get_complete_url(api_base=api_base, model=self.model)
        assert url == expected_url

    @pytest.mark.parametrize(
        "api_base",
        [
            "https://my-resource.services.ai.azure.com",
            "https://my-resource.services.ai.azure.com/",
        ],
    )
    def test_defaults_to_v1_rerank_when_base_has_no_path(self, api_base):
        url = self.config.get_complete_url(api_base=api_base, model=self.model)
        assert url == "https://my-resource.services.ai.azure.com/v1/rerank"

    def test_preserves_query_params(self):
        url = self.config.get_complete_url(
            api_base="https://my-resource.services.ai.azure.com/v1?r=1",
            model=self.model,
        )
        assert url == "https://my-resource.services.ai.azure.com/v1/rerank?r=1"


class TestAzureAIRerankConfigValidateEnvironment:
    def test_uses_api_key_when_set(self):
        headers = AzureAIRerankConfig().validate_environment(
            headers={},
            model="azure_ai/cohere-rerank-v3-english",
            api_key="my-key",
        )

        assert headers["Authorization"] == "Bearer my-key"

    def test_falls_back_to_entra_token(self, monkeypatch):
        monkeypatch.delenv("AZURE_AI_API_KEY", raising=False)
        monkeypatch.setattr(litellm, "azure_key", None)

        headers = AzureAIRerankConfig().validate_environment(
            headers={},
            model="azure_ai/cohere-rerank-v3-english",
            litellm_params={"azure_ad_token": "entra-token"},
        )

        assert headers["Authorization"] == "Bearer entra-token"
