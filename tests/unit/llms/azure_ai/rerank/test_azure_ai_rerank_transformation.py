
from typing import Final

import httpx
import pytest
import respx

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

    def test_falls_back_to_entra_token(self, monkeypatch, no_ambient_azure_credentials):
        monkeypatch.delenv("AZURE_AI_API_KEY", raising=False)
        monkeypatch.setattr(litellm, "azure_key", None)

        headers = AzureAIRerankConfig().validate_environment(
            headers={},
            model="azure_ai/cohere-rerank-v3-english",
            litellm_params={"azure_ad_token": "entra-token"},
        )

        assert headers["Authorization"] == "Bearer entra-token"


AZURE_AI_RERANK_BASE: Final = "https://my-resource.services.ai.azure.com"
COHERE_RERANK_RESPONSE: Final = {
    "id": "rerank-1",
    "results": [{"index": 0, "relevance_score": 0.9}],
    "meta": {"billed_units": {"search_units": 1}},
}


@pytest.fixture
def process_wide_azure_ai_key(monkeypatch: pytest.MonkeyPatch, no_ambient_azure_credentials: None) -> None:
    monkeypatch.setenv("AZURE_AI_API_KEY", "process-wide-key")
    monkeypatch.setattr(litellm, "azure_key", None)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)


def _rerank(respx_mock: respx.MockRouter, **credentials: str) -> httpx.Request:
    route: Final = respx_mock.post(f"{AZURE_AI_RERANK_BASE}/v1/rerank").mock(
        return_value=httpx.Response(200, json=COHERE_RERANK_RESPONSE)
    )
    litellm.rerank(
        model="azure_ai/cohere-rerank-v3.5",
        query="hello",
        documents=["hello world"],
        api_base=AZURE_AI_RERANK_BASE,
        **credentials,
    )
    return route.calls.last.request


@pytest.mark.usefixtures("process_wide_azure_ai_key")
def test_rerank_authenticates_with_the_entry_entra_token_over_the_process_wide_key(
    respx_mock: respx.MockRouter,
) -> None:
    assert _rerank(respx_mock, azure_ad_token="entry-token").headers["authorization"] == "Bearer entry-token"


@pytest.mark.usefixtures("process_wide_azure_ai_key")
def test_rerank_without_entry_entra_still_uses_the_process_wide_key(respx_mock: respx.MockRouter) -> None:
    assert _rerank(respx_mock).headers["authorization"] == "Bearer process-wide-key"


@pytest.mark.usefixtures("process_wide_azure_ai_key")
def test_rerank_explicit_api_key_wins_over_the_entry_entra_token(respx_mock: respx.MockRouter) -> None:
    request: Final = _rerank(respx_mock, api_key="entry-key", azure_ad_token="entry-token")

    assert request.headers["authorization"] == "Bearer entry-key"


def test_rerank_falls_back_to_the_azure_key_without_entry_entra(
    monkeypatch: pytest.MonkeyPatch, no_ambient_azure_credentials: None
) -> None:
    monkeypatch.setattr(litellm, "azure_key", "process-wide-azure-key")

    headers: Final = AzureAIRerankConfig().validate_environment(headers={}, model="azure_ai/cohere-rerank-v3.5")

    assert headers["Authorization"] == "Bearer process-wide-azure-key"
