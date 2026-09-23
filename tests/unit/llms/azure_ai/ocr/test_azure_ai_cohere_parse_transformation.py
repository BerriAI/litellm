import pytest

from litellm.llms.azure_ai.ocr.cohere_parse_transformation import AzureAICohereParseConfig
from litellm.llms.azure_ai.ocr.common_utils import get_azure_ai_ocr_config
from litellm.llms.azure_ai.ocr.document_intelligence.transformation import AzureDocumentIntelligenceOCRConfig
from litellm.llms.azure_ai.ocr.transformation import AzureAIOCRConfig

MODEL = "azure_ai/Cohere-parse-v5"
API_BASE = "https://resource.services.ai.azure.com"
PARSE_URL = f"{API_BASE}/providers/cohere/v2/parse"


@pytest.mark.parametrize(
    "model, expected_config",
    [
        ("Cohere-parse-v5", AzureAICohereParseConfig),
        ("cohere-parse-v5", AzureAICohereParseConfig),
        ("cohere/parse-v5", AzureAICohereParseConfig),
        ("invoice-parser", AzureAIOCRConfig),
        ("parse-v5", AzureAIOCRConfig),
        ("mistral-ocr-4-0", AzureAIOCRConfig),
        ("mistral-document-ai-2512", AzureAIOCRConfig),
        ("doc-intelligence/prebuilt-read", AzureDocumentIntelligenceOCRConfig),
    ],
)
def test_azure_ai_ocr_routing(model: str, expected_config: type) -> None:
    assert type(get_azure_ai_ocr_config(model)) is expected_config


@pytest.mark.parametrize(
    "api_base, expected_url",
    [
        (API_BASE, PARSE_URL),
        (f"{API_BASE}/", PARSE_URL),
        (f"{API_BASE}/models", PARSE_URL),
        (f"{API_BASE}/providers/cohere/v2", PARSE_URL),
        (f"{API_BASE}/providers/cohere/v2/parse", PARSE_URL),
    ],
)
def test_get_complete_url_targets_the_cohere_provider_route(api_base: str, expected_url: str) -> None:
    url = AzureAICohereParseConfig().get_complete_url(api_base=api_base, model="Cohere-parse-v5", optional_params={})

    assert url == expected_url


def test_get_complete_url_falls_back_to_env_api_base(monkeypatch) -> None:
    monkeypatch.setenv("AZURE_AI_API_BASE", API_BASE)

    url = AzureAICohereParseConfig().get_complete_url(api_base=None, model="Cohere-parse-v5", optional_params={})

    assert url == PARSE_URL


def test_get_complete_url_requires_api_base(monkeypatch) -> None:
    monkeypatch.delenv("AZURE_AI_API_BASE", raising=False)

    with pytest.raises(ValueError, match="AZURE_AI_API_BASE"):
        AzureAICohereParseConfig().get_complete_url(api_base=None, model="Cohere-parse-v5", optional_params={})


def test_get_complete_url_rejects_relative_api_base() -> None:
    with pytest.raises(ValueError, match="absolute URL"):
        AzureAICohereParseConfig().get_complete_url(
            api_base="resource.services.ai.azure.com", model="Cohere-parse-v5", optional_params={}
        )


def test_validate_environment_requires_api_base(monkeypatch) -> None:
    monkeypatch.delenv("AZURE_AI_API_BASE", raising=False)

    with pytest.raises(ValueError, match="AZURE_AI_API_BASE"):
        AzureAICohereParseConfig().validate_environment(headers={}, model="Cohere-parse-v5", api_key="key")
