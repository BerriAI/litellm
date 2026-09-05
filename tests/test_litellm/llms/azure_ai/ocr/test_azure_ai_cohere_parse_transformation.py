import base64
import json

import pytest

import litellm
from litellm.llms.azure_ai.ocr.cohere_parse_transformation import AzureAICohereParseConfig
from litellm.llms.azure_ai.ocr.common_utils import get_azure_ai_ocr_config
from litellm.llms.azure_ai.ocr.document_intelligence.transformation import AzureDocumentIntelligenceOCRConfig
from litellm.llms.azure_ai.ocr.transformation import AzureAIOCRConfig

MODEL = "azure_ai/Cohere-parse-v5"
API_BASE = "https://resource.services.ai.azure.com"
PARSE_URL = f"{API_BASE}/providers/cohere/v2/parse"
IMAGE_URL = "https://example.com/receipt.png"
PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)
PNG_DATA_URI = f"data:image/png;base64,{base64.b64encode(PNG_BYTES).decode()}"


def _parse_response() -> dict:
    return {
        "id": "882bf973-9dfa-4d02-9d30-709247008efd",
        "pages": [{"index": 0, "type": "markdown", "markdown": {"content": "# Receipt\n\nTotal Due: $4.00"}}],
        "meta": {"api_version": {"version": "2"}, "billed_units": {"pages": 1}},
    }


@pytest.fixture()
def disable_aiohttp_transport(monkeypatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    yield
    litellm.in_memory_llm_clients_cache.flush_cache()


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


@pytest.mark.asyncio
async def test_aocr_inlines_remote_image_and_posts_to_foundry(disable_aiohttp_transport, respx_mock):
    respx_mock.get(IMAGE_URL).respond(content=PNG_BYTES, headers={"Content-Type": "image/png"})
    route = respx_mock.post(PARSE_URL).respond(json=_parse_response())

    response = await litellm.aocr(
        model=MODEL,
        document={"type": "image_url", "image_url": IMAGE_URL},
        api_base=API_BASE,
        api_key="azure-key",
    )

    request = route.calls.last.request
    assert request.headers["Authorization"] == "Bearer azure-key"
    assert json.loads(request.content) == {
        "model": "Cohere-parse-v5",
        "document": {"type": "image_url", "image_url": PNG_DATA_URI},
        "output_format": "markdown",
    }
    assert response.pages[0].markdown == "# Receipt\n\nTotal Due: $4.00"
    assert response.usage_info.pages_processed == 1


@pytest.mark.asyncio
async def test_aocr_passes_data_uri_through_without_fetching(disable_aiohttp_transport, respx_mock):
    route = respx_mock.post(PARSE_URL).respond(json=_parse_response())

    await litellm.aocr(
        model=MODEL,
        document={"type": "image_url", "image_url": PNG_DATA_URI},
        api_base=API_BASE,
        api_key="azure-key",
        output_format="blocks",
    )

    body = json.loads(route.calls.last.request.content)
    assert body["document"]["image_url"] == PNG_DATA_URI
    assert body["output_format"] == "blocks"


def test_ocr_sync_inlines_remote_image(respx_mock):
    respx_mock.get(IMAGE_URL).respond(content=PNG_BYTES, headers={"Content-Type": "image/png"})
    route = respx_mock.post(PARSE_URL).respond(json=_parse_response())

    response = litellm.ocr(
        model=MODEL,
        document={"type": "image_url", "image_url": IMAGE_URL},
        api_base=API_BASE,
        api_key="azure-key",
    )

    assert json.loads(route.calls.last.request.content)["document"]["image_url"] == PNG_DATA_URI
    assert response.pages[0].markdown == "# Receipt\n\nTotal Due: $4.00"


@pytest.mark.asyncio
async def test_aocr_rejects_pdf_before_calling_foundry(disable_aiohttp_transport, respx_mock):
    route = respx_mock.post(PARSE_URL).respond(json=_parse_response())

    with pytest.raises(litellm.BadRequestError, match="only accepts `image_url` documents") as exc_info:
        await litellm.aocr(
            model=MODEL,
            document={"type": "document_url", "document_url": "https://example.com/doc.pdf"},
            api_base=API_BASE,
            api_key="azure-key",
        )

    assert exc_info.value.llm_provider == "azure_ai"
    assert not route.called


@pytest.mark.asyncio
async def test_ahealth_check_ocr_sends_an_image_to_the_foundry_cohere_parse_deployment(
    disable_aiohttp_transport, respx_mock
):
    route = respx_mock.post(PARSE_URL).respond(json=_parse_response())

    result = await litellm.ahealth_check(
        model_params={"model": MODEL, "api_base": API_BASE, "api_key": "test-key"}, mode="ocr"
    )

    document = json.loads(route.calls.last.request.content)["document"]
    assert document["type"] == "image_url"
    assert document["image_url"].startswith("data:image/png;base64,")
    assert "error" not in result
