from unittest.mock import MagicMock

import httpx
import pytest

from litellm.llms.azure_ai.ocr.cohere_parse_transformation import (
    AzureAICohereParseConfig,
)
from litellm.llms.azure_ai.ocr.common_utils import get_azure_ai_ocr_config
from litellm.llms.azure_ai.ocr.transformation import AzureAIOCRConfig


@pytest.fixture
def local_model_cost_map(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")

    import litellm

    return litellm.get_model_cost_map(url="")


def test_cohere_parse_uses_its_own_azure_ocr_config():
    assert isinstance(get_azure_ai_ocr_config("Cohere-parse-v5"), AzureAICohereParseConfig)
    assert type(get_azure_ai_ocr_config("parse-v5.0")) is AzureAIOCRConfig


@pytest.mark.parametrize(
    "api_base,expected",
    [
        (
            "https://example.services.ai.azure.com",
            "https://example.services.ai.azure.com/providers/cohere/v2/parse",
        ),
        (
            "https://region.api.cognitive.microsoft.com/models",
            "https://region.api.cognitive.microsoft.com/providers/cohere/v2/parse",
        ),
        (
            "https://cohere-parse-v5.region.models.ai.azure.com",
            "https://cohere-parse-v5.region.models.ai.azure.com/v2/parse",
        ),
        (
            "https://example.services.ai.azure.com/providers/cohere/v2/parse",
            "https://example.services.ai.azure.com/providers/cohere/v2/parse",
        ),
    ],
)
def test_cohere_parse_url(api_base: str, expected: str):
    config = AzureAICohereParseConfig()

    assert (
        config.get_complete_url(
            api_base=api_base,
            model="Cohere-parse-v5",
            optional_params={},
        )
        == expected
    )


def test_cohere_parse_uses_bearer_authentication():
    config = AzureAICohereParseConfig()

    headers = config.validate_environment(
        headers={},
        model="Cohere-parse-v5",
        api_key="test-key",
        api_base="https://example.services.ai.azure.com",
    )

    assert headers == {
        "Authorization": "Bearer test-key",
        "Content-Type": "application/json",
    }


def test_cohere_parse_validates_output_format():
    config = AzureAICohereParseConfig()

    assert config.map_ocr_params(
        non_default_params={"output_format": "blocks"},
        optional_params={},
        model="Cohere-parse-v5",
    ) == {"output_format": "blocks"}
    with pytest.raises(ValueError, match="either 'markdown' or 'blocks'"):
        config.map_ocr_params(
            non_default_params={"output_format": "json"},
            optional_params={},
            model="Cohere-parse-v5",
        )


def test_cohere_parse_request_uses_exact_azure_model_and_cohere_schema():
    config = AzureAICohereParseConfig()

    request = config.transform_ocr_request(
        model="Cohere-parse-v5",
        document={
            "type": "image_url",
            "image_url": "data:image/png;base64,aGVsbG8=",
        },
        optional_params={"output_format": "blocks"},
        headers={},
    )

    assert request.data == {
        "model": "Cohere-parse-v5",
        "document": {
            "type": "image_url",
            "image_url": "data:image/png;base64,aGVsbG8=",
        },
        "output_format": "blocks",
    }
    assert request.files is None


@pytest.mark.parametrize(
    "document,error",
    [
        (
            {"type": "document_url", "document_url": "https://example.com/document.pdf"},
            "document_url and PDF inputs are not supported",
        ),
        (
            {"type": "document_url", "document_url": "data:application/pdf;base64,JVBERi0="},
            "document_url and PDF inputs are not supported",
        ),
        (
            {"type": "image_url", "image_url": "data:application/pdf;base64,JVBERi0="},
            "does not support PDF inputs",
        ),
    ],
)
def test_cohere_parse_rejects_document_urls_and_pdfs(document: dict[str, str], error: str):
    config = AzureAICohereParseConfig()

    with pytest.raises(ValueError, match=error):
        config.transform_ocr_request(
            model="Cohere-parse-v5",
            document=document,
            optional_params={},
            headers={},
        )


@pytest.mark.asyncio
async def test_cohere_parse_async_rejects_document_urls():
    config = AzureAICohereParseConfig()

    with pytest.raises(ValueError, match="document_url and PDF inputs are not supported"):
        await config.async_transform_ocr_request(
            model="Cohere-parse-v5",
            document={"type": "document_url", "document_url": "https://example.com/document.pdf"},
            optional_params={},
            headers={},
        )


def test_cohere_parse_response_is_normalized_to_litellm_ocr():
    config = AzureAICohereParseConfig()
    native_response = {
        "id": "parse-123",
        "pages": [
            {
                "type": "markdown",
                "index": 0,
                "markdown": {
                    "content": "# Invoice",
                    "images": [
                        {
                            "id": "img-0",
                            "description": "Company logo",
                            "bounding_box": {"top_left_x": 1},
                        }
                    ],
                },
            }
        ],
        "meta": {"billed_units": {"pages": 1}},
    }
    raw_response = httpx.Response(
        status_code=200,
        json=native_response,
        request=httpx.Request("POST", "https://example.test/v2/parse"),
    )

    response = config.transform_ocr_response(
        model="Cohere-parse-v5",
        raw_response=raw_response,
        logging_obj=MagicMock(),
    )

    assert response.model == "Cohere-parse-v5"
    assert response.pages[0].markdown == "# Invoice"
    assert response.pages[0].images is not None
    assert response.pages[0].images[0].model_extra["id"] == "img-0"
    assert response.usage_info is not None
    assert response.usage_info.pages_processed == 1
    assert response.get_provider_native_response() == native_response


def test_cohere_parse_model_metadata(local_model_cost_map: dict):
    model_info = local_model_cost_map["azure_ai/Cohere-parse-v5"]

    assert model_info["litellm_provider"] == "azure_ai"
    assert model_info["mode"] == "ocr"
    assert model_info["max_input_tokens"] == 8192
    assert model_info["max_output_tokens"] == 64000
    assert model_info["ocr_cost_per_page"] == 0.0015
    assert model_info["deprecation_date"] == "2026-12-15"
    assert model_info["supported_modalities"] == ["image"]
    assert model_info["supported_output_modalities"] == ["text"]
