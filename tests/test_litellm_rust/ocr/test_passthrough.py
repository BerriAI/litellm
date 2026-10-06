import json
from datetime import datetime
from typing import Final

import httpx
import pytest

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.azure_ai.passthrough.transformation import AzureAIPassthroughConfig
from litellm.llms.base_llm.ocr.transformation import OCRResponse

pytestmark = pytest.mark.requires_rust_extension
FOUNDRY_BASE: Final = "https://my-resource.services.ai.azure.com"
MISTRAL_BODY: Final = {
    "pages": [{"index": 0, "markdown": "page one"}, {"index": 1, "markdown": "page two"}],
    "model": "mistral-document-ai-2512",
    "usage_info": {"pages_processed": 2, "doc_size_bytes": 4321},
}
COHERE_BODY: Final = {"id": "parse-1", "pages": [], "meta": {"billed_units": {"pages": 3}}}


def _relay(model: str, native_path: str, body: object) -> tuple[object, Logging]:
    logging_obj: Final = Logging(
        model=model,
        messages=[],
        stream=False,
        call_type="allm_passthrough_route",
        start_time=datetime.now(),
        litellm_call_id="call-1",
        function_id="fn-1",
    )
    logging_obj.update_environment_variables(
        model=model,
        litellm_params={"api_base": FOUNDRY_BASE, "custom_llm_provider": "azure_ai"},
        optional_params={},
        custom_llm_provider="azure_ai",
    )
    response: Final = httpx.Response(
        status_code=200,
        headers={"content-type": "application/json"},
        content=json.dumps(body).encode(),
        request=httpx.Request("POST", f"{FOUNDRY_BASE}/{native_path}?api-version=2024-05-01-preview"),
    )
    result: Final = AzureAIPassthroughConfig().logging_non_streaming_response(
        model=model,
        custom_llm_provider="azure_ai",
        httpx_response=response,
        request_data={"model": model},
        logging_obj=logging_obj,
        endpoint=f"{model}/{native_path}",
    )
    return result, logging_obj


@pytest.mark.parametrize(
    ("model", "native_path", "body", "pages"),
    [
        ("mistral-document-ai-2512", "providers/mistral/azure/ocr", MISTRAL_BODY, 2),
        ("Cohere-parse-v5", "providers/cohere/v2/parse", COHERE_BODY, 3),
    ],
)
def test_ocr_relay_is_costed_per_page(model: str, native_path: str, body: object, pages: int) -> None:
    result, logging_obj = _relay(model, native_path, body)
    per_page: Final = litellm.get_model_info(f"azure_ai/{model}")["ocr_cost_per_page"]

    assert isinstance(result, OCRResponse)
    assert result.usage_info is not None
    assert result.usage_info.pages_processed == pages
    assert logging_obj.call_type == "aocr"
    assert per_page is not None and per_page > 0
    assert logging_obj._response_cost_calculator(result=result) == pytest.approx(pages * per_page)  # pyright: ignore[reportPrivateUsage]  # the per-page cost path is what the relay routes into


@pytest.mark.parametrize(
    ("native_path", "body", "logged"),
    [
        ("models/info", {"name": "mistral-document-ai-2512"}, {"name": "mistral-document-ai-2512"}),
        ("providers/mistral/azure/ocr", ["not", "an", "ocr", "body"], '["not", "an", "ocr", "body"]'),
    ],
)
def test_non_ocr_relay_keeps_the_passthrough_object(native_path: str, body: object, logged: object) -> None:
    result, logging_obj = _relay("mistral-document-ai-2512", native_path, body)

    assert result == {"response": logged}
    assert logging_obj.call_type == "allm_passthrough_route"
