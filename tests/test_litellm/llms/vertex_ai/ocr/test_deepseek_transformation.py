from unittest.mock import MagicMock

import httpx
import pytest

from litellm.cost_calculator import completion_cost
from litellm.llms.vertex_ai.ocr.deepseek_transformation import VertexAIDeepSeekOCRConfig

PROMPT_TOKENS = 901
COMPLETION_TOKENS = 212
INPUT_COST_PER_TOKEN = 3e-07
OUTPUT_COST_PER_TOKEN = 1.2e-06


def _deepseek_chat_response() -> httpx.Response:
    return httpx.Response(
        status_code=200,
        json={
            "choices": [{"message": {"role": "assistant", "content": "# OCR text"}}],
            "usage": {
                "prompt_tokens": PROMPT_TOKENS,
                "completion_tokens": COMPLETION_TOKENS,
                "total_tokens": PROMPT_TOKENS + COMPLETION_TOKENS,
            },
        },
        request=httpx.Request("POST", "https://us-central1-aiplatform.googleapis.com"),
    )


@pytest.mark.parametrize("model", ["deepseek-ocr-maas", "deepseek-ai/deepseek-ocr-maas"])
def test_response_is_priced_from_token_usage_for_either_model_name(local_model_cost_map: None, model: str) -> None:
    response = VertexAIDeepSeekOCRConfig().transform_ocr_response(
        model=model,
        raw_response=_deepseek_chat_response(),
        logging_obj=MagicMock(),
    )

    cost = completion_cost(
        completion_response=response,
        model=f"vertex_ai/{model}",
        custom_llm_provider="vertex_ai",
        call_type="ocr",
    )

    assert response.model == "deepseek-ai/deepseek-ocr-maas"
    assert cost == pytest.approx(PROMPT_TOKENS * INPUT_COST_PER_TOKEN + COMPLETION_TOKENS * OUTPUT_COST_PER_TOKEN)
    assert cost > 0


def test_json_content_without_pages_reports_the_canonical_model(local_model_cost_map: None) -> None:
    raw_response = httpx.Response(
        200,
        json={
            "choices": [{"message": {"role": "assistant", "content": '{"text": "# OCR text"}'}}],
            "usage": {"prompt_tokens": PROMPT_TOKENS, "completion_tokens": COMPLETION_TOKENS},
        },
        request=httpx.Request("POST", "https://example.invalid"),
    )

    response = VertexAIDeepSeekOCRConfig().transform_ocr_response(
        model="deepseek-ocr-maas",
        raw_response=raw_response,
        logging_obj=MagicMock(),
    )

    assert response.model == "deepseek-ai/deepseek-ocr-maas"
    assert response.pages[0].markdown == '{"text": "# OCR text"}'
