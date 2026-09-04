from __future__ import annotations

import json
from collections.abc import Callable
from typing import Final, cast

import httpx

from litellm.llms.base_llm.ocr.transformation import OCRResponse
from litellm.llms.vertex_ai.ocr.deepseek_transformation import VertexAIDeepSeekOCRConfig

PROMPT_TOKENS: Final = 281
COMPLETION_TOKENS: Final = 6


def _response(content: str | dict[str, object] = "invoice 123") -> OCRResponse:
    transform_response: Final = cast(  # cast-ok: the legacy provider method has untyped variadic parameters
        Callable[..., OCRResponse], VertexAIDeepSeekOCRConfig().transform_ocr_response
    )
    return transform_response(
        model="deepseek-ai/deepseek-ocr-maas",
        raw_response=httpx.Response(
            status_code=200,
            json={
                "choices": [{"message": {"content": content}}],
                "usage": {
                    "prompt_tokens": PROMPT_TOKENS,
                    "completion_tokens": COMPLETION_TOKENS,
                    "total_tokens": PROMPT_TOKENS + COMPLETION_TOKENS,
                },
            },
        ),
        logging_obj=None,
    )


def test_deepseek_ocr_preserves_billable_token_usage() -> None:
    response: Final = _response()

    assert response.usage_info is not None
    assert response.usage_info.prompt_tokens == PROMPT_TOKENS
    assert response.usage_info.completion_tokens == COMPLETION_TOKENS
    assert response.usage_info.total_tokens == PROMPT_TOKENS + COMPLETION_TOKENS
    assert response.usage_info.pages_processed is None


def test_deepseek_ocr_uses_outer_usage_for_structured_content() -> None:
    response: Final = _response(
        json.dumps(
            {
                "pages": [{"index": 0, "markdown": "invoice 123"}],
                "usage_info": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
        )
    )

    assert response.pages[0].markdown == "invoice 123"
    assert response.usage_info is not None
    assert response.usage_info.prompt_tokens == PROMPT_TOKENS
    assert response.usage_info.completion_tokens == COMPLETION_TOKENS
    assert response.usage_info.total_tokens == PROMPT_TOKENS + COMPLETION_TOKENS


def test_deepseek_ocr_treats_invalid_json_as_markdown() -> None:
    response: Final = _response("{invalid json")

    assert response.pages[0].markdown == "{invalid json"
    assert response.usage_info is not None
    assert response.usage_info.total_tokens == PROMPT_TOKENS + COMPLETION_TOKENS
