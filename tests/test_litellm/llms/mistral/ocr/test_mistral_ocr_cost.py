"""
Cost tests for Mistral OCR models against the real litellm cost map
(no monkeypatching of get_model_info). These regress the pricing entries
for mistral-ocr-4-0 and mistral-ocr-latest, which now both resolve to
OCR 4 at $4 / 1000 pages.
"""

import pytest

import litellm
from litellm.cost_calculator import completion_cost
from litellm.llms.base_llm.ocr.transformation import OCRPage, OCRResponse, OCRUsageInfo

OCR4_COST_PER_PAGE = 0.004


def _ocr_response(model: str, pages_processed: int) -> OCRResponse:
    return OCRResponse(
        pages=[OCRPage(index=i, markdown=f"page {i}") for i in range(pages_processed)],
        model=model,
        usage_info=OCRUsageInfo(pages_processed=pages_processed),
    )


@pytest.mark.parametrize("model", ["mistral-ocr-4-0", "mistral-ocr-latest"])
def test_model_info_ocr4_price(model: str) -> None:
    info = litellm.get_model_info(model=f"mistral/{model}", custom_llm_provider="mistral")
    assert info["ocr_cost_per_page"] == OCR4_COST_PER_PAGE


@pytest.mark.parametrize("model", ["mistral-ocr-4-0", "mistral-ocr-latest"])
@pytest.mark.parametrize("pages_processed", [1, 3, 10])
def test_ocr4_cost_scales_with_pages(model: str, pages_processed: int) -> None:
    cost = completion_cost(
        completion_response=_ocr_response(model, pages_processed),
        model=f"mistral/{model}",
        custom_llm_provider="mistral",
        call_type="ocr",
    )
    assert cost == pytest.approx(OCR4_COST_PER_PAGE * pages_processed)
