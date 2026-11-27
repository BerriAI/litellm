from typing import Final

import litellm
import pytest

from litellm.cost_calculator import completion_cost
from litellm.llms.base_llm.ocr.transformation import OCRPage, OCRResponse, OCRUsageInfo


def _ocr_cost(provider: str, prices: dict[str, float], usage: OCRUsageInfo) -> float:
    model: Final = f"{provider}/ocr-cost-test-model"
    litellm.register_model({model: {"litellm_provider": provider, "mode": "ocr", **prices}}, persist_across_reloads=False)
    return completion_cost(
        completion_response=OCRResponse(
            pages=[OCRPage(index=0, markdown="page")], model="ocr-cost-test-model", usage_info=usage
        ),
        model=model,
        custom_llm_provider=provider,
        call_type="ocr",
    )


def test_ocr_cost_prefers_credit_pricing_when_pages_processed_is_none():
    cost: Final = _ocr_cost(
        "reducto", {"ocr_cost_per_credit": 0.003}, OCRUsageInfo(pages_processed=None, credits=10)
    )

    assert cost == 0.03


def test_ocr_cost_prefers_zero_credit_pricing_over_page_pricing():
    cost: Final = _ocr_cost(
        "reducto",
        {"ocr_cost_per_credit": 0.0, "ocr_cost_per_page": 0.5},
        OCRUsageInfo(pages_processed=2, credits=10),
    )

    assert cost == 0.0


def test_ocr_cost_falls_back_to_page_pricing():
    cost: Final = _ocr_cost("mistral", {"ocr_cost_per_page": 0.5}, OCRUsageInfo(pages_processed=2))

    assert cost == 1.0


def test_ocr_cost_returns_zero_when_no_pricing_and_no_pages():
    cost: Final = _ocr_cost("reducto", {}, OCRUsageInfo(pages_processed=None, credits=5))

    assert cost == 0.0


def test_ocr_cost_raises_when_pages_processed_missing_for_page_pricing():
    with pytest.raises(ValueError, match="OCR response pages_processed is None"):
        _ocr_cost("mistral", {"ocr_cost_per_page": 0.5}, OCRUsageInfo(pages_processed=None))
