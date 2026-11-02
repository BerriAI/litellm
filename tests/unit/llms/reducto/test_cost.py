from typing import Final

import pytest

import litellm

from litellm.cost_calculator import completion_cost
from litellm.llms.base_llm.ocr.transformation import OCRPage, OCRResponse, OCRUsageInfo


def test_ocr_cost_prefers_credit_pricing_when_pages_processed_is_none() -> None:
    model: Final = "litellm-test-credit-ocr"
    litellm.register_model(
        {f"reducto/{model}": {"litellm_provider": "reducto", "mode": "ocr", "ocr_cost_per_credit": 0.003}}
    )

    response: Final = OCRResponse(
        pages=[OCRPage(index=0, markdown="credit priced")],
        model=model,
        usage_info=OCRUsageInfo(pages_processed=None, credits=10),
    )

    cost: Final = completion_cost(
        completion_response=response,
        model=f"reducto/{model}",
        custom_llm_provider="reducto",
        call_type="ocr",
    )

    assert cost == 0.03


def test_ocr_cost_prefers_zero_credit_pricing_over_page_pricing() -> None:
    model: Final = "litellm-test-zero-credit-ocr"
    litellm.register_model(
        {
            f"reducto/{model}": {
                "litellm_provider": "reducto",
                "mode": "ocr",
                "ocr_cost_per_credit": 0.0,
                "ocr_cost_per_page": 0.5,
            }
        }
    )

    response: Final = OCRResponse(
        pages=[OCRPage(index=0, markdown="free credit priced")],
        model=model,
        usage_info=OCRUsageInfo(pages_processed=2, credits=10),
    )

    cost: Final = completion_cost(
        completion_response=response,
        model=f"reducto/{model}",
        custom_llm_provider="reducto",
        call_type="ocr",
    )

    assert cost == 0.0


def test_ocr_cost_falls_back_to_page_pricing() -> None:
    model: Final = "litellm-test-page-ocr"
    litellm.register_model(
        {f"mistral/{model}": {"litellm_provider": "mistral", "mode": "ocr", "ocr_cost_per_page": 0.5}}
    )

    response: Final = OCRResponse(
        pages=[OCRPage(index=0, markdown="page priced")],
        model=model,
        usage_info=OCRUsageInfo(pages_processed=2),
    )

    cost: Final = completion_cost(
        completion_response=response,
        model=f"mistral/{model}",
        custom_llm_provider="mistral",
        call_type="ocr",
    )

    assert cost == 1.0


def test_ocr_cost_returns_zero_when_no_pricing_and_no_pages() -> None:
    model: Final = "litellm-test-unpriced-ocr"
    litellm.register_model({f"reducto/{model}": {"litellm_provider": "reducto", "mode": "ocr"}})

    response: Final = OCRResponse(
        pages=[OCRPage(index=0, markdown="unpriced")],
        model=model,
        usage_info=OCRUsageInfo(pages_processed=None, credits=5),
    )

    cost: Final = completion_cost(
        completion_response=response,
        model=f"reducto/{model}",
        custom_llm_provider="reducto",
        call_type="ocr",
    )

    assert cost == 0.0


def test_ocr_cost_raises_when_pages_processed_missing_for_page_pricing() -> None:
    model: Final = "litellm-test-missing-page-ocr"
    litellm.register_model(
        {f"mistral/{model}": {"litellm_provider": "mistral", "mode": "ocr", "ocr_cost_per_page": 0.5}}
    )

    response: Final = OCRResponse(
        pages=[OCRPage(index=0, markdown="missing pages")],
        model=model,
        usage_info=OCRUsageInfo(pages_processed=None),
    )

    with pytest.raises(ValueError, match="OCR response pages_processed is None"):
        completion_cost(
            completion_response=response,
            model=f"mistral/{model}",
            custom_llm_provider="mistral",
            call_type="ocr",
        )
