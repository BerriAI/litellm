import litellm

from litellm.cost_calculator import completion_cost
from litellm.llms.base_llm.ocr.transformation import OCRPage, OCRResponse, OCRUsageInfo


def test_ocr_cost_prefers_credit_pricing_when_pages_processed_is_none(monkeypatch):
    monkeypatch.setattr(
        litellm,
        "get_model_info",
        lambda model, custom_llm_provider=None: {"ocr_cost_per_credit": 0.003},
    )

    response = OCRResponse(
        pages=[OCRPage(index=0, markdown="credit priced")],
        model="parse-v3",
        usage_info=OCRUsageInfo(pages_processed=None, credits=10),
    )

    cost = completion_cost(
        completion_response=response,
        model="reducto/parse-v3",
        custom_llm_provider="reducto",
        call_type="ocr",
    )

    assert cost == 0.03


def test_ocr_cost_falls_back_to_page_pricing(monkeypatch):
    monkeypatch.setattr(
        litellm,
        "get_model_info",
        lambda model, custom_llm_provider=None: {"ocr_cost_per_page": 0.5},
    )

    response = OCRResponse(
        pages=[OCRPage(index=0, markdown="page priced")],
        model="mistral-ocr-latest",
        usage_info=OCRUsageInfo(pages_processed=2),
    )

    cost = completion_cost(
        completion_response=response,
        model="mistral/mistral-ocr-latest",
        custom_llm_provider="mistral",
        call_type="ocr",
    )

    assert cost == 1.0


def test_ocr_cost_returns_zero_when_no_pricing_and_no_pages(monkeypatch):
    monkeypatch.setattr(
        litellm,
        "get_model_info",
        lambda model, custom_llm_provider=None: {},
    )

    response = OCRResponse(
        pages=[OCRPage(index=0, markdown="unpriced")],
        model="parse-v3",
        usage_info=OCRUsageInfo(pages_processed=None, credits=5),
    )

    cost = completion_cost(
        completion_response=response,
        model="reducto/parse-v3",
        custom_llm_provider="reducto",
        call_type="ocr",
    )

    assert cost == 0.0
