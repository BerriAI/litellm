from typing import Final

import pytest

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging, use_custom_pricing_for_model
from litellm.llms.base_llm.ocr.transformation import OCRPage, OCRResponse, OCRUsageInfo
from litellm.ocr.main import _prepare_ocr_request

OCR_MODEL: Final = "mistral/some-unmapped-ocr-model-for-testing"
DOCUMENT: Final = {"type": "document_url", "document_url": "https://example.com/doc.pdf"}


def _logging_obj() -> Logging:
    return Logging(
        model=OCR_MODEL,
        messages=[],
        stream=False,
        call_type="ocr",
        start_time=None,
        litellm_call_id="test-ocr-request-pricing",
        function_id="1234",
    )


def _prepare(kwargs: dict[str, object]) -> Logging:
    logging_obj: Final = _logging_obj()
    _prepare_ocr_request(
        model=OCR_MODEL,
        document=dict(DOCUMENT),
        api_key="test-key",
        api_base=None,
        timeout=None,
        custom_llm_provider=None,
        extra_headers=None,
        kwargs={"litellm_logging_obj": logging_obj, **kwargs},
    )
    return logging_obj


def test_prepare_ocr_request_forwards_custom_pricing_to_logging_params() -> None:
    logging_obj: Final = _prepare({"ocr_cost_per_page": 0.05, "ocr_cost_per_credit": 0.5})

    assert logging_obj.litellm_params["ocr_cost_per_page"] == 0.05
    assert logging_obj.litellm_params["ocr_cost_per_credit"] == 0.5
    assert use_custom_pricing_for_model(logging_obj.litellm_params) is True


def test_prepare_ocr_request_without_custom_pricing_leaves_logging_params_unpriced() -> None:
    logging_obj: Final = _prepare({})

    assert "ocr_cost_per_page" not in logging_obj.litellm_params
    assert use_custom_pricing_for_model(logging_obj.litellm_params) is False


def test_direct_ocr_call_bills_request_level_per_page_pricing() -> None:
    assert OCR_MODEL not in litellm.model_cost
    logging_obj: Final = _prepare({"ocr_cost_per_page": 0.05})
    response: Final = OCRResponse(
        pages=[OCRPage(index=index, markdown=f"page {index}") for index in range(3)],
        model=OCR_MODEL,
        usage_info=OCRUsageInfo(pages_processed=3),
    )

    assert logging_obj._response_cost_calculator(result=response) == pytest.approx(0.05 * 3)
