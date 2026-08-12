"""
Regression tests: OCR cost must honour deployment-specific custom pricing.

Before the fix, `ocr_cost()` resolved pricing exclusively through
`litellm.get_model_info(model=..., custom_llm_provider=...)`, i.e. a cost map
lookup keyed by model name. Custom pricing set on a deployment is registered
under the router's deployment id, and the shared "{provider}/{model}" key has
its pricing fields stripped, so the lookup could never see it. An OCR model
absent from the cost map therefore billed $0 no matter how it was priced in
config, even though `ocr_cost_per_page` / `ocr_cost_per_credit` are declared
fields of `CustomPricingLiteLLMParams`.
"""

import pytest

import litellm
from litellm.cost_calculator import completion_cost, ocr_cost
from litellm.llms.base_llm.ocr.transformation import OCRPage, OCRResponse, OCRUsageInfo

# A model deliberately absent from the cost map.
UNMAPPED_MODEL = "azure_ai/some-unmapped-ocr-model-for-testing"
CUSTOM_COST_PER_PAGE = 0.004
CUSTOM_COST_PER_CREDIT = 0.25


def _ocr_response(model: str, pages_processed: int = 1, credits: int | None = None) -> OCRResponse:
    # NOTE: model_construct() is used rather than OCRResponse(...) because the
    # OCRResponse field `object: str = "ocr"` shadows the builtin `object` used
    # in the `tables` / `keyValuePairs` annotations above it, so pydantic tries
    # to resolve "ocr" as a forward-referenced type and schema building fails.
    # That is an unrelated defect; validation is not what these tests exercise.
    usage_info = OCRUsageInfo(pages_processed=pages_processed)
    if credits is not None:
        usage_info.credits = credits
    return OCRResponse.model_construct(
        pages=[OCRPage(index=i, markdown=f"page {i}") for i in range(pages_processed)],
        model=model,
        usage_info=usage_info,
    )


def test_unmapped_ocr_model_has_no_map_pricing() -> None:
    """Guard the premise: the model really is absent from the cost map."""
    assert UNMAPPED_MODEL not in litellm.model_cost


@pytest.mark.parametrize("pages_processed", [1, 3, 10])
def test_ocr_cost_uses_custom_per_page_pricing(pages_processed: int) -> None:
    cost, _ = ocr_cost(
        model=UNMAPPED_MODEL,
        custom_llm_provider="azure_ai",
        response=_ocr_response(UNMAPPED_MODEL, pages_processed=pages_processed),
        model_info={"ocr_cost_per_page": CUSTOM_COST_PER_PAGE},
    )
    assert cost == pytest.approx(CUSTOM_COST_PER_PAGE * pages_processed)


def test_ocr_cost_uses_custom_per_credit_pricing() -> None:
    cost, _ = ocr_cost(
        model=UNMAPPED_MODEL,
        custom_llm_provider="azure_ai",
        response=_ocr_response(UNMAPPED_MODEL, pages_processed=2, credits=4),
        model_info={"ocr_cost_per_credit": CUSTOM_COST_PER_CREDIT},
    )
    assert cost == pytest.approx(CUSTOM_COST_PER_CREDIT * 4)


def test_unmapped_ocr_model_without_custom_pricing_still_bills_zero() -> None:
    """Unchanged behaviour when nothing is configured — no map entry, no override."""
    cost, _ = ocr_cost(
        model=UNMAPPED_MODEL,
        custom_llm_provider="azure_ai",
        response=_ocr_response(UNMAPPED_MODEL, pages_processed=5),
    )
    assert cost == 0.0


def test_custom_pricing_does_not_override_a_mapped_model_when_absent() -> None:
    """model_info without OCR pricing must fall through to the cost map."""
    mapped_model = "mistral/mistral-ocr-4-0"
    cost, _ = ocr_cost(
        model=mapped_model,
        custom_llm_provider="mistral",
        response=_ocr_response(mapped_model, pages_processed=2),
        model_info={"id": "some-deployment-id"},
    )
    assert cost == pytest.approx(0.004 * 2)


def test_ocr_custom_pricing_end_to_end_through_completion_cost() -> None:
    """The whole path: litellm_params.metadata.model_info -> ocr_cost."""
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging

    logging_obj = LiteLLMLogging(
        model=UNMAPPED_MODEL,
        messages=[],
        stream=False,
        call_type="ocr",
        start_time=None,
        litellm_call_id="test-ocr-custom-pricing",
        function_id="1234",
    )
    logging_obj.litellm_params = {"metadata": {"model_info": {"ocr_cost_per_page": CUSTOM_COST_PER_PAGE}}}

    cost = completion_cost(
        completion_response=_ocr_response(UNMAPPED_MODEL, pages_processed=3),
        model=UNMAPPED_MODEL,
        custom_llm_provider="azure_ai",
        call_type="ocr",
        custom_pricing=True,
        litellm_logging_obj=logging_obj,
    )
    assert cost == pytest.approx(CUSTOM_COST_PER_PAGE * 3)
