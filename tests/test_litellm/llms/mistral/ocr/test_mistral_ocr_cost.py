"""
Cost tests for Mistral OCR models against the real litellm cost map
(no monkeypatching of get_model_info). These regress the pricing entries
for mistral-ocr-4-0 and mistral-ocr-latest, which now both resolve to
OCR 4 at $4 / 1000 pages.
"""

import json
from pathlib import Path

import pytest

import litellm
from litellm.cost_calculator import completion_cost
from litellm.llms.base_llm.ocr.transformation import OCRPage, OCRResponse, OCRUsageInfo

OCR4_COST_PER_PAGE = 0.004

REPO_ROOT = Path(__file__).parents[5]
MAIN_COST_MAP = REPO_ROOT / "model_prices_and_context_window.json"
BACKUP_COST_MAP = REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json"

OCR3_MODEL = "mistral/mistral-ocr-2512"
OCR3_COST_PER_PAGE = 0.002
OCR3_ANNOTATION_COST_PER_PAGE = 0.003

AZURE_DOC_AI_MODEL = "azure_ai/mistral-document-ai-2512"
AZURE_DOC_AI_COST_PER_PAGE = 0.003


def _ocr_response(model: str, pages_processed: int) -> OCRResponse:
    return OCRResponse(
        pages=[OCRPage(index=i, markdown=f"page {i}") for i in range(pages_processed)],
        model=model,
        usage_info=OCRUsageInfo(pages_processed=pages_processed),
    )


def _annotated_ocr_response(model: str, pages_processed: int | None, annotation_pages: int) -> OCRResponse:
    return OCRResponse(
        pages=[],
        model=model,
        usage_info=OCRUsageInfo(pages_processed=pages_processed, pages_processed_annotation=annotation_pages),
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



@pytest.mark.parametrize("cost_map_path", [MAIN_COST_MAP, BACKUP_COST_MAP])
def test_ocr3_pricing_entry(cost_map_path: Path) -> None:
    with open(cost_map_path) as f:
        info = json.load(f).get(OCR3_MODEL)

    assert info is not None, f"{OCR3_MODEL} missing from {cost_map_path.name}"
    assert info["litellm_provider"] == "mistral"
    assert info["mode"] == "ocr"
    assert info["supported_endpoints"] == ["/v1/ocr"]
    assert info["ocr_cost_per_page"] == OCR3_COST_PER_PAGE
    assert info["annotation_cost_per_page"] == OCR3_ANNOTATION_COST_PER_PAGE


def test_ocr3_model_info_price(local_model_cost_map) -> None:
    info = litellm.get_model_info(model=OCR3_MODEL, custom_llm_provider="mistral")
    assert info["ocr_cost_per_page"] == OCR3_COST_PER_PAGE


@pytest.mark.parametrize("pages_processed", [1, 3, 10])
def test_ocr3_cost_scales_with_pages(local_model_cost_map, pages_processed: int) -> None:
    cost = completion_cost(
        completion_response=_ocr_response("mistral-ocr-2512", pages_processed),
        model=OCR3_MODEL,
        custom_llm_provider="mistral",
        call_type="ocr",
    )
    assert cost == pytest.approx(OCR3_COST_PER_PAGE * pages_processed)


def test_ocr3_bills_ocr_and_annotation_pages_at_their_own_rates(local_model_cost_map) -> None:
    cost = completion_cost(
        completion_response=_annotated_ocr_response("mistral-ocr-2512", 2, 3),
        model=OCR3_MODEL,
        custom_llm_provider="mistral",
        call_type="ocr",
    )
    assert cost == pytest.approx(2 * OCR3_COST_PER_PAGE + 3 * OCR3_ANNOTATION_COST_PER_PAGE)


def test_ocr3_bills_annotation_only_response(local_model_cost_map) -> None:
    cost = completion_cost(
        completion_response=_annotated_ocr_response("mistral-ocr-2512", 0, 3),
        model=OCR3_MODEL,
        custom_llm_provider="mistral",
        call_type="ocr",
    )
    assert cost == pytest.approx(3 * OCR3_ANNOTATION_COST_PER_PAGE)


def test_ocr3_bills_annotation_pages_when_pages_processed_missing(local_model_cost_map) -> None:
    cost = completion_cost(
        completion_response=_annotated_ocr_response("mistral-ocr-2512", None, 4),
        model=OCR3_MODEL,
        custom_llm_provider="mistral",
        call_type="ocr",
    )
    assert cost == pytest.approx(4 * OCR3_ANNOTATION_COST_PER_PAGE)


def test_azure_doc_ai_annotation_pages_fall_back_to_ocr_rate(local_model_cost_map) -> None:
    info = litellm.get_model_info(model=AZURE_DOC_AI_MODEL, custom_llm_provider="azure_ai")
    assert info.get("annotation_cost_per_page") is None
    assert info["ocr_cost_per_page"] == AZURE_DOC_AI_COST_PER_PAGE
    cost = completion_cost(
        completion_response=_annotated_ocr_response("mistral-document-ai-2512", 0, 1),
        model=AZURE_DOC_AI_MODEL,
        custom_llm_provider="azure_ai",
        call_type="ocr",
    )
    assert cost == pytest.approx(AZURE_DOC_AI_COST_PER_PAGE)
