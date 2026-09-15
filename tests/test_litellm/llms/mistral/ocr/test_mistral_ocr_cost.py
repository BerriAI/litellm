"""
Cost tests for Mistral OCR models against the real litellm cost map
(no monkeypatching of get_model_info). These regress the pricing entries
for mistral-ocr-4-0 and mistral-ocr-latest, which now both resolve to
OCR 4 at $4 / 1000 pages.
"""

from pathlib import Path

from litellm.llms.base_llm.ocr.transformation import OCRPage, OCRResponse, OCRUsageInfo

OCR4_COST_PER_PAGE = 0.004
OCR4_ANNOTATION_COST_PER_PAGE = 0.005

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
