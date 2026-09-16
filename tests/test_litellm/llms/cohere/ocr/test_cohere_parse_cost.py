from pathlib import Path

import pytest

import litellm
from litellm.llms.base_llm.ocr.transformation import OCRPage, OCRResponse, OCRUsageInfo

REPO_ROOT = Path(__file__).parents[5]
COST_MAPS = [
    REPO_ROOT / "model_prices_and_context_window.json",
    REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json",
]
MODELS = [("cohere/parse-v5.0", "cohere"), ("azure_ai/Cohere-parse-v5", "azure_ai")]


def _ocr_response(model: str, pages_processed: int) -> OCRResponse:
    return OCRResponse(
        pages=[OCRPage(index=i, markdown=f"page {i}") for i in range(pages_processed)],
        model=model,
        usage_info=OCRUsageInfo(pages_processed=pages_processed),
    )


@pytest.mark.parametrize("model, provider", MODELS)
def test_model_info_resolves_ocr_mode_and_price(local_model_cost_map, model: str, provider: str) -> None:
    info = litellm.get_model_info(model=model, custom_llm_provider=provider)

    assert info["mode"] == "ocr"
