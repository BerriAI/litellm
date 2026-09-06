import json
from pathlib import Path

import pytest

import litellm
from litellm.cost_calculator import completion_cost
from litellm.llms.base_llm.ocr.transformation import OCRPage, OCRResponse, OCRUsageInfo

COST_PER_PAGE = 0.0015
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


@pytest.mark.parametrize("cost_map_path", COST_MAPS, ids=lambda path: path.name)
@pytest.mark.parametrize("model, provider", MODELS)
def test_pricing_entry(cost_map_path: Path, model: str, provider: str) -> None:
    with open(cost_map_path) as f:
        info = json.load(f).get(model)

    assert info is not None, f"{model} missing from {cost_map_path.name}"
    assert info["litellm_provider"] == provider
    assert info["mode"] == "ocr"
    assert info["supported_endpoints"] == ["/v1/ocr"]
    assert info["ocr_cost_per_page"] == COST_PER_PAGE


@pytest.mark.parametrize("model, provider", MODELS)
def test_model_info_resolves_ocr_mode_and_price(local_model_cost_map, model: str, provider: str) -> None:
    info = litellm.get_model_info(model=model, custom_llm_provider=provider)

    assert info["mode"] == "ocr"
    assert info["ocr_cost_per_page"] == COST_PER_PAGE


@pytest.mark.parametrize("model, provider", MODELS)
@pytest.mark.parametrize("pages_processed", [1, 3])
def test_cost_scales_with_billed_pages(local_model_cost_map, model: str, provider: str, pages_processed: int) -> None:
    cost = completion_cost(
        completion_response=_ocr_response(model.split("/", 1)[1], pages_processed),
        model=model,
        custom_llm_provider=provider,
        call_type="ocr",
    )

    assert cost == pytest.approx(COST_PER_PAGE * pages_processed)
