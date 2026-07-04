"""
Metadata / pricing tests for the Azure AI Foundry Mistral OCR model
(azure_ai/mistral-ocr-2503).

Regresses that the entry exists in both the main and backup cost maps with
the expected OCR pricing, and that get_model_info resolves it against the
local cost map.
"""

import json
from importlib.resources import files
from pathlib import Path

import pytest

MODEL = "azure_ai/mistral-ocr-2503"
OCR_COST_PER_PAGE = 0.004

REPO_ROOT = Path(__file__).parents[4]
MAIN_COST_MAP = REPO_ROOT / "model_prices_and_context_window.json"
BACKUP_COST_MAP = REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json"


@pytest.mark.parametrize("cost_map_path", [MAIN_COST_MAP, BACKUP_COST_MAP])
def test_azure_ai_mistral_ocr_pricing_entry(cost_map_path: Path) -> None:
    with open(cost_map_path) as f:
        info = json.load(f).get(MODEL)

    assert info is not None, f"{MODEL} missing from {cost_map_path.name}"
    assert info["litellm_provider"] == "azure_ai"
    assert info["mode"] == "ocr"
    assert info["supported_endpoints"] == ["/v1/ocr"]
    assert info["ocr_cost_per_page"] == OCR_COST_PER_PAGE


@pytest.fixture
def use_local_model_cost_map():
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")

    import litellm
    from litellm.utils import _invalidate_model_cost_lowercase_map

    original_model_cost = litellm.model_cost
    litellm.model_cost = json.loads(
        files("litellm")
        .joinpath("model_prices_and_context_window_backup.json")
        .read_text(encoding="utf-8")
    )
    litellm.get_model_info.cache_clear()
    _invalidate_model_cost_lowercase_map()
    try:
        yield litellm
    finally:
        litellm.model_cost = original_model_cost
        litellm.get_model_info.cache_clear()
        _invalidate_model_cost_lowercase_map()
        monkeypatch.undo()


def test_azure_ai_mistral_ocr_model_info(use_local_model_cost_map) -> None:
    info = use_local_model_cost_map.get_model_info(model=MODEL)

    assert info["litellm_provider"] == "azure_ai"
    assert info["mode"] == "ocr"
    assert info["ocr_cost_per_page"] == OCR_COST_PER_PAGE
