"""
Test Azure AI Mistral Foundry model metadata and cost for the models added
alongside Mistral Document AI (with OCR 4) and Mistral Medium 3.5 in Microsoft
Foundry: azure_ai/mistral-ocr-4-0 and azure_ai/mistral-medium-3-5.
"""

import json
from importlib.resources import files

import pytest


@pytest.fixture(scope="module")
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


def test_azure_ai_mistral_medium_3_5_model_info(use_local_model_cost_map):
    model_info = use_local_model_cost_map.get_model_info(model="azure_ai/mistral-medium-3-5")

    assert model_info["litellm_provider"] == "azure_ai"
    assert model_info["mode"] == "chat"
    assert model_info["max_input_tokens"] == 262144
    assert model_info["max_output_tokens"] == 262144
    assert model_info["max_tokens"] == 262144
    assert model_info["input_cost_per_token"] == pytest.approx(1.5e-06)
    assert model_info["output_cost_per_token"] == pytest.approx(7.5e-06)
    assert model_info["supports_function_calling"] is True
    assert model_info["supports_reasoning"] is True
    assert model_info["supports_response_schema"] is True
    assert model_info["supports_tool_choice"] is True
    assert model_info["supports_vision"] is True
    assert model_info["supports_assistant_prefill"] is True


def test_azure_ai_mistral_medium_3_5_cost_per_token(use_local_model_cost_map):
    from litellm.llms.azure_ai.cost_calculator import cost_per_token
    from litellm.types.utils import Usage

    usage = Usage(
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        total_tokens=2_000_000,
    )

    prompt_cost, completion_cost = cost_per_token(model="mistral-medium-3-5", usage=usage)

    assert prompt_cost == pytest.approx(1.5)
    assert completion_cost == pytest.approx(7.5)


def test_azure_ai_mistral_ocr_4_model_info(use_local_model_cost_map):
    model_info = use_local_model_cost_map.get_model_info(model="azure_ai/mistral-ocr-4-0")

    assert model_info["litellm_provider"] == "azure_ai"
    assert model_info["mode"] == "ocr"
    assert model_info["ocr_cost_per_page"] == pytest.approx(0.004)
    assert model_info["annotation_cost_per_page"] == pytest.approx(0.005)

    raw_entry = use_local_model_cost_map.model_cost["azure_ai/mistral-ocr-4-0"]
    assert raw_entry["supported_endpoints"] == ["/v1/ocr"]


@pytest.mark.parametrize("pages_processed", [1, 3, 10])
def test_azure_ai_mistral_ocr_4_cost_scales_with_pages(use_local_model_cost_map, pages_processed):
    from litellm.cost_calculator import completion_cost
    from litellm.llms.base_llm.ocr.transformation import OCRPage, OCRResponse, OCRUsageInfo

    response = OCRResponse(
        pages=[OCRPage(index=i, markdown=f"page {i}") for i in range(pages_processed)],
        model="mistral-ocr-4-0",
        usage_info=OCRUsageInfo(pages_processed=pages_processed),
    )

    cost = completion_cost(
        completion_response=response,
        model="azure_ai/mistral-ocr-4-0",
        custom_llm_provider="azure_ai",
        call_type="ocr",
    )

    assert cost == pytest.approx(0.004 * pages_processed)
