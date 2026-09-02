import json
from pathlib import Path

import pytest

from litellm.cost_calculator import cost_per_token
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

REPO_ROOT = Path(__file__).parents[2]
MAIN_PATH = REPO_ROOT / "model_prices_and_context_window.json"
BACKUP_PATH = REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json"

MODEL_LIMITS = (
    ("azure_ai/FW-DeepSeek-V4-Flash-0731", "chat", 1048576, 131072, "2027-08-15"),
    ("azure_ai/FW-GLM-5.3", "chat", 1048576, 131072, "2027-09-01"),
    ("azure_ai/MAI-Thinking-1", "chat", 256000, 64000, "2026-11-04"),
    ("azure_ai/Codestral-2501", "chat", 256000, 4096, None),
    ("azure_ai/Kimi-K3", "chat", 1048576, 131072, "2026-11-26"),
    ("azure_ai/grok-4.6", "chat", 200000, 128000, "2027-08-24"),
    ("azure_ai/MAI-Image-2.5-Pro", "image_generation", 32000, None, "2026-10-31"),
    ("azure_ai/mistral-ocr-4-0", "ocr", 128000, 4096, "2027-10-01"),
)

CHAT_PRICING = (
    ("azure_ai/FW-DeepSeek-V4-Flash-0731", 2.2e-07, 6.6e-07, 7e-09),
    ("azure_ai/FW-GLM-5.3", 1.4e-06, 4.4e-06, 2.6e-07),
    ("azure_ai/MAI-Thinking-1", 2e-06, 8e-06, 2e-07),
    ("azure_ai/Codestral-2501", 3e-07, 9e-07, None),
    ("azure_ai/Kimi-K3", 3e-06, 1.5e-05, 3e-07),
    ("azure_ai/grok-4.6", 2e-06, 6e-06, 5e-07),
)

REASONING_MODELS = (
    "azure_ai/FW-DeepSeek-V4-Flash-0731",
    "azure_ai/FW-GLM-5.3",
    "azure_ai/MAI-Thinking-1",
    "azure_ai/Kimi-K3",
    "azure_ai/grok-4.6",
)


def _load(path: Path) -> dict[str, dict[str, object]]:
    with open(path, encoding="utf-8") as model_cost_file:
        return json.load(model_cost_file)


@pytest.mark.parametrize("model, mode, max_input, max_output, deprecation", MODEL_LIMITS)
def test_azure_ai_catalog_model_limits(
    model: str,
    mode: str,
    max_input: int,
    max_output: int | None,
    deprecation: str | None,
) -> None:
    info = _load(MAIN_PATH).get(model)
    assert info is not None, f"{model} missing from model_prices_and_context_window.json"

    assert info["litellm_provider"] == "azure_ai"
    assert info["mode"] == mode
    assert info["max_input_tokens"] == max_input
    assert info.get("max_output_tokens") == max_output
    assert info.get("max_tokens") == max_output
    assert info.get("deprecation_date") == deprecation


@pytest.mark.parametrize("model, input_cost, output_cost, cache_cost", CHAT_PRICING)
def test_azure_ai_catalog_chat_pricing(
    local_model_cost_map: None,
    model: str,
    input_cost: float,
    output_cost: float,
    cache_cost: float | None,
) -> None:
    info = _load(MAIN_PATH)[model]
    prompt_cost, completion_cost = cost_per_token(model=model, prompt_tokens=1000, completion_tokens=500)

    assert info["input_cost_per_token"] == input_cost
    assert info["output_cost_per_token"] == output_cost
    assert info.get("cache_read_input_token_cost") == cache_cost
    assert prompt_cost == pytest.approx(1000 * input_cost)
    assert completion_cost == pytest.approx(500 * output_cost)


@pytest.mark.parametrize("model", REASONING_MODELS)
def test_azure_ai_catalog_reasoning_capabilities(model: str) -> None:
    info = _load(MAIN_PATH)[model]

    assert info["supports_reasoning"] is True
    assert info["supports_function_calling"] is True
    assert info["supports_tool_choice"] is True


@pytest.mark.parametrize(
    "model",
    ("azure_ai/FW-DeepSeek-V4-Flash-0731", "azure_ai/FW-GLM-5.3", "azure_ai/Kimi-K3"),
)
def test_azure_ai_catalog_reasoning_effort_levels(model: str) -> None:
    assert _load(MAIN_PATH)[model]["reasoning_effort_levels"] == ["low", "high", "max"]


def test_mai_image_2_5_pro_pricing_and_modalities() -> None:
    info = _load(MAIN_PATH)["azure_ai/MAI-Image-2.5-Pro"]

    assert info["input_cost_per_token"] == 5e-06
    assert info["input_cost_per_image_token"] == 8e-06
    assert info["output_cost_per_image_token"] == 0.000106
    assert info["supported_modalities"] == ["text", "image"]
    assert info["supported_output_modalities"] == ["image"]
    assert info["supported_endpoints"] == ["/v1/images/generations", "/v1/images/edits"]


def test_mistral_ocr_4_pricing_and_modalities() -> None:
    info = _load(MAIN_PATH)["azure_ai/mistral-ocr-4-0"]

    assert info["ocr_cost_per_page"] == 0.004
    assert info["annotation_cost_per_page"] == 0.005
    assert info["supported_modalities"] == ["image"]
    assert info["supported_output_modalities"] == ["text"]
    assert info["supports_pdf_input"] is True
    assert info["supported_endpoints"] == ["/v1/ocr"]


@pytest.mark.parametrize("model, _mode, _max_input, _max_output, _deprecation", MODEL_LIMITS)
def test_azure_ai_catalog_models_route_to_azure_ai(
    model: str,
    _mode: str,
    _max_input: int,
    _max_output: int | None,
    _deprecation: str | None,
) -> None:
    routed_model, provider, _, _ = get_llm_provider(model=model)

    assert routed_model == model.split("/", 1)[1]
    assert provider == "azure_ai"


@pytest.mark.parametrize("model, _mode, _max_input, _max_output, _deprecation", MODEL_LIMITS)
def test_azure_ai_catalog_backup_matches_main(
    model: str,
    _mode: str,
    _max_input: int,
    _max_output: int | None,
    _deprecation: str | None,
) -> None:
    assert _load(BACKUP_PATH).get(model) == _load(MAIN_PATH).get(model), (
        f"{model} differs between main and backup model cost maps"
    )
