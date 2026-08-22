import json
import os
from types import MappingProxyType
from typing import Final

CACHE_WRITE_AND_READ_RATIOS_ON_INPUT_RATE: Final = MappingProxyType(
    {"claude": (1.25, 0.1), "gemini": (1.0, 0.1), "gpt-5": (1.0, 0.1)}
)


def _load_cost_map() -> dict:
    json_path: Final = os.path.join(os.path.dirname(__file__), "../../../../model_prices_and_context_window.json")
    assert os.path.exists(json_path), f"Could not find model_prices_and_context_window.json at {json_path}"
    with open(json_path, "r") as f:
        return json.load(f)


def _cache_vendor(model: str) -> str | None:
    name: Final = model.removeprefix("databricks/databricks-")
    if name.startswith("claude"):
        return "claude"
    if name.startswith("gemini"):
        return "gemini"
    if name.startswith("gpt-5") and "oss" not in name:
        return "gpt-5"
    return None


def test_databricks_pricing_integrity():
    """
    Verifies that for all Databricks models in model_prices_and_context_window.json:
    USD Price == DBU Price * 0.07
    """
    json_path = os.path.join(os.path.dirname(__file__), "../../../../model_prices_and_context_window.json")

    # Verify file exists
    assert os.path.exists(json_path), f"Could not find model_prices_and_context_window.json at {json_path}"

    with open(json_path, "r") as f:
        data = json.load(f)

    conversion_rate = 0.07  # 1 DBU = 0.07 USD
    errors = []

    for model, info in data.items():
        if info.get("litellm_provider") == "databricks":
            # Check Input Cost
            input_usd = info.get("input_cost_per_token")
            input_dbu = info.get("input_dbu_cost_per_token")

            if input_usd is not None and input_dbu is not None:
                expected = input_dbu * conversion_rate
                # Allow small floating point difference
                if abs(input_usd - expected) > 1e-9:
                    errors.append(f"{model} input mismatch: USD={input_usd}, DBU={input_dbu}, Expected={expected}")

            # Check Output Cost
            output_usd = info.get("output_cost_per_token")
            output_dbu = info.get("output_dbu_cost_per_token")

            if output_usd is not None and output_dbu is not None:
                expected = output_dbu * conversion_rate
                if abs(output_usd - expected) > 1e-9:
                    errors.append(f"{model} output mismatch: USD={output_usd}, DBU={output_dbu}, Expected={expected}")

    assert not errors, "\n" + "\n".join(errors)


def test_databricks_proprietary_models_have_cache_pricing():
    """
    Every Databricks-hosted Claude / Gemini / GPT-5 chat model must carry cached token pricing,
    derived from its input rate with the vendor's cache write and cache read multipliers
    """
    data: Final = _load_cost_map()
    errors: Final[list[str]] = []  # mutable-ok: pytest failure message accumulator

    priced: Final[list[str]] = []  # mutable-ok: pytest failure message accumulator
    for model, info in data.items():
        vendor: Final = _cache_vendor(model) if model.startswith("databricks/") else None
        if vendor is None or info.get("mode") != "chat":
            continue
        priced.append(model)
        write_ratio, read_ratio = CACHE_WRITE_AND_READ_RATIOS_ON_INPUT_RATE[vendor]
        input_usd = info["input_cost_per_token"]
        for field, ratio in (
            ("cache_creation_input_token_cost", write_ratio),
            ("cache_read_input_token_cost", read_ratio),
        ):
            actual = info.get(field)
            expected = input_usd * ratio
            if actual is None:
                errors.append(f"{model} missing {field}")
            elif abs(actual - expected) > 1e-12:
                errors.append(f"{model} {field} mismatch: got {actual}, expected {expected}")
        if not info.get("supports_prompt_caching"):
            errors.append(f"{model} missing supports_prompt_caching")

    assert priced, "no Databricks proprietary chat models found"
    assert not errors, "\n" + "\n".join(errors)
