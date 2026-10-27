from pathlib import Path
from typing import Final

import pytest
from pydantic import TypeAdapter

from litellm import cost_per_token, get_model_info
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

REPO_ROOT: Final = Path(__file__).parents[2]
COST_MAP_ADAPTER: Final = TypeAdapter(dict[str, dict[str, object]])

GLOBAL_TIER: Final = "azure/gpt-6.1-terra"
DATA_ZONE_TIERS: Final = {
    "azure/us/gpt-6.1-terra": 1.1,
    "azure/eu/gpt-6.1-terra": 1.2,
    "azure/apac/gpt-6.1-terra": 1.2,
}
AZURE_TIERS: Final = (GLOBAL_TIER, *DATA_ZONE_TIERS)

PER_TOKEN_FIELDS: Final = (
    "input_cost_per_token",
    "output_cost_per_token",
    "cache_read_input_token_cost",
    "cache_creation_input_token_cost",
    "input_cost_per_token_above_272k_tokens",
    "output_cost_per_token_above_272k_tokens",
)


def _cost_map(path: Path) -> dict[str, dict[str, object]]:
    return COST_MAP_ADAPTER.validate_json(path.read_bytes())


@pytest.mark.parametrize("model", AZURE_TIERS)
def test_azure_gpt_6_1_terra_is_priced_and_routed(model: str) -> None:
    _, provider, _, _ = get_llm_provider(model=model)
    assert provider == "azure"

    info = get_model_info(model=model, custom_llm_provider=provider)
    assert info["litellm_provider"] == "azure"
    assert info["mode"] == "chat"
    assert info["supports_function_calling"] is True
    assert info["supports_prompt_caching"] is True
    assert info["supports_reasoning"] is True
    assert info["supports_response_schema"] is True
    assert info["supports_tool_choice"] is True
    assert info["supports_vision"] is True
    assert info["supports_web_search"] is True

    prompt_cost, completion_cost = cost_per_token(model=model, prompt_tokens=1_000_000, completion_tokens=1_000_000)
    assert prompt_cost > 0
    assert completion_cost > 0


def test_azure_gpt_6_1_terra_bills_the_published_tier_differentials() -> None:
    cost_map = _cost_map(REPO_ROOT / "model_prices_and_context_window.json")
    global_entry = cost_map[GLOBAL_TIER]

    for model, multiplier in DATA_ZONE_TIERS.items():
        entry = cost_map[model]
        for field in PER_TOKEN_FIELDS:
            assert entry[field] == pytest.approx(global_entry[field] * multiplier), f"{model}.{field}"


@pytest.mark.parametrize("model", AZURE_TIERS)
def test_azure_gpt_6_1_terra_long_context_tier_is_charged_on_top(model: str) -> None:
    info = get_model_info(model=model, custom_llm_provider="azure")
    assert info["input_cost_per_token_above_272k_tokens"] == pytest.approx(info["input_cost_per_token"] * 2)
    assert info["output_cost_per_token_above_272k_tokens"] == pytest.approx(info["output_cost_per_token"] * 1.5)


def test_azure_gpt_6_1_terra_entry_source_and_backup_match() -> None:
    main_map = _cost_map(REPO_ROOT / "model_prices_and_context_window.json")
    backup_map = _cost_map(REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json")

    for model in AZURE_TIERS:
        assert backup_map[model] == main_map[model], f"{model} differs between main and backup cost maps"