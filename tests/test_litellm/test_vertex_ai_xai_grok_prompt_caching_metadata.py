from pathlib import Path
from typing import Final

import pytest
from pydantic import TypeAdapter

import litellm
from litellm import get_model_info
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.utils import supports_prompt_caching

REPO_ROOT: Final = Path(__file__).parents[2]
MODEL: Final = "vertex_ai/xai/grok-4.6"
GROK_KEY_PREFIXES: Final = ("vertex_ai/xai/grok-", "azure_ai/grok-", "xai/grok-")
COST_MAP_ADAPTER: Final = TypeAdapter(dict[str, dict[str, object]])
MAIN_COST_MAP: Final = REPO_ROOT / "model_prices_and_context_window.json"
BACKUP_COST_MAP: Final = REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json"


def _cost_map(path: Path) -> dict[str, dict[str, object]]:
    return COST_MAP_ADAPTER.validate_json(path.read_bytes())


@pytest.mark.usefixtures("local_model_cost_map")
def test_grok_models_with_cache_read_price_advertise_prompt_caching() -> None:
    cached_grok_models = tuple(
        key
        for key, entry in litellm.model_cost.items()
        if key.startswith(GROK_KEY_PREFIXES) and entry.get("cache_read_input_token_cost")
    )
    assert cached_grok_models, "expected at least one grok model with a cache read price"

    missing_flag = tuple(
        key for key in cached_grok_models if get_model_info(model=key).get("supports_prompt_caching") is not True
    )
    assert missing_flag == (), (
        f"grok models with cache_read_input_token_cost fail get_model_info supports_prompt_caching: {missing_flag}"
    )


@pytest.mark.usefixtures("local_model_cost_map")
def test_vertex_ai_grok_4_6_supports_prompt_caching_via_get_model_info() -> None:
    routed_model, provider, _, _ = get_llm_provider(model=MODEL)
    assert (routed_model, provider) == ("xai/grok-4.6", "vertex_ai")

    routed_info = get_model_info(model=routed_model, custom_llm_provider=provider)
    assert routed_info["litellm_provider"] == "vertex_ai"
    assert routed_info.get("supports_prompt_caching") is True
    assert routed_info.get("cache_read_input_token_cost")

    catalog_info = get_model_info(model=MODEL)
    assert catalog_info["key"] == MODEL
    assert catalog_info.get("supports_prompt_caching") is True
    assert catalog_info.get("cache_read_input_token_cost")

    assert supports_prompt_caching(model=MODEL) is True


def test_vertex_ai_grok_entries_source_and_backup_match() -> None:
    main_map = _cost_map(MAIN_COST_MAP)
    backup_map = _cost_map(BACKUP_COST_MAP)

    vertex_grok_keys = tuple(key for key in main_map if key.startswith("vertex_ai/xai/grok-"))
    assert vertex_grok_keys, "expected at least one vertex_ai/xai/grok- entry"

    mismatched = tuple(key for key in vertex_grok_keys if backup_map.get(key) != main_map[key])
    assert mismatched == (), f"vertex grok entries differ between source and backup: {mismatched}"
