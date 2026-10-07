from pathlib import Path
from typing import Final

import pytest
from pydantic import TypeAdapter

from litellm import get_model_info
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

REPO_ROOT: Final = Path(__file__).parents[2]
COST_MAP_ADAPTER: Final = TypeAdapter(dict[str, dict[str, object]])
MODELS: Final = (
    "openrouter/openai/gpt-image-2",
    "openrouter/openai/gpt-image-2.5-flare",
    "openrouter/openai/gpt-image-2.5-sunburst",
)
PRICE_FIELDS: Final = ("input_cost_per_token", "input_cost_per_image_token", "output_cost_per_image_token")


def _cost_map(path: Path) -> dict[str, dict[str, object]]:
    return COST_MAP_ADAPTER.validate_json(path.read_bytes())


MAIN_COST_MAP: Final = _cost_map(REPO_ROOT / "model_prices_and_context_window.json")
BACKUP_COST_MAP: Final = _cost_map(REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json")


@pytest.mark.parametrize("model", MODELS)
def test_openrouter_gpt_image_row_is_identical_in_main_and_backup(model: str) -> None:
    assert model in MAIN_COST_MAP, f"{model} is missing from model_prices_and_context_window.json"
    assert BACKUP_COST_MAP.get(model) == MAIN_COST_MAP[model]


@pytest.mark.usefixtures("local_model_cost_map")
@pytest.mark.parametrize("model", MODELS)
def test_openrouter_gpt_image_model_info_comes_from_the_openrouter_row(model: str) -> None:
    routed_model, provider, _, _ = get_llm_provider(model=model)
    assert (routed_model, provider) == (model.removeprefix("openrouter/"), "openrouter")

    info = get_model_info(model=routed_model, custom_llm_provider=provider)
    row = MAIN_COST_MAP[model]
    assert (info["key"], info["litellm_provider"], info["mode"]) == (model, "openrouter", "image_generation")
    assert {field: info[field] for field in PRICE_FIELDS} == {field: row[field] for field in PRICE_FIELDS}
