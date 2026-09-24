import json
from pathlib import Path
from typing import Final


REPO_ROOT: Final = Path(__file__).parents[2]
MAIN_PATH: Final = REPO_ROOT / "model_prices_and_context_window.json"
BACKUP_PATH: Final = (
    REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json"
)

MODEL: Final = "bing_grounding/search"
BING_GROUNDING_COST_PER_QUERY: Final = 0.014


def _load(path: Path) -> dict[str, dict[str, object]]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def test_bing_grounding_pricing():
    model_info = _load(MAIN_PATH)[MODEL]

    assert model_info["input_cost_per_query"] == BING_GROUNDING_COST_PER_QUERY
    assert model_info["litellm_provider"] == "bing_grounding"
    assert model_info["mode"] == "search"


def test_bing_grounding_backup_matches_main():
    main_cost = _load(MAIN_PATH)
    backup_cost = _load(BACKUP_PATH)

    assert MODEL in main_cost
    assert MODEL in backup_cost
    assert backup_cost[MODEL] == main_cost[MODEL]