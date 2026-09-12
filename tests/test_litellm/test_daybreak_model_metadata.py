import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[2]
MAIN_PATH = REPO_ROOT / "model_prices_and_context_window.json"
BACKUP_PATH = REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json"

DAYBREAK_MODELS = (
    "gpt-5.6-cyber",
    "daybreak-red-latest",
    "daybreak-blue-latest",
)
BLUE_ALIAS = "daybreak-blue-latest"
BLUE_SNAPSHOT = "gpt-5.6-sol"
OFFICIAL_ALIAS_SNAPSHOTS = (
    ("gpt-daybreak-blue-latest", "gpt-5.6-sol"),
    ("gpt-daybreak-red-latest", "gpt-5.6-cyber"),
)
PRICE_FIELDS = (
    "input_cost_per_token",
    "output_cost_per_token",
    "cache_read_input_token_cost",
    "input_cost_per_token_above_272k_tokens",
    "output_cost_per_token_above_272k_tokens",
)


def _load(path):
    with open(path) as f:
        return json.load(f)


def test_blue_alias_matches_its_snapshot_computer_use():
    cost_map = _load(MAIN_PATH)

    assert cost_map[BLUE_ALIAS]["supports_computer_use"] is True
    assert cost_map[BLUE_SNAPSHOT]["supports_computer_use"] is True


@pytest.mark.parametrize(("alias", "snapshot"), OFFICIAL_ALIAS_SNAPSHOTS)
def test_official_alias_tracks_snapshot(alias, snapshot):
    cost_map = _load(MAIN_PATH)
    alias_info = cost_map[alias]
    snapshot_info = cost_map[snapshot]

    assert alias_info["supported_endpoints"] == ["/v1/responses"]
    assert alias_info["mode"] == "responses"
    assert alias_info["source"] == f"https://developers.openai.com/api/docs/models/{alias}"
    assert {field: alias_info.get(field) for field in PRICE_FIELDS} == {
        field: snapshot_info.get(field) for field in PRICE_FIELDS
    }
    assert alias_info["max_output_tokens"] == snapshot_info["max_output_tokens"]


@pytest.mark.parametrize("model", (*DAYBREAK_MODELS, BLUE_SNAPSHOT, *(alias for alias, _ in OFFICIAL_ALIAS_SNAPSHOTS)))
def test_backup_matches_main(model):
    main_cost = _load(MAIN_PATH)
    backup_cost = _load(BACKUP_PATH)

    assert backup_cost.get(model) == main_cost.get(model), f"{model} differs between main and backup model cost maps"
