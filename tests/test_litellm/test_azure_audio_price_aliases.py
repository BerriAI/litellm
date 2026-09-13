"""Undated azure aliases for the audio models must exist and match their dated
variants. Azure deployments are commonly created under an admin-chosen name, so
the served model name means nothing to the cost lookup and `base_model:
azure/gpt-audio-mini` is what prices the call. That key resolved to nothing, the
lookup raised "This model isn't mapped yet", and the proxy logged the request at
$0. Issue #33170."""

import json
from pathlib import Path

import pytest

import litellm

pytestmark = pytest.mark.usefixtures("local_model_cost_map")


COST_FIELDS = (
    "input_cost_per_token",
    "output_cost_per_token",
    "input_cost_per_audio_token",
    "output_cost_per_audio_token",
)

ALIAS_PAIRS = (
    ("azure/gpt-audio-mini", "azure/gpt-audio-mini-2025-10-06"),
    ("azure/gpt-realtime-mini", "azure/gpt-realtime-mini-2025-10-06"),
)


def _load_root_cost_map() -> dict:
    root_map_path = Path(__file__).parents[2] / "model_prices_and_context_window.json"
    with open(root_map_path) as f:
        return json.load(f)


@pytest.mark.parametrize("undated, dated", ALIAS_PAIRS)
def test_undated_azure_audio_alias_matches_dated_entry(undated, dated):
    undated_info = litellm.get_model_info(undated)
    dated_info = litellm.get_model_info(dated)

    for field in COST_FIELDS:
        assert undated_info.get(field) == dated_info.get(field), field
        assert (undated_info.get(field) or 0) > 0, f"{undated}.{field} must be non-zero"

    assert undated_info.get("litellm_provider") == "azure"
    assert undated_info.get("mode") == dated_info.get("mode")


@pytest.mark.parametrize("undated, dated", ALIAS_PAIRS)
def test_undated_azure_audio_alias_is_exact_mirror(undated, dated):
    """The undated alias must be a byte-for-byte mirror of its dated entry, covering
    every field (incl. realtime-specific cache/audio cost keys) so any future drift
    between the pair is caught, not just the core COST_FIELDS."""
    model_map = litellm.model_cost
    assert undated in model_map, f"{undated} missing from model cost map"
    assert model_map[undated] == model_map[dated], (
        f"{undated} must exactly mirror {dated}; "
        f"diff keys: {[k for k in set(model_map[undated]) | set(model_map[dated]) if model_map[undated].get(k) != model_map[dated].get(k)]}"
    )


@pytest.mark.parametrize("undated, dated", ALIAS_PAIRS)
def test_undated_azure_audio_alias_is_in_the_root_cost_map(undated, dated):
    """`local_model_cost_map` pins `litellm.model_cost` to the packaged backup, but a
    proxy left on its defaults fetches the root map instead, and that is the copy
    that ships to the CDN. An alias added to only one of the two files still bills
    $0 for every proxy reading the other, which is the very bug this file guards, so
    assert the root map directly and assert the two files agree."""
    root_map = _load_root_cost_map()
    assert undated in root_map, f"{undated} missing from the root cost map"
    assert root_map[undated] == root_map[dated], f"{undated} must exactly mirror {dated} in the root cost map"
    assert root_map[undated] == litellm.model_cost[undated], (
        f"{undated} differs between the root cost map and the packaged backup"
    )
