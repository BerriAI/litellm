"""
Registry regression tests for xAI entries in the model cost map.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[4]
PRICES_PATH = REPO_ROOT / "model_prices_and_context_window.json"
BACKUP_PRICES_PATH = REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json"

# https://docs.x.ai/developers/model-capabilities/text/multi-agent
# "The multi-agent model does not work with the OpenAI Chat Completions API."
RESPONSES_ONLY_MODELS = (
    "xai/grok-4.20-multi-agent-0309",
    "xai/grok-4.20-multi-agent-beta-0309",
)

MAP_PATHS = (PRICES_PATH, BACKUP_PRICES_PATH)


@pytest.fixture(scope="module", params=[p.name for p in MAP_PATHS])
def cost_map(request: pytest.FixtureRequest) -> dict:
    path = next(p for p in MAP_PATHS if p.name == request.param)
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("model", RESPONSES_ONLY_MODELS)
def test_multi_agent_models_are_responses_only(cost_map: dict, model: str):
    entry = cost_map[model]
    assert entry["supported_endpoints"] == ["/v1/responses"]
    assert entry["mode"] == "responses"


def test_surviving_xai_chat_models_still_serve_chat_completions(cost_map: dict):
    """Guard against the removal above over-reaching into live models."""
    chat_models = [
        key
        for key, value in cost_map.items()
        if isinstance(value, dict) and value.get("litellm_provider") == "xai" and value.get("mode") == "chat"
    ]
    assert "xai/grok-4.3" in chat_models
    assert "xai/grok-4.6" in chat_models


def test_both_cost_maps_agree_on_xai_entries():
    prices = json.loads(PRICES_PATH.read_text(encoding="utf-8"))
    backup = json.loads(BACKUP_PRICES_PATH.read_text(encoding="utf-8"))
    xai_keys = {k for k, v in prices.items() if isinstance(v, dict) and v.get("litellm_provider") == "xai"}
    assert xai_keys
    assert {k: prices[k] for k in xai_keys} == {k: backup[k] for k in xai_keys}
