"""
Regression tests for the Cloudflare Workers AI text-generation catalog in the
model-cost map.

The Cloudflare list was badly stale (only 4 ancient entries). These tests pin
the newly added current Workers AI models (sourced from Cloudflare's live
``/ai/models/search?task=Text Generation`` catalog) and guard against the root
``model_prices_and_context_window.json`` and the bundled
``litellm/model_prices_and_context_window_backup.json`` drifting out of sync for
the ``cloudflare/`` namespace.
"""

import json
import os

import pytest

import litellm

ROOT_MAP = os.path.join(
    os.path.dirname(os.path.dirname(litellm.__file__)),
    "model_prices_and_context_window.json",
)
BACKUP_MAP = os.path.join(
    os.path.dirname(litellm.__file__),
    "model_prices_and_context_window_backup.json",
)


@pytest.fixture(autouse=True)
def _use_local_model_cost_map(monkeypatch):
    original_model_cost = litellm.model_cost
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm.model_cost = litellm.get_model_cost_map(url="")
    try:
        yield
    finally:
        litellm.model_cost = original_model_cost


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _cloudflare_keys(data: dict) -> set:
    return {k for k in data if k.startswith("cloudflare/")}


def test_glm_5_2_entry_is_present_and_well_formed():
    entry = litellm.model_cost["cloudflare/@cf/zai-org/glm-5.2"]
    assert entry["litellm_provider"] == "cloudflare"
    assert entry["mode"] == "chat"
    assert entry["supports_function_calling"] is True
    assert entry["input_cost_per_token"] > 0
    assert entry["output_cost_per_token"] > 0


def test_vision_model_is_flagged_supports_vision():
    entry = litellm.model_cost["cloudflare/@cf/meta/llama-3.2-11b-vision-instruct"]
    assert entry["litellm_provider"] == "cloudflare"
    assert entry.get("supports_vision") is True


def test_additional_current_models_are_present():
    for key in (
        "cloudflare/@cf/openai/gpt-oss-120b",
        "cloudflare/@cf/meta/llama-3.3-70b-instruct-fp8-fast",
    ):
        entry = litellm.model_cost[key]
        assert entry["litellm_provider"] == "cloudflare"
        assert entry["mode"] == "chat"
        assert entry["supports_function_calling"] is True
        assert entry["input_cost_per_token"] > 0
        assert entry["output_cost_per_token"] > 0


def test_root_and_backup_have_identical_cloudflare_keys():
    if not os.path.exists(ROOT_MAP):
        pytest.skip("root cost map only ships in source checkouts")
    assert _cloudflare_keys(_load(ROOT_MAP)) == _cloudflare_keys(_load(BACKUP_MAP))


def test_root_and_backup_cloudflare_entries_are_byte_for_byte_equal():
    if not os.path.exists(ROOT_MAP):
        pytest.skip("root cost map only ships in source checkouts")
    root = {k: v for k, v in _load(ROOT_MAP).items() if k.startswith("cloudflare/")}
    backup = {k: v for k, v in _load(BACKUP_MAP).items() if k.startswith("cloudflare/")}
    assert root == backup
