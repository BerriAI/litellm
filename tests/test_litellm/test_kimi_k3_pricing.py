"""
Regression test: Kimi K3 had no entry in either model-cost map, so every call
routed through ``openrouter/moonshotai/kimi-k3`` or ``novita/moonshotai/kimi-k3``
logged $0 spend instead of its real cost.

These tests pin the published per-token costs in both the primary price map and
the ``litellm/`` backup, keep the two files in agreement, and assert that
``cost_per_token`` resolves to a non-zero, exact dollar amount so the silent
$0 regression cannot come back.

``openrouter/moonshotai/kimi-k3`` deliberately carries no ``max_output_tokens``:
OpenRouter reports ``top_provider.max_completion_tokens: null`` for this model,
so no published value exists. The assertion below pins that absence so nobody
backfills it with the context length by assumption.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm.cost_calculator import cost_per_token

OPENROUTER_MODEL = "openrouter/moonshotai/kimi-k3"
NOVITA_MODEL = "novita/moonshotai/kimi-k3"

EXPECTED_INPUT_COST = 3e-06
EXPECTED_OUTPUT_COST = 1.5e-05
EXPECTED_CACHE_READ_COST = 3e-07
EXPECTED_MAX_INPUT_TOKENS = 1048576


def _load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _backup_path() -> str:
    return os.path.join(
        os.path.dirname(litellm.__file__),
        "model_prices_and_context_window_backup.json",
    )


def _main_path() -> str:
    return os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "model_prices_and_context_window.json",
    )


@pytest.fixture(scope="module")
def use_local_model_cost_map():
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")

    from litellm.utils import _invalidate_model_cost_lowercase_map

    original_model_cost = litellm.model_cost
    litellm.model_cost = _load_json(_backup_path())
    litellm.get_model_info.cache_clear()
    _invalidate_model_cost_lowercase_map()
    try:
        yield litellm
    finally:
        litellm.model_cost = original_model_cost
        litellm.get_model_info.cache_clear()
        _invalidate_model_cost_lowercase_map()
        monkeypatch.undo()


@pytest.mark.parametrize("model", [OPENROUTER_MODEL, NOVITA_MODEL])
class TestKimiK3PricingData:
    """Both price maps must carry the published Kimi K3 costs, and agree."""

    def test_costs_present_in_both_maps(self, model):
        for path in (_main_path(), _backup_path()):
            entry = _load_json(path)[model]
            assert entry["input_cost_per_token"] == EXPECTED_INPUT_COST
            assert entry["output_cost_per_token"] == EXPECTED_OUTPUT_COST
            assert entry["cache_read_input_token_cost"] == EXPECTED_CACHE_READ_COST
            assert entry["max_input_tokens"] == EXPECTED_MAX_INPUT_TOKENS
            assert entry["mode"] == "chat"
            assert entry["supports_prompt_caching"] is True
            assert entry["supports_vision"] is True
            assert entry["supports_video_input"] is True

    def test_maps_agree(self, model):
        assert _load_json(_main_path())[model] == _load_json(_backup_path())[model]

    def test_output_costs_more_than_input(self, model):
        entry = _load_json(_backup_path())[model]
        assert entry["output_cost_per_token"] > entry["input_cost_per_token"]
        assert entry["cache_read_input_token_cost"] < entry["input_cost_per_token"]


def test_openrouter_entry_has_no_unverified_max_output_tokens():
    for path in (_main_path(), _backup_path()):
        assert "max_output_tokens" not in _load_json(path)[OPENROUTER_MODEL]


def test_novita_entry_max_output_tokens():
    for path in (_main_path(), _backup_path()):
        assert _load_json(path)[NOVITA_MODEL]["max_output_tokens"] == EXPECTED_MAX_INPUT_TOKENS


@pytest.mark.parametrize(
    "model,provider",
    [(OPENROUTER_MODEL, "openrouter"), (NOVITA_MODEL, "novita")],
)
def test_get_model_info_surfaces_costs(use_local_model_cost_map, model, provider):
    info = use_local_model_cost_map.get_model_info(model)

    assert info["litellm_provider"] == provider
    assert info["input_cost_per_token"] == EXPECTED_INPUT_COST
    assert info["output_cost_per_token"] == EXPECTED_OUTPUT_COST
    assert info["cache_read_input_token_cost"] == EXPECTED_CACHE_READ_COST
    assert info["max_input_tokens"] == EXPECTED_MAX_INPUT_TOKENS


@pytest.mark.parametrize(
    "model,provider",
    [(OPENROUTER_MODEL, "openrouter"), (NOVITA_MODEL, "novita")],
)
def test_cost_per_token_is_not_zero(use_local_model_cost_map, model, provider):
    prompt_usd, completion_usd = cost_per_token(
        model=model,
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        custom_llm_provider=provider,
    )

    assert prompt_usd == pytest.approx(3.0)
    assert completion_usd == pytest.approx(15.0)
    assert prompt_usd + completion_usd > 0
