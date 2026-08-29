"""
xAI retired eight slugs on 2026-05-15 but kept them resolvable: chat slugs redirect to
grok-4.3 and bill at grok-4.3's rates, while the grok-code-fast slugs are aliases of
grok-build-0.1 and bill at its rates, so the registry must price them that way or spend
tracking is wrong.

https://docs.x.ai/developers/migration/may-15-retirement
https://docs.x.ai/developers/models/grok-build-0.1
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[4]
PRICES_PATH = REPO_ROOT / "model_prices_and_context_window.json"
BACKUP_PRICES_PATH = REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json"
MAP_PATHS = (PRICES_PATH, BACKUP_PRICES_PATH)

REDIRECT_TARGET = "xai/grok-4.3"
REDIRECTED_SLUGS = (
    "xai/grok-3",
    "xai/grok-3-latest",
    "xai/grok-4",
    "xai/grok-4-0709",
    "xai/grok-4-1-fast-non-reasoning",
    "xai/grok-4-1-fast-non-reasoning-latest",
    "xai/grok-4-1-fast-reasoning",
    "xai/grok-4-1-fast-reasoning-latest",
    "xai/grok-4-fast-non-reasoning",
    "xai/grok-4-fast-reasoning",
    "xai/grok-4-latest",
)
CODE_REDIRECT_TARGET = "xai/grok-build-0.1"
CODE_SLUGS = (
    "xai/grok-code-fast",
    "xai/grok-code-fast-1",
    "xai/grok-code-fast-1-0825",
)
RETIREMENT_DATE = "2026-05-15"

BASE_COST_FIELDS = ("input_cost_per_token", "output_cost_per_token", "cache_read_input_token_cost")
TIER_COST_FIELDS = (
    "input_cost_per_token_above_200k_tokens",
    "output_cost_per_token_above_200k_tokens",
    "cache_read_input_token_cost_above_200k_tokens",
)
STALE_TIER_FIELDS = (
    "input_cost_per_token_above_128k_tokens",
    "output_cost_per_token_above_128k_tokens",
    "cache_read_input_token_cost_above_128k_tokens",
)


@pytest.fixture(scope="module", params=[p.name for p in MAP_PATHS])
def cost_map(request: pytest.FixtureRequest) -> dict:
    path = next(p for p in MAP_PATHS if p.name == request.param)
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("slug", REDIRECTED_SLUGS)
def test_redirected_slug_bills_at_the_target_rate(cost_map: dict, slug: str):
    target = cost_map[REDIRECT_TARGET]
    entry = cost_map[slug]
    for field in BASE_COST_FIELDS:
        assert entry[field] == target[field], field


@pytest.mark.parametrize("slug", CODE_SLUGS)
def test_code_slug_bills_at_grok_build_rate(cost_map: dict, slug: str):
    """grok-code-fast* are aliases of grok-build-0.1, not grok-4.3 redirects."""
    target = cost_map[CODE_REDIRECT_TARGET]
    entry = cost_map[slug]
    for field in (*BASE_COST_FIELDS, *TIER_COST_FIELDS):
        assert entry[field] == target[field], field


@pytest.mark.parametrize("slug", (*REDIRECTED_SLUGS, *CODE_SLUGS))
def test_redirected_slug_keeps_its_retirement_date(cost_map: dict, slug: str):
    assert cost_map[slug]["deprecation_date"] == RETIREMENT_DATE


@pytest.mark.parametrize("slug", REDIRECTED_SLUGS)
def test_no_slug_keeps_the_superseded_128k_tier(cost_map: dict, slug: str):
    """The 128k tier belonged to the retired model; grok-4.3 tiers at 200k."""
    for field in STALE_TIER_FIELDS:
        assert field not in cost_map[slug], field


@pytest.mark.parametrize("slug", REDIRECTED_SLUGS)
def test_redirected_slug_carries_the_target_tier_rates(cost_map: dict, slug: str):
    """The request executes as grok-4.3, so it is tiered at grok-4.3's 200k boundary."""
    target = cost_map[REDIRECT_TARGET]
    entry = cost_map[slug]
    for field in TIER_COST_FIELDS:
        assert entry[field] == target[field], field


def test_a_live_xai_model_is_untouched(cost_map: dict):
    """Guard against the repricing leaking onto models xAI still serves directly."""
    assert cost_map["xai/grok-4.6"]["input_cost_per_token"] != cost_map[REDIRECT_TARGET]["input_cost_per_token"]
    assert "deprecation_date" not in cost_map["xai/grok-4.6"]


def test_both_cost_maps_agree_on_the_redirected_slugs():
    prices = json.loads(PRICES_PATH.read_text(encoding="utf-8"))
    backup = json.loads(BACKUP_PRICES_PATH.read_text(encoding="utf-8"))
    for slug in (*REDIRECTED_SLUGS, *CODE_SLUGS, REDIRECT_TARGET, CODE_REDIRECT_TARGET):
        assert prices[slug] == backup[slug], slug


def test_every_retired_chat_slug_is_covered(cost_map: dict):
    """The list above must stay in step with what the registry marks retired."""
    marked = {
        key
        for key, entry in cost_map.items()
        if isinstance(entry, dict)
        and entry.get("litellm_provider") == "xai"
        and entry.get("deprecation_date") == RETIREMENT_DATE
        and entry.get("mode") == "chat"
    }
    assert marked == {*REDIRECTED_SLUGS, *CODE_SLUGS}


def test_a_live_tiered_model_keeps_its_own_rates(cost_map: dict):
    """grok-4-1-fast is not retired, so it must keep its 128k tier and its own prices."""
    entry = cost_map["xai/grok-4-1-fast"]
    assert "deprecation_date" not in entry
    assert entry["input_cost_per_token_above_128k_tokens"] == 4e-07
    assert entry["input_cost_per_token"] == 2e-07
