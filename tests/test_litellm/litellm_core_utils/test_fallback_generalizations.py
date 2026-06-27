"""
Tests for the declarative fallback-generalizations mechanism.

Covers both the pure module (litellm.litellm_core_utils.fallback_generalizations)
and its end-to-end wiring into provider routing (get_llm_provider) and model-info
resolution (get_model_info / supports_*).
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

import litellm
from litellm.litellm_core_utils.fallback_generalizations import (
    get_fallback_generalization_rules,
    match_fallback_generalization,
    set_fallback_generalizations,
)


@pytest.fixture
def restore_generalizations():
    """Save the active rules, let the test install its own, then restore."""
    previous = list(get_fallback_generalization_rules())
    try:
        yield set_fallback_generalizations
    finally:
        set_fallback_generalizations(previous)


# --------------------------------------------------------------------------- #
# Pure module behaviour
# --------------------------------------------------------------------------- #


def test_match_returns_model_info_of_first_matching_rule(restore_generalizations):
    restore_generalizations(
        [
            {
                "name": "first",
                "pattern": r"^acme-",
                "model_info": {"litellm_provider": "openai", "tag": "first"},
            },
            {
                "name": "second",
                "pattern": r"^acme-pro-",
                "model_info": {"litellm_provider": "anthropic", "tag": "second"},
            },
        ]
    )
    # Both rules match "acme-pro-1"; first-in-list wins (documented precedence).
    matched = match_fallback_generalization("acme-pro-1")
    assert matched is not None
    assert matched["tag"] == "first"


def test_match_is_case_insensitive(restore_generalizations):
    restore_generalizations(
        [{"name": "r", "pattern": r"^claude-opus", "model_info": {"ok": True}}]
    )
    assert match_fallback_generalization("CLAUDE-OPUS-9-9") == {"ok": True}


def test_no_match_returns_none(restore_generalizations):
    restore_generalizations(
        [{"name": "r", "pattern": r"^claude-", "model_info": {"ok": True}}]
    )
    assert match_fallback_generalization("gpt-4o") is None
    assert match_fallback_generalization("") is None


def test_empty_rules_match_nothing(restore_generalizations):
    restore_generalizations([])
    assert match_fallback_generalization("claude-opus-9-9") is None


def test_malformed_rules_are_skipped_not_fatal(restore_generalizations):
    restore_generalizations(
        [
            "not-even-a-dict",
            None,
            {"name": "no-pattern", "model_info": {"x": 1}},
            {"name": "bad-info", "pattern": r"^a", "model_info": "not-a-dict"},
            {"name": "bad-regex", "pattern": r"^claude-(", "model_info": {"x": 1}},
            {"name": "good", "pattern": r"^claude-", "model_info": {"good": True}},
        ]
    )
    # Non-dict entries and dicts with bad fields are all skipped; the one
    # valid rule still matches.
    assert match_fallback_generalization("claude-opus-9-9") == {"good": True}


def test_setting_rules_invalidates_compiled_cache(restore_generalizations):
    restore_generalizations([{"name": "r", "pattern": r"^aaa", "model_info": {"v": 1}}])
    assert match_fallback_generalization("aaa-1") == {"v": 1}
    # Re-install different rules; the compiled cache must be rebuilt.
    set_fallback_generalizations(
        [{"name": "r", "pattern": r"^bbb", "model_info": {"v": 2}}]
    )
    assert match_fallback_generalization("aaa-1") is None
    assert match_fallback_generalization("bbb-1") == {"v": 2}


# --------------------------------------------------------------------------- #
# End-to-end: provider routing + model-info resolution
# --------------------------------------------------------------------------- #


@pytest.fixture
def myco_rule(restore_generalizations):
    """A self-contained rule carrying provider, pricing, context and capabilities."""
    restore_generalizations(
        [
            {
                "name": "myco",
                "pattern": r"^myco-[a-z]+-\d+$",
                "model_info": {
                    "litellm_provider": "openai",
                    "mode": "chat",
                    "input_cost_per_token": 1e-06,
                    "output_cost_per_token": 2e-06,
                    "max_input_tokens": 12345,
                    "max_output_tokens": 678,
                    "supports_vision": True,
                    "supports_function_calling": True,
                },
            }
        ]
    )
    return "myco-fast-1"


def test_unknown_model_routes_via_rule(myco_rule):
    _, provider, _, _ = litellm.get_llm_provider(model=myco_rule)
    assert provider == "openai"


def test_unknown_model_gets_pricing_context_and_capabilities(myco_rule):
    info = litellm.get_model_info(myco_rule)
    assert info["litellm_provider"] == "openai"
    assert info["input_cost_per_token"] == 1e-06
    assert info["output_cost_per_token"] == 2e-06
    assert info["max_input_tokens"] == 12345
    assert info["supports_vision"] is True


def test_supports_helper_reads_through_generalization(myco_rule):
    assert litellm.supports_vision(myco_rule) is True
    assert litellm.supports_function_calling(myco_rule) is True


def test_exact_entry_takes_precedence_over_rule(restore_generalizations):
    """An exact cost-map entry must win over a rule that also matches it."""
    restore_generalizations(
        [
            {
                "name": "shadow-gpt4o",
                "pattern": r"^gpt-4o$",
                "model_info": {
                    "litellm_provider": "anthropic",
                    "input_cost_per_token": 999.0,
                },
            }
        ]
    )
    info = litellm.get_model_info("gpt-4o")
    # Resolved from the real exact entry, not the shadowing rule.
    assert info["litellm_provider"] == "openai"
    assert info["input_cost_per_token"] != 999.0


def test_unknown_model_without_matching_rule_still_unmapped(restore_generalizations):
    restore_generalizations(
        [
            {
                "name": "claude",
                "pattern": r"^claude-",
                "model_info": {"litellm_provider": "anthropic"},
            }
        ]
    )
    with pytest.raises(Exception):
        litellm.get_model_info("totally-unknown-model-xyz")


# --------------------------------------------------------------------------- #
# Shipped anthropic-claude rule
# --------------------------------------------------------------------------- #


@pytest.fixture
def shipped_cost_map(monkeypatch):
    """Activate the bundled cost map so the shipped anthropic-claude rule is installed."""
    original_cost = litellm.model_cost
    previous_rules = list(get_fallback_generalization_rules())
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm.model_cost = litellm.get_model_cost_map(url="")
    litellm.get_model_info.cache_clear()
    try:
        yield
    finally:
        litellm.model_cost = original_cost
        litellm.get_model_info.cache_clear()
        set_fallback_generalizations(previous_rules)


def test_shipped_rule_prices_unmapped_claude_at_opus_tier(shipped_cost_map):
    """The shipped anthropic-claude rule prices an unmapped/new Claude at the Opus tier
    and flags it adaptive, so an unknown Claude is over-costed rather than silently free."""
    model = "claude-opus-9-9"
    assert model not in litellm.model_cost
    info = litellm.get_model_info(model)
    assert info["litellm_provider"] == "anthropic"
    assert info["input_cost_per_token"] == 5e-06
    assert info["output_cost_per_token"] == 2.5e-05
    assert info["cache_creation_input_token_cost"] == 6.25e-06
    assert info["cache_read_input_token_cost"] == 5e-07
    assert info["supports_adaptive_thinking"] is True
