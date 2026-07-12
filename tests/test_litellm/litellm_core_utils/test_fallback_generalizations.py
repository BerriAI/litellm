"""
Tests for the declarative fallback-generalizations mechanism.

Covers the pure module (litellm.litellm_core_utils.fallback_generalizations): the
routing/capability rule split, install-time validation, capability unioning; and
its end-to-end wiring into provider routing (get_llm_provider) and model-info
resolution (get_model_info) including the shipped rules in the bundled cost map.
"""

import logging
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.fallback_generalizations import (
    get_fallback_generalization_rules,
    match_capability_generalizations,
    match_routing_generalization,
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


class _RecordingHandler(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@pytest.fixture
def warning_messages():
    handler = _RecordingHandler()
    previous_level = verbose_logger.level
    verbose_logger.setLevel(logging.WARNING)
    verbose_logger.addHandler(handler)
    try:
        yield handler.messages
    finally:
        verbose_logger.removeHandler(handler)
        verbose_logger.setLevel(previous_level)


# --------------------------------------------------------------------------- #
# Engine: routing rules
# --------------------------------------------------------------------------- #


def test_routing_inference_first_match_wins(restore_generalizations):
    restore_generalizations(
        [
            {"name": "first", "pattern": r"^acme-", "model_info": {"litellm_provider": "openai"}},
            {"name": "second", "pattern": r"^acme-pro-", "model_info": {"litellm_provider": "anthropic"}},
        ]
    )
    assert match_routing_generalization("acme-pro-1") == "openai"
    assert match_routing_generalization("gpt-4o") is None
    assert match_routing_generalization("") is None


def test_capability_rules_do_not_route(restore_generalizations):
    restore_generalizations([{"name": "caps", "pattern": r"^acme-", "model_info": {"supports_vision": True}}])
    assert match_routing_generalization("acme-pro-1") is None


def test_routing_match_is_case_insensitive(restore_generalizations):
    restore_generalizations(
        [{"name": "r", "pattern": r"^claude-opus", "model_info": {"litellm_provider": "anthropic"}}]
    )
    assert match_routing_generalization("CLAUDE-OPUS-9-9") == "anthropic"


# --------------------------------------------------------------------------- #
# Engine: capability rules
# --------------------------------------------------------------------------- #


def test_capability_union_is_last_wins_in_file_order(restore_generalizations):
    restore_generalizations(
        [
            {
                "name": "broad",
                "pattern": r"^acme-",
                "model_info": {"mode": "chat", "supports_vision": True, "max_input_tokens": 1000},
            },
            {
                "name": "narrow",
                "pattern": r"^acme-pro-",
                "model_info": {"supports_vision": False, "supports_reasoning": True},
            },
        ]
    )
    assert match_capability_generalizations("acme-pro-1") == {
        "mode": "chat",
        "supports_vision": False,
        "max_input_tokens": 1000,
        "supports_reasoning": True,
    }
    assert match_capability_generalizations("acme-basic-1") == {
        "mode": "chat",
        "supports_vision": True,
        "max_input_tokens": 1000,
    }


def test_routing_rules_are_excluded_from_capability_results(restore_generalizations):
    restore_generalizations(
        [
            {"name": "route", "pattern": r"^acme-", "model_info": {"litellm_provider": "openai"}},
            {"name": "caps", "pattern": r"^acme-pro-", "model_info": {"supports_vision": True}},
        ]
    )
    assert match_capability_generalizations("acme-pro-1") == {"supports_vision": True}
    assert match_capability_generalizations("acme-basic-1") is None


def test_no_capability_match_returns_none(restore_generalizations):
    restore_generalizations([{"name": "r", "pattern": r"^claude-", "model_info": {"ok": True}}])
    assert match_capability_generalizations("gpt-4o") is None
    assert match_capability_generalizations("") is None
    restore_generalizations([])
    assert match_capability_generalizations("claude-opus-9-9") is None


def test_reinstalling_rules_replaces_compiled_rules(restore_generalizations):
    restore_generalizations([{"name": "r", "pattern": r"^aaa", "model_info": {"v": 1}}])
    assert match_capability_generalizations("aaa-1") == {"v": 1}
    set_fallback_generalizations([{"name": "r", "pattern": r"^bbb", "model_info": {"v": 2}}])
    assert match_capability_generalizations("aaa-1") is None
    assert match_capability_generalizations("bbb-1") == {"v": 2}


# --------------------------------------------------------------------------- #
# Engine: install-time validation and legacy-schema shim
# --------------------------------------------------------------------------- #


def test_legacy_mixed_rule_acts_as_both_kinds(restore_generalizations):
    """A legacy rule mixing ``litellm_provider`` with capability keys routes AND
    contributes its full model_info (provider included) to the capability union."""
    restore_generalizations(
        [
            {
                "name": "legacy-mixed",
                "pattern": r"^acme-",
                "model_info": {"litellm_provider": "anthropic", "supports_vision": True},
            },
            {"name": "new-caps", "pattern": r"^acme-pro-", "model_info": {"supports_reasoning": True}},
        ]
    )
    assert match_routing_generalization("acme-pro-1") == "anthropic"
    assert match_capability_generalizations("acme-pro-1") == {
        "litellm_provider": "anthropic",
        "supports_vision": True,
        "supports_reasoning": True,
    }


LEGACY_MAIN_RULES = [
    {
        "name": "anthropic-claude-adaptive-thinking",
        "pattern": "(?:opus|sonnet|haiku)[-._](?:4[-._](?:[6-9]|[1-9]\\d)(?!\\d)|(?:[5-9]|[1-9]\\d{1,})[-._]\\d{1,2}(?!\\d))",
        "description": "Claude opus/sonnet/haiku at version 4.6 or higher: 4.6 through 4.99, then any 5.x, 6.x or later major. The minor is capped at two digits so an 8-digit date suffix such as claude-opus-4-20250514 is never read as a >= 4.6 minor. Turns on adaptive thinking for new families with no code change.",
        "extends": "anthropic-claude",
        "model_info": {"supports_adaptive_thinking": True},
    },
    {
        "name": "anthropic-claude",
        "pattern": "^claude-[a-z]+-\\d+[-.]\\d+(?:-\\d{8})?$",
        "description": "Any Claude family-major-minor id, optionally with an 8-digit date suffix, anchored to the whole name. Version-neutral fallback that gives an unmapped Claude provider routing and baseline capabilities; it carries no pricing, so cost stays on the standard unpriced behavior rather than a guessed number.",
        "model_info": {
            "litellm_provider": "anthropic",
            "mode": "chat",
            "max_input_tokens": 200000,
            "max_output_tokens": 64000,
            "max_tokens": 64000,
            "supports_function_calling": True,
            "supports_parallel_function_calling": True,
            "supports_vision": True,
            "supports_tool_choice": True,
            "supports_assistant_prefill": True,
            "supports_prompt_caching": True,
            "supports_response_schema": True,
            "supports_reasoning": True,
            "supports_pdf_input": True,
            "supports_system_messages": True,
        },
    },
]


def test_legacy_main_schema_keeps_unmapped_claude_working(restore_generalizations):
    """Pins the remote-map transition window: a released proxy running this engine
    against main's old-schema block (mixed provider+capability rule plus ``extends``,
    copied verbatim above) must keep unmapped-Claude inference and info resolution
    working until the new-schema JSON reaches main."""
    restore_generalizations([dict(rule) for rule in LEGACY_MAIN_RULES])
    litellm.get_model_info.cache_clear()

    _, provider, _, _ = litellm.get_llm_provider(model="claude-opus-9-9")
    assert provider == "anthropic"

    info = litellm.get_model_info("claude-opus-9-9")
    assert info["litellm_provider"] == "anthropic"
    assert info["supports_adaptive_thinking"] is True
    assert info["supports_function_calling"] is True
    assert info["max_input_tokens"] == 200000
    assert not info.get("input_cost_per_token")

    low = litellm.get_model_info("claude-opus-4-0")
    assert low["litellm_provider"] == "anthropic"
    assert low["supports_function_calling"] is True
    assert low.get("supports_adaptive_thinking") is None


def test_non_string_provider_rule_warns_and_is_skipped(restore_generalizations, warning_messages):
    restore_generalizations([{"name": "bad-provider", "pattern": r"^acme-", "model_info": {"litellm_provider": 42}}])
    assert any("bad-provider" in message for message in warning_messages)
    assert match_routing_generalization("acme-1") is None
    assert match_capability_generalizations("acme-1") is None


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
    assert match_capability_generalizations("claude-opus-9-9") == {"good": True}


# --------------------------------------------------------------------------- #
# End-to-end: provider routing + model-info resolution
# --------------------------------------------------------------------------- #


def test_unknown_model_routes_via_routing_rule(restore_generalizations):
    restore_generalizations([{"name": "myco", "pattern": r"^myco-", "model_info": {"litellm_provider": "openai"}}])
    _, provider, _, _ = litellm.get_llm_provider(model="myco-fast-1")
    assert provider == "openai"


def test_slash_prefixed_unknown_provider_raises_despite_matching_rule(restore_generalizations):
    """Routing rules are for bare-id inference only; an id under an unknown
    provider namespace must keep raising even when a rule matches inside it."""
    restore_generalizations([{"name": "myco", "pattern": r"myco-", "model_info": {"litellm_provider": "openai"}}])
    with pytest.raises(litellm.BadRequestError):
        litellm.get_llm_provider(model="mycoz/myco-fast-1")


def test_capability_info_backfills_requested_provider(restore_generalizations):
    restore_generalizations(
        [
            {
                "name": "beeco-caps",
                "pattern": r"^beeco-[a-z]+-\d+$",
                "model_info": {
                    "mode": "chat",
                    "max_input_tokens": 12345,
                    "supports_vision": True,
                    "supports_function_calling": True,
                },
            }
        ]
    )
    litellm.get_model_info.cache_clear()
    info = litellm.get_model_info("beeco-fast-1", custom_llm_provider="groq")
    assert info["litellm_provider"] == "groq"
    assert info["max_input_tokens"] == 12345
    assert info["supports_vision"] is True
    other = litellm.get_model_info("beeco-fast-1", custom_llm_provider="openai")
    assert other["litellm_provider"] == "openai"


def test_routing_only_match_does_not_resolve_model_info(restore_generalizations):
    restore_generalizations([{"name": "route", "pattern": r"^ceeco-", "model_info": {"litellm_provider": "openai"}}])
    litellm.get_model_info.cache_clear()
    with pytest.raises(Exception):
        litellm.get_model_info("ceeco-fast-1", custom_llm_provider="openai")


def test_exact_entry_takes_precedence_over_rule(restore_generalizations):
    restore_generalizations(
        [{"name": "shadow-gpt4o", "pattern": r"^gpt-4o$", "model_info": {"input_cost_per_token": 999.0}}]
    )
    litellm.get_model_info.cache_clear()
    info = litellm.get_model_info("gpt-4o")
    assert info["litellm_provider"] == "openai"
    assert info["input_cost_per_token"] != 999.0


# --------------------------------------------------------------------------- #
# Shipped rules (bundled cost map)
# --------------------------------------------------------------------------- #


@pytest.fixture
def shipped_cost_map(monkeypatch):
    """Activate the bundled cost map so the shipped rules are installed."""
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


def test_shipped_bare_claude_id_routes_to_anthropic(shipped_cost_map):
    _, provider, _, _ = litellm.get_llm_provider(model="claude-haiku-4-6")
    assert provider == "anthropic"


def test_shipped_bedrock_syntax_claude_id_routes_to_bedrock(shipped_cost_map):
    """Regression: a bedrock-syntax id must infer bedrock even when its version also
    matches an unanchored Anthropic capability pattern. The old first-match-wins engine
    routed global.anthropic.claude-haiku-4-6 to anthropic via the adaptive rule."""
    for model in [
        "global.anthropic.claude-haiku-4-6",
        "us.anthropic.claude-haiku-4-6",
        "anthropic.claude-haiku-4-6",
        "eu.anthropic.claude-opus-5-0",
    ]:
        assert model not in litellm.model_cost
        _, provider, _, _ = litellm.get_llm_provider(model=model)
        assert provider == "bedrock", model


def test_shipped_bedrock_rule_ignores_foreign_provider_prefixes(shipped_cost_map):
    """Regression: the unanchored bedrock-claude-ids pattern substring-matched
    bedrockz/anthropic.claude-..., so get_llm_provider inferred bedrock and keys
    scoped to bedrock/* passed the model access check for that id."""
    model = "bedrockz/anthropic.claude-3-5-sonnet-20240620"
    assert match_routing_generalization(model) is None
    with pytest.raises(litellm.BadRequestError):
        litellm.get_llm_provider(model=model)


def test_shipped_rules_resolve_unmapped_bedrock_claude_with_bedrock_provider(shipped_cost_map):
    model = "us.anthropic.claude-haiku-4-6"
    assert model not in litellm.model_cost
    info = litellm.get_model_info(model, custom_llm_provider="bedrock")
    assert info["litellm_provider"] == "bedrock"
    assert info["supports_adaptive_thinking"] is True
    assert info["supports_function_calling"] is True
    assert info["max_input_tokens"] == 200000
    assert info.get("supports_mid_conversation_system") is None
    assert not info.get("input_cost_per_token")
    assert not info.get("output_cost_per_token")


def test_shipped_rules_stack_adaptive_and_mid_conversation_flags(shipped_cost_map):
    model = "claude-opus-4-9"
    assert model not in litellm.model_cost
    info = litellm.get_model_info(model, custom_llm_provider="anthropic")
    assert info["litellm_provider"] == "anthropic"
    assert info["supports_adaptive_thinking"] is True
    assert info["supports_mid_conversation_system"] is True
    assert info["supports_function_calling"] is True


@pytest.mark.parametrize(
    "model,provider",
    [
        ("claude-opus-4-9@20260101", "vertex_ai"),
        ("databricks-claude-opus-5-1", "databricks"),
    ],
)
def test_shipped_rules_are_provider_neutral_for_unmapped_ids(shipped_cost_map, model, provider):
    assert model not in litellm.model_cost
    info = litellm.get_model_info(model, custom_llm_provider=provider)
    assert info["litellm_provider"] == provider
    assert info["supports_adaptive_thinking"] is True
    assert info["supports_mid_conversation_system"] is True
    assert info["supports_function_calling"] is True


@pytest.mark.parametrize(
    "model,provider,adaptive,mid_conversation",
    [
        ("us.anthropic.claude-opus-4-5", "bedrock", None, None),
        ("claude-haiku-4-6", "anthropic", True, None),
        ("claude-haiku-4-7", "anthropic", True, None),
        ("claude-haiku-4-8", "anthropic", True, True),
        ("claude-haiku-4-9", "anthropic", True, True),
        ("claude-haiku-4-10", "anthropic", True, True),
        ("claude-haiku-5-0", "anthropic", True, True),
        ("claude-sonnet-5-1", "anthropic", True, True),
    ],
)
def test_shipped_version_boundaries(shipped_cost_map, model, provider, adaptive, mid_conversation):
    assert model not in litellm.model_cost
    info = litellm.get_model_info(model, custom_llm_provider=provider)
    assert info["litellm_provider"] == provider
    assert info["supports_function_calling"] is True
    assert not info.get("input_cost_per_token")
    assert info.get("supports_adaptive_thinking") is adaptive, model
    assert info.get("supports_mid_conversation_system") is mid_conversation, model


def test_shipped_rules_cover_new_families_like_fable_at_5_plus(shipped_cost_map):
    """Both version gates accept any claude-<family>- id at major 5 or higher, bare
    major or major-minor, so a new family shaped like claude-fable-5 gets adaptive
    thinking and mid-conversation system support without a cost-map entry."""
    model = "claude-fable-5-1"
    assert model not in litellm.model_cost
    info = litellm.get_model_info(model, custom_llm_provider="anthropic")
    assert info["supports_mid_conversation_system"] is True
    assert info["supports_adaptive_thinking"] is True
    assert info["supports_function_calling"] is True


def test_shipped_rules_flag_bare_5_plus_majors_of_any_family(shipped_cost_map):
    """A bare 5+ major with no minor gets both flags at the rule level; the mapped
    claude-fable-5 entry itself still resolves from the cost map, so this pins the
    pattern via the capability union rather than get_model_info."""
    matched = match_capability_generalizations("claude-fable-5")
    assert matched is not None
    assert matched["supports_adaptive_thinking"] is True
    assert matched["supports_mid_conversation_system"] is True


def test_shipped_version_gates_are_family_agnostic_at_4x(shipped_cost_map):
    """Both version gates apply to any claude-<family>- id, 4.x included: a non-core
    family at 4.9 gets adaptive and mid-conversation, while the same family at 4.5
    gets baseline only. Only opus/sonnet/haiku ever shipped 4.x ids, so the
    family-agnostic 4.6+ gate changes nothing for real models."""
    high = litellm.get_model_info("claude-newfam-4-9", custom_llm_provider="anthropic")
    assert high["supports_adaptive_thinking"] is True
    assert high["supports_mid_conversation_system"] is True
    assert high["supports_function_calling"] is True

    low = litellm.get_model_info("claude-newfam-4-5", custom_llm_provider="anthropic")
    assert low.get("supports_adaptive_thinking") is None
    assert low.get("supports_mid_conversation_system") is None
    assert low["supports_function_calling"] is True


def test_shipped_rules_give_bare_majors_the_full_baseline_union(shipped_cost_map):
    """A bare-major unmapped id (no minor) resolves the same baseline union as its
    major-minor sibling: the baseline pattern's minor is optional, so claude-newt-5
    is not left with version flags but no mode, token limits, or capability facts."""
    model = "anthropic/claude-newt-5"
    assert model not in litellm.model_cost
    info = litellm.get_model_info(model)
    assert info["litellm_provider"] == "anthropic"
    assert info["mode"] == "chat"
    assert info["max_tokens"] == 64000
    assert info["supports_function_calling"] is True
    assert info["supports_adaptive_thinking"] is True
    assert info["supports_mid_conversation_system"] is True


def test_shipped_routing_rule_covers_bare_majors(shipped_cost_map):
    _, provider, _, _ = litellm.get_llm_provider(model="claude-newt-5")
    assert provider == "anthropic"


def test_shipped_adaptive_rule_requires_claude_prefix(shipped_cost_map):
    """A non-Claude name embedding a core-family 4.6+/5.x version substring must not
    resolve from the rules; serving it a zero-priced rule entry would silently
    swallow cost tracking for arbitrary custom deployment names."""
    model = "openai/team-sonnet-5-1-alias"
    assert model not in litellm.model_cost
    assert match_capability_generalizations("team-sonnet-5-1-alias") is None
    with pytest.raises(Exception):
        litellm.get_model_info(model)


def test_shipped_exact_entry_beats_rules(shipped_cost_map):
    model = "us.anthropic.claude-sonnet-4-6"
    assert model in litellm.model_cost
    info = litellm.get_model_info(model, custom_llm_provider="bedrock")
    assert info["litellm_provider"] == "bedrock_converse"
    assert info["input_cost_per_token"] == 3.3e-06
    assert info["max_input_tokens"] == 1000000
    assert info["supports_adaptive_thinking"] is True
    assert info.get("supports_mid_conversation_system") is None


def test_shipped_rules_lose_to_exact_entries_across_cost_ladder_variants(shipped_cost_map):
    """A route-mangled variant of an exactly-mapped model must never resolve from
    rules. The cost calculator tries model-name variants in order; a rule-derived
    unpriced entry served for an early variant (here bedrock/claude-haiku-4-5-20251001,
    whose bare form is exactly mapped under anthropic) would zero out the bill even
    though the exact priced bedrock entry is one variant later. An exactly-mapped id
    under a mismatched provider raises instead of resolving from rules."""
    from litellm import completion_cost
    from litellm.types.utils import ModelResponse, Usage

    assert "claude-haiku-4-5-20251001" in litellm.model_cost
    with pytest.raises(Exception):
        litellm.get_model_info("claude-haiku-4-5-20251001", custom_llm_provider="bedrock")

    entry = litellm.model_cost["us.anthropic.claude-haiku-4-5-20251001-v1:0"]
    response = ModelResponse(model="claude-haiku-4-5-20251001", usage=Usage(prompt_tokens=100, completion_tokens=50))
    cost = completion_cost(
        completion_response=response,
        model="bedrock/invoke/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        custom_llm_provider="bedrock",
    )
    assert cost == 100 * entry["input_cost_per_token"] + 50 * entry["output_cost_per_token"]
    assert cost > 0


def test_shipped_adaptive_rule_gates_on_version_not_pricing(shipped_cost_map):
    """The version-gated adaptive-thinking capability rule marks an unmapped Claude
    adaptive only from >= 4.6, including provider-prefixed ids the anchored routing
    rule cannot match, while leaving the dated Opus 4.0 form non-adaptive."""
    from litellm.llms.anthropic.common_utils import AnthropicModelInfo

    adaptive = "us.anthropic.claude-opus-4-9"
    non_adaptive = "us.anthropic.claude-opus-4-20250514"
    assert adaptive not in litellm.model_cost
    assert non_adaptive not in litellm.model_cost
    assert AnthropicModelInfo._is_adaptive_thinking_model(adaptive, "anthropic") is True
    assert AnthropicModelInfo._is_adaptive_thinking_model(non_adaptive, "anthropic") is False


def test_shipped_rules_resolve_unmapped_future_bedrock_claude_with_both_flags(shipped_cost_map):
    """An unmapped Bedrock Claude >= 4.8 resolves for custom_llm_provider="bedrock" with
    baseline capabilities, both version-gated flags, the bedrock provider backfilled, and
    no fabricated pricing."""
    model = "us.anthropic.claude-opus-4-9"
    assert model not in litellm.model_cost
    info = litellm.get_model_info(model, custom_llm_provider="bedrock")
    assert info["litellm_provider"] == "bedrock"
    assert info["supports_mid_conversation_system"] is True
    assert info["supports_adaptive_thinking"] is True
    assert info["supports_function_calling"] is True
    assert not info.get("input_cost_per_token")


def test_shipped_mid_conversation_gate_on_bedrock_ids(shipped_cost_map):
    """Bedrock-syntax ids gain ``supports_mid_conversation_system`` only from 4.8 upward,
    bare 5+ majors and new families included; 4.7-and-below Bedrock ids never gain it.
    The flag comes from the provider-neutral capability rule rather than a bedrock-scoped
    one, so the same gate covers native and vertex-shaped ids too."""
    for flagged in (
        "us.anthropic.claude-opus-4-8",
        "jp.anthropic.claude-opus-4-8",
        "anthropic.claude-sonnet-5",
        "us.anthropic.claude-fable-5",
        "anthropic.claude-sonnet-5-20260101-v1:0",
    ):
        matched = match_capability_generalizations(flagged)
        assert matched is not None, flagged
        assert matched["supports_mid_conversation_system"] is True, flagged
    for unflagged in (
        "us.anthropic.claude-opus-4-7",
        "us.anthropic.claude-sonnet-4-6",
        "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "anthropic.claude-3-5-sonnet-20240620-v1:0",
    ):
        matched = match_capability_generalizations(unflagged)
        assert matched is None or not matched.get("supports_mid_conversation_system"), unflagged
