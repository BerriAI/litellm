import json
from typing import Final

import pytest

from litellm.litellm_core_utils.llm_cost_calc.utils import get_service_tier_cost_key
from litellm.proxy.common_utils.codex_model_translation import (
    CodexCapabilities,
    CodexDeploymentCapabilities,
    CodexServiceTier,
    codex_fallback_entry,
    combined_capabilities,
    no_priced_deployment_lacks_tier,
)

_PROMPT: Final = "base prompt"
_SILENT: Final = CodexDeploymentCapabilities()
_ABSENT: Final = object()


def _entry(capabilities: CodexCapabilities, **kwargs: object) -> dict[str, object]:
    entry: Final = codex_fallback_entry(
        slug="m", display_name="M", priority=3, capabilities=capabilities, instructions=_PROMPT, **kwargs
    )
    return json.loads(entry.model_dump_json())


_CLAUDE_LIKE: Final = CodexCapabilities(
    context_window=1_000_000, supports_vision=True, supports_function_calling=True, supports_reasoning=True
)


def test_a_model_that_calls_tools_and_reasons_gets_codexs_own_coding_setup():
    entry: Final = _entry(_CLAUDE_LIKE)

    assert entry["apply_patch_tool_type"] == "freeform"
    assert entry["truncation_policy"] == {"mode": "tokens", "limit": 10000}
    assert entry["supports_reasoning_summary_parameter"] is True
    assert entry["supports_reasoning_summaries"] is True
    assert entry["input_modalities"] == ["text", "image"]
    assert entry["context_window"] == 1_000_000


@pytest.mark.parametrize(
    ("flag", "field", "by_value"),
    [
        ("supports_function_calling", "apply_patch_tool_type", {True: "freeform", False: None, None: None}),
        ("supports_reasoning", "supports_reasoning_summary_parameter", {True: True, False: False, None: _ABSENT}),
        ("supports_reasoning", "supports_reasoning_summaries", {True: True, False: False, None: False}),
        ("supports_parallel_function_calling", "supports_parallel_tool_calls", {True: True, False: False, None: False}),
    ],
)
@pytest.mark.parametrize("value", [True, False, None])
def test_each_codex_feature_follows_what_the_deployments_declare(flag, field, by_value, value):
    """A missing `supports_reasoning_summary_parameter` is Codex's own default (true, `default_true` in
    `codex-rs/protocol/src/openai_models.rs` at rust-v0.166.0), so a model nobody describes keeps it."""
    entry: Final = _entry(CodexCapabilities(**{flag: value}))

    assert entry.get(field, _ABSENT) == by_value[value]


@pytest.mark.parametrize(
    ("vision", "modalities"), [(False, ["text"]), (True, ["text", "image"]), (None, ["text", "image"])]
)
def test_images_are_offered_unless_a_deployment_cannot_read_them(vision, modalities):
    assert _entry(CodexCapabilities(supports_vision=vision))["input_modalities"] == modalities


def test_the_rest_of_the_entry_is_what_the_listing_decides():
    tier: Final = CodexServiceTier(id="flex", name="Flex", description="cheaper")

    entry: Final = _entry(
        CodexCapabilities(),
        service_tiers=(tier,),
        supports_search_tool=True,
        reasoning_efforts=("low", "high"),
        client_version="0.166.0",
    )

    assert (entry["slug"], entry["display_name"], entry["priority"], entry["base_instructions"]) == (
        "m",
        "M",
        3,
        _PROMPT,
    )
    assert entry["service_tiers"] == [{"id": "flex", "name": "Flex", "description": "cheaper"}]
    assert entry["supports_search_tool"] is True
    assert [level["effort"] for level in entry["supported_reasoning_levels"]] == ["low", "high"]
    assert entry["default_reasoning_level"] == "low"
    assert entry["context_window"] is None
    assert (entry["shell_type"], entry["visibility"], entry["supported_in_api"]) == ("unified_exec", "list", True)
    assert (entry["support_verbosity"], entry["use_responses_lite"]) == (False, False)
    assert "tool_mode" not in entry


_REQUIRED_SINCE_CODEX_0_140: Final = frozenset(
    {
        "slug",
        "display_name",
        "description",
        "supported_reasoning_levels",
        "shell_type",
        "visibility",
        "supported_in_api",
        "priority",
        "availability_nux",
        "upgrade",
        "base_instructions",
        "supports_reasoning_summaries",
        "support_verbosity",
        "default_verbosity",
        "apply_patch_tool_type",
        "truncation_policy",
        "supports_parallel_tool_calls",
        "experimental_supported_tools",
    }
)
_CODEX_ENUMS: Final = {
    "shell_type": {"unified_exec", "disabled"},
    "visibility": {"list", "hide", "none"},
    "apply_patch_tool_type": {"freeform", None},
}
_CODEX_MODALITIES: Final = {"text", "image"}
_CODEX_TRUNCATION_MODES: Final = {"bytes", "tokens"}


@pytest.mark.parametrize("described", [True, False], ids=["described", "undescribed"])
def test_every_entry_carries_the_fields_and_enum_values_every_supported_codex_decodes(described):
    """`codex-rs/protocol/src/openai_models.rs` ModelInfo at rust-v0.140.0 and rust-v0.166.0, the fields with
    no `#[serde(default)]` and the variants of InputModality, ConfigShellToolType, ModelVisibility,
    ApplyPatchToolType and TruncationMode in both, read on 2026-11-02. Codex rejects the whole catalog when
    one entry does not decode."""
    entry: Final = _entry(
        CodexCapabilities(
            context_window=128_000,
            supports_vision=False,
            supports_function_calling=True,
            supports_parallel_function_calling=True,
            supports_reasoning=True,
        )
        if described
        else CodexCapabilities()
    )

    assert entry.keys() >= _REQUIRED_SINCE_CODEX_0_140
    assert all(entry[field] in values for field, values in _CODEX_ENUMS.items())
    assert set(entry["input_modalities"]) <= _CODEX_MODALITIES and "text" in entry["input_modalities"]
    assert entry["truncation_policy"]["mode"] in _CODEX_TRUNCATION_MODES


def test_a_deployment_that_says_nothing_does_not_vote():
    declares: Final = CodexDeploymentCapabilities(
        supports_vision=True,
        supports_function_calling=True,
        supports_parallel_function_calling=True,
        supports_reasoning=True,
    )

    assert combined_capabilities((declares, _SILENT)) == CodexCapabilities(
        supports_vision=True,
        supports_function_calling=True,
        supports_parallel_function_calling=True,
        supports_reasoning=True,
    )
    assert combined_capabilities((_SILENT, _SILENT)) == CodexCapabilities()
    assert combined_capabilities(()) == CodexCapabilities()


@pytest.mark.parametrize(
    "flag",
    ["supports_vision", "supports_function_calling", "supports_parallel_function_calling", "supports_reasoning"],
)
def test_one_deployment_lacking_a_capability_takes_it_off_the_whole_name(flag):
    has: Final = CodexDeploymentCapabilities(**{flag: True})
    lacks: Final = CodexDeploymentCapabilities(**{flag: False})

    for order in ((has, lacks, _SILENT), (_SILENT, lacks, has)):
        assert getattr(combined_capabilities(order), flag) is False


@pytest.mark.parametrize(
    ("windows", "listed", "expected"),
    [
        ((200_000, 128_000, None), 400_000, 128_000),
        ((None, 64_000), 400_000, 64_000),
        ((None, None), 400_000, 400_000),
        ((), 400_000, 400_000),
        ((0, -1, 32_000), None, 32_000),
        ((0,), None, None),
    ],
)
def test_the_window_is_the_smallest_any_deployment_declares(windows, listed, expected):
    deployments: Final = tuple(CodexDeploymentCapabilities(max_input_tokens=window) for window in windows)

    assert combined_capabilities(deployments, listed).context_window == expected


def _priced(*tier_ids: str) -> CodexDeploymentCapabilities:
    return CodexDeploymentCapabilities(
        prices_tokens=True,
        priced_service_tier_keys=frozenset(
            get_service_tier_cost_key("input_cost_per_token", tier) for tier in tier_ids
        ),
    )


@pytest.mark.parametrize(
    ("deployments", "tier", "kept"),
    [
        ((_priced("priority"), _priced("priority", "flex")), "priority", True),
        ((_priced("priority"), _priced("flex")), "priority", False),
        ((_priced("priority"), _priced()), "priority", False),
        ((_priced("priority"), _SILENT), "priority", True),
        ((_SILENT, _SILENT), "priority", True),
        ((_priced("priority"),), "fast", True),
        ((_priced("flex"),), "fast", False),
        ((_priced("priority", "flex"),), "made-up-tier", False),
        ((), "priority", True),
    ],
)
def test_a_tier_stays_unless_a_deployment_that_prices_tokens_has_no_price_for_it(deployments, tier, kept):
    """A deployment with no token price at all (a subscription upstream, a deployment name the cost map does
    not know) says nothing about which tiers its upstream sells, so it does not vote."""
    assert no_priced_deployment_lacks_tier(tier, deployments) is kept
