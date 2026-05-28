"""
Validate Claude Opus 4.8 model configuration entries (LIT-3410).

Anthropic released Claude Opus 4.8 alongside the existing Opus 4.7 SKU with the
same pricing ($5 / MTok input, $25 / MTok output), 1M-token context window, and
128k max output. This file pins the entries in
``model_prices_and_context_window.json`` (and the bundled backup JSON the
runtime actually loads) so a future regression that drops one of the SKUs is
caught at unit-test time.
"""

import json
import os

import pytest

import litellm

MAIN_JSON = os.path.join(
    os.path.dirname(__file__), "..", "..", "model_prices_and_context_window.json"
)
BACKUP_JSON = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "litellm",
    "model_prices_and_context_window_backup.json",
)

# Anthropic-published SKU set for Opus 4.8 (anthropic + bedrock + vertex + azure_ai)
_NEW_4_8_KEYS = [
    "claude-opus-4-8",
    "anthropic.claude-opus-4-8",
    "us.anthropic.claude-opus-4-8",
    "eu.anthropic.claude-opus-4-8",
    "au.anthropic.claude-opus-4-8",
    "global.anthropic.claude-opus-4-8",
    "vertex_ai/claude-opus-4-8",
    "vertex_ai/claude-opus-4-8@default",
    "azure_ai/claude-opus-4-8",
]


def _load(path):
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def main_map():
    return _load(MAIN_JSON)


@pytest.fixture(scope="module")
def backup_map():
    return _load(BACKUP_JSON)


@pytest.mark.parametrize("key", _NEW_4_8_KEYS)
def test_opus_4_8_entry_present_in_main_and_backup(key, main_map, backup_map):
    """Every Opus 4.8 SKU must exist in BOTH JSON files — runtime loads the
    bundled backup at import time, so an entry only present in the source file
    will silently miss in pip-installed deployments."""
    assert key in main_map, f"{key} missing from model_prices_and_context_window.json"
    assert key in backup_map, (
        f"{key} missing from litellm/model_prices_and_context_window_backup.json"
    )


@pytest.mark.parametrize(
    "key,expected_input,expected_output",
    [
        # Anthropic native + Bedrock single-region + Vertex + Azure: $5 / $25 per MTok
        ("claude-opus-4-8", 5e-06, 2.5e-05),
        ("anthropic.claude-opus-4-8", 5e-06, 2.5e-05),
        ("global.anthropic.claude-opus-4-8", 5e-06, 2.5e-05),
        ("vertex_ai/claude-opus-4-8", 5e-06, 2.5e-05),
        ("vertex_ai/claude-opus-4-8@default", 5e-06, 2.5e-05),
        ("azure_ai/claude-opus-4-8", 5e-06, 2.5e-05),
        # AWS Bedrock cross-region inference profiles (10% uplift)
        ("us.anthropic.claude-opus-4-8", 5.5e-06, 2.75e-05),
        ("eu.anthropic.claude-opus-4-8", 5.5e-06, 2.75e-05),
        ("au.anthropic.claude-opus-4-8", 5.5e-06, 2.75e-05),
    ],
)
def test_opus_4_8_pricing(key, expected_input, expected_output, main_map, backup_map):
    """Pricing must match Anthropic's published rate card exactly. A drift in
    the bundled backup (which is what `litellm.get_model_info` reads in
    air-gapped/no-HTTP deployments) silently overcharges or undercharges every
    Opus 4.8 request, so we pin both files."""
    for label, mp in [("main", main_map), ("backup", backup_map)]:
        entry = mp[key]
        assert entry["input_cost_per_token"] == expected_input, (
            f"{label}.{key} input_cost_per_token={entry['input_cost_per_token']}, "
            f"expected {expected_input}"
        )
        assert entry["output_cost_per_token"] == expected_output, (
            f"{label}.{key} output_cost_per_token={entry['output_cost_per_token']}, "
            f"expected {expected_output}"
        )


@pytest.mark.parametrize("key", _NEW_4_8_KEYS)
def test_opus_4_8_context_window(key, main_map, backup_map):
    """All Opus 4.8 SKUs except Microsoft Foundry (azure_ai) ship the published
    1M-token context window with 128k max output; Microsoft Foundry caps input
    at 200k tokens per Anthropic's docs while keeping the same 128k output."""
    expected_in = 200000 if key.startswith("azure_ai/") else 1000000
    for label, mp in [("main", main_map), ("backup", backup_map)]:
        entry = mp[key]
        assert entry["max_input_tokens"] == expected_in, (
            f"{label}.{key} max_input_tokens={entry['max_input_tokens']}, "
            f"expected {expected_in}"
        )
        assert entry["max_output_tokens"] == 128000
        assert entry["max_tokens"] == 128000


@pytest.mark.parametrize(
    "key,expected_provider",
    [
        ("claude-opus-4-8", "anthropic"),
        ("anthropic.claude-opus-4-8", "bedrock_converse"),
        ("us.anthropic.claude-opus-4-8", "bedrock_converse"),
        ("eu.anthropic.claude-opus-4-8", "bedrock_converse"),
        ("au.anthropic.claude-opus-4-8", "bedrock_converse"),
        ("global.anthropic.claude-opus-4-8", "bedrock_converse"),
        ("vertex_ai/claude-opus-4-8", "vertex_ai-anthropic_models"),
        ("vertex_ai/claude-opus-4-8@default", "vertex_ai-anthropic_models"),
        ("azure_ai/claude-opus-4-8", "azure_ai"),
    ],
)
def test_opus_4_8_provider_routing(
    key, expected_provider, main_map, backup_map
):
    """Each SKU must route to the correct ``litellm_provider`` so that
    provider-specific cost / capability handlers (e.g. the bedrock_converse
    cross-region uplift code) fire."""
    for label, mp in [("main", main_map), ("backup", backup_map)]:
        assert mp[key]["litellm_provider"] == expected_provider, (
            f"{label}.{key} routed to {mp[key]['litellm_provider']}, "
            f"expected {expected_provider}"
        )


@pytest.mark.parametrize("key", _NEW_4_8_KEYS)
def test_opus_4_8_capability_flags(key, main_map, backup_map):
    """Opus 4.8 must declare the same capability surface as Opus 4.7 per the
    Anthropic models page (vision, tools, PDF input, prompt caching, computer
    use, reasoning / adaptive-thinking with xhigh + max + minimal effort tiers).
    Drift here breaks customer integrations that gate behaviour on
    ``model_info.supports_*``."""
    must_be_true = [
        "supports_function_calling",
        "supports_tool_choice",
        "supports_vision",
        "supports_pdf_input",
        "supports_prompt_caching",
        "supports_reasoning",
        "supports_response_schema",
        "supports_xhigh_reasoning_effort",
        "supports_max_reasoning_effort",
        "supports_minimal_reasoning_effort",
    ]
    for label, mp in [("main", main_map), ("backup", backup_map)]:
        entry = mp[key]
        for flag in must_be_true:
            assert entry.get(flag) is True, f"{label}.{key} {flag} is not True"
        # All Opus 4.x SKUs explicitly disable assistant prefill on the
        # response API surface
        assert entry.get("supports_assistant_prefill") is False, (
            f"{label}.{key} should declare supports_assistant_prefill=False"
        )


def test_opus_4_8_bedrock_output_config_effort_ceiling(main_map, backup_map):
    """The Bedrock variants must declare ``bedrock_output_config_effort_ceiling``
    so that ``normalize_bedrock_opus_output_config_effort`` (litellm/llms/bedrock/
    common_utils.py) caps an inbound ``effort=max`` down to ``xhigh`` when
    Bedrock doesn't yet expose the max tier (mirrors the existing 4-7 behaviour
    that test_bedrock_common_utils.py already pins)."""
    bedrock_keys = [k for k in _NEW_4_8_KEYS if "anthropic.claude-opus-4-8" in k]
    assert bedrock_keys, "expected at least one bedrock 4-8 key"
    for label, mp in [("main", main_map), ("backup", backup_map)]:
        for k in bedrock_keys:
            assert mp[k].get("bedrock_output_config_effort_ceiling") == "xhigh", (
                f"{label}.{k} effort ceiling is "
                f"{mp[k].get('bedrock_output_config_effort_ceiling')!r}, expected 'xhigh'"
            )


def test_opus_4_8_present_in_bundled_backup_via_loader():
    """End-to-end: the bundled backup model cost map (the one
    ``GetModelCostMap.load_local_model_cost_map`` returns and that
    ``litellm.get_model_info`` falls back to when the remote fetch fails or
    is disabled) must contain every new SKU with the published pricing.

    This is what paid-tier customers' billing code reads in air-gapped
    deployments. A missing SKU here translates directly to a "this model
    isn't mapped yet" production error.

    We exercise the loader directly (instead of ``litellm.get_model_info``)
    because module-import time has already taken a snapshot of ``litellm
    .model_cost`` from the remote URL by the time pytest reaches this
    test, and ``monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", ...)``
    after the fact cannot undo that snapshot."""
    from litellm.litellm_core_utils.get_model_cost_map import GetModelCostMap

    backup = GetModelCostMap.load_local_model_cost_map()
    expectations = {
        "claude-opus-4-8": ("anthropic", 5e-06, 2.5e-05),
        "anthropic.claude-opus-4-8": ("bedrock_converse", 5e-06, 2.5e-05),
        "us.anthropic.claude-opus-4-8": ("bedrock_converse", 5.5e-06, 2.75e-05),
        "eu.anthropic.claude-opus-4-8": ("bedrock_converse", 5.5e-06, 2.75e-05),
        "au.anthropic.claude-opus-4-8": ("bedrock_converse", 5.5e-06, 2.75e-05),
        "global.anthropic.claude-opus-4-8": ("bedrock_converse", 5e-06, 2.5e-05),
        "vertex_ai/claude-opus-4-8": ("vertex_ai-anthropic_models", 5e-06, 2.5e-05),
        "vertex_ai/claude-opus-4-8@default": (
            "vertex_ai-anthropic_models",
            5e-06,
            2.5e-05,
        ),
        "azure_ai/claude-opus-4-8": ("azure_ai", 5e-06, 2.5e-05),
    }
    for model, (provider, in_cost, out_cost) in expectations.items():
        entry = backup.get(model)
        assert entry is not None, (
            f"{model} missing from bundled backup model cost map"
        )
        assert entry["litellm_provider"] == provider, (
            f"{model}: litellm_provider={entry['litellm_provider']!r}, "
            f"expected {provider!r}"
        )
        assert entry["input_cost_per_token"] == in_cost, (
            f"{model}: input_cost_per_token mismatch"
        )
        assert entry["output_cost_per_token"] == out_cost, (
            f"{model}: output_cost_per_token mismatch"
        )


def test_opus_4_8_listed_in_bedrock_converse_models():
    """Routing for the Bedrock native invoke endpoint depends on
    ``litellm.constants.BEDROCK_CONVERSE_MODELS`` listing the bare model id.
    Without this entry, callers using the deployment id
    ``anthropic.claude-opus-4-8`` are dispatched to the legacy bedrock invoke
    transformation instead of bedrock_converse, which drops the
    ``output_config`` (effort) parameter."""
    from litellm import constants

    assert "anthropic.claude-opus-4-8" in constants.BEDROCK_CONVERSE_MODELS, (
        "anthropic.claude-opus-4-8 must be in BEDROCK_CONVERSE_MODELS "
        "(litellm/constants.py)"
    )
