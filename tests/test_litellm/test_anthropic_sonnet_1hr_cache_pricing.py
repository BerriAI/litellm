"""
Validate that the native (first-party) Anthropic Claude Sonnet 4.5 / 4.6 entries
carry the 1-hour prompt-cache write tier (`cache_creation_input_token_cost_above_1hr`)
in `model_prices_and_context_window.json`.

Anthropic's first-party API charges a separate 1-hour cache write rate (2x base
input) alongside the 5-minute write (1.25x base input) and cache read (0.1x base
input). The 1h/5m ratio is therefore 1.6. Without the 1-hour field, cost tracking
on 1-hour-TTL prompt caching falls back to the 5-minute rate and undercounts spend.

The native (non-bedrock) `claude-sonnet-4-5*` / `claude-sonnet-4-6` entries were
missing this field, while every sibling (`vertex_ai/`, `azure_ai/`, the
`*.anthropic.*` Bedrock profiles) and the older `claude-sonnet-4-20250514` already
carried it. This test guards against regression.

Values (per token):
  Sonnet base input 3e-06  -> 5m 3.75e-06, 1h 6e-06
  Sonnet 4.5 long-context (>200K) base 6e-06 -> 5m 7.5e-06, 1h 1.2e-05
"""

import json
import os

import pytest


@pytest.fixture(scope="module")
def model_data():
    json_path = os.path.join(
        os.path.dirname(__file__), "../../model_prices_and_context_window.json"
    )
    with open(json_path) as f:
        return json.load(f)


# (model_key, expected 1hr write per token, expected 1hr long-context tier or None)
EXPECTED = [
    ("claude-sonnet-4-5", 6e-06, 1.2e-05),
    ("claude-sonnet-4-5-20250929", 6e-06, 1.2e-05),
    ("claude-sonnet-4-5-20250929-v1:0", 6e-06, 1.2e-05),
    ("claude-sonnet-4-6", 6e-06, None),
]


@pytest.mark.parametrize("model_key, expected_1hr, expected_1hr_lc", EXPECTED)
def test_anthropic_sonnet_1hr_cache_write_pricing(
    model_data, model_key, expected_1hr, expected_1hr_lc
):
    assert model_key in model_data, f"Missing model entry: {model_key}"
    info = model_data[model_key]

    # Regular 1hr cache write rate must be present and exact.
    assert "cache_creation_input_token_cost_above_1hr" in info, (
        f"{model_key}: missing cache_creation_input_token_cost_above_1hr - "
        "Anthropic charges a separate 1-hour cache write rate for this model"
    )
    assert info["cache_creation_input_token_cost_above_1hr"] == expected_1hr, (
        f"{model_key}: 1hr cache write rate "
        f"{info['cache_creation_input_token_cost_above_1hr']} does not match "
        f"expected {expected_1hr}"
    )

    # 1hr write must be 1.6x the 5-minute write (Anthropic 2x-base / 1.25x-base).
    ratio = (
        info["cache_creation_input_token_cost_above_1hr"]
        / info["cache_creation_input_token_cost"]
    )
    assert (
        abs(ratio - 1.6) < 1e-9
    ), f"{model_key}: 1hr/5min ratio is {ratio}, expected 1.6"

    # Long-context (>200K) 1hr tier, where the model publishes a >200K tier.
    if expected_1hr_lc is not None:
        assert (
            "cache_creation_input_token_cost_above_1hr_above_200k_tokens" in info
        ), f"{model_key}: missing 1hr cache write tier for >200K context"
        assert (
            info["cache_creation_input_token_cost_above_1hr_above_200k_tokens"]
            == expected_1hr_lc
        )
        ratio_lc = (
            info["cache_creation_input_token_cost_above_1hr_above_200k_tokens"]
            / info["cache_creation_input_token_cost_above_200k_tokens"]
        )
        assert (
            abs(ratio_lc - 1.6) < 1e-9
        ), f"{model_key}: long-context 1hr/5min ratio is {ratio_lc}, expected 1.6"
    else:
        assert "cache_creation_input_token_cost_above_1hr_above_200k_tokens" not in info


CLAUDE_3_EXPECTED = [
    ("claude-3-haiku-20240307", 5e-07),
    ("claude-3-opus-20240229", 3e-05),
]


@pytest.mark.parametrize("model_key, expected_1hr", CLAUDE_3_EXPECTED)
def test_claude_3_1hr_cache_write_pricing(model_data, model_key, expected_1hr):
    """Haiku 3 and Opus 3 both carried Sonnet's 6e-06 1hr rate, overbilling Haiku 3
    1-hour cache writes 12x and underbilling Opus 3 5x."""
    info = model_data[model_key]

    assert info["cache_creation_input_token_cost_above_1hr"] == expected_1hr


@pytest.mark.parametrize("model_key, expected_1hr", CLAUDE_3_EXPECTED)
def test_backup_matches_main_for_claude_3_1hr_cache_write(model_key, expected_1hr):
    json_path = os.path.join(
        os.path.dirname(__file__),
        "../../litellm/model_prices_and_context_window_backup.json",
    )
    with open(json_path) as f:
        backup = json.load(f)

    assert (
        backup[model_key]["cache_creation_input_token_cost_above_1hr"] == expected_1hr
    )


def test_first_party_anthropic_1hr_cache_writes_are_2x_base_input(model_data):
    """Anthropic charges 1-hour cache writes at 2x base input for every first-party
    model, so any entry that drifts off that multiple is a copy-paste error."""
    offenders = tuple(
        (
            model_key,
            info["input_cost_per_token"],
            info["cache_creation_input_token_cost_above_1hr"],
        )
        for model_key, info in model_data.items()
        if isinstance(info, dict)
        and info.get("litellm_provider") == "anthropic"
        and info.get("input_cost_per_token")
        and info.get("cache_creation_input_token_cost_above_1hr")
        and abs(
            info["cache_creation_input_token_cost_above_1hr"]
            - 2 * info["input_cost_per_token"]
        )
        > 1e-12
    )

    assert offenders == (), f"1hr cache write is not 2x base input for: {offenders}"
