import json
from pathlib import Path

import pytest

# Regression coverage for six routes that resolved to NO cost-map entry, which litellm
# treats as $0 rather than as an error: the request succeeds, the response is normal, and
# the spend log records ~$0 with no warning, metric, or exception. Observed in production
# as snowflake/claude-opus-4-7 logging $1.16 across 365 calls / 37.4M tokens against a real
# cost near $185.
#
# Generic schema and map-size checks cannot catch a re-introduction: deleting one of these
# entries, or corrupting a price to another valid float, keeps the file schema-valid. These
# assertions pin the exact values instead.

EXPECTED = {
    # Snowflake Cortex serves these Opus versions, but the map only carried the Opus 4.0-era
    # `snowflake/claude-4-opus`. Limits mirror the existing snowflake siblings (Cortex caps
    # output well below the upstream Anthropic models), not the anthropic/* entries.
    "snowflake/claude-opus-4-5": {
        "litellm_provider": "snowflake",
        "input_cost_per_token": 5e-06,
        "output_cost_per_token": 2.5e-05,
        "cache_read_input_token_cost": 5e-07,
        "max_input_tokens": 200000,
        "max_output_tokens": 16384,
    },
    "snowflake/claude-opus-4-6": {
        "litellm_provider": "snowflake",
        "input_cost_per_token": 5e-06,
        "output_cost_per_token": 2.5e-05,
        "cache_read_input_token_cost": 5e-07,
        "max_input_tokens": 200000,
        "max_output_tokens": 16384,
    },
    "snowflake/claude-opus-4-7": {
        "litellm_provider": "snowflake",
        "input_cost_per_token": 5e-06,
        "output_cost_per_token": 2.5e-05,
        "cache_read_input_token_cost": 5e-07,
        "max_input_tokens": 200000,
        "max_output_tokens": 16384,
    },
    # `inception/mercury-2` was already present; only the OpenRouter-routed variant was missing.
    "openrouter/inception/mercury-2": {
        "litellm_provider": "openrouter",
        "input_cost_per_token": 2.5e-07,
        "output_cost_per_token": 7.5e-07,
        "cache_read_input_token_cost": 2.5e-08,
        "max_input_tokens": 128000,
        "max_output_tokens": 50000,
    },
    # `openrouter/openai/gpt-5.2` was already present, so 5.1 and 5.5 were plain gaps.
    # OpenRouter passes OpenAI list pricing through unchanged.
    "openrouter/openai/gpt-5.1": {
        "litellm_provider": "openrouter",
        "input_cost_per_token": 1.25e-06,
        "output_cost_per_token": 1e-05,
        "cache_read_input_token_cost": 1.25e-07,
        "max_input_tokens": 272000,
        "max_output_tokens": 128000,
    },
    "openrouter/openai/gpt-5.5": {
        "litellm_provider": "openrouter",
        "input_cost_per_token": 5e-06,
        "output_cost_per_token": 3e-05,
        "cache_read_input_token_cost": 5e-07,
        "max_input_tokens": 1050000,
        "max_output_tokens": 128000,
        # gpt-5.5's window runs to 1.05M, and OpenAI charges a higher rate past 272k.
        # `_get_token_base_cost` only applies those rates when these fields exist, so
        # omitting them silently UNDERCHARGES every long-context request - the same
        # silent under-recording this file exists to prevent, one tier down. 5.1 and the
        # existing 5.2 entry cap at 272k and are correctly untiered.
        "input_cost_per_token_above_272k_tokens": 1e-05,
        "output_cost_per_token_above_272k_tokens": 4.5e-05,
        "cache_read_input_token_cost_above_272k_tokens": 1e-06,
    },
}

# Adaptive-thinking Claude versions, mirroring their canonical anthropic/* entries.
# 4-5 predates adaptive thinking and must NOT claim it.
ADAPTIVE = {"snowflake/claude-opus-4-6", "snowflake/claude-opus-4-7"}

# Opus 4.7/4.8 reject top_p/top_k and temperature != 1, and litellm's drop/raise gating for
# that is cost-map driven - so every variant must carry an explicit
# `supports_sampling_params: false`. Enforced repo-wide by
# test_claude_fable_5_config.py::test_sampling_params_flag_on_all_models_that_removed_them,
# which caught this entry missing the flag; asserted here too so the reason travels with the
# entry rather than living only in an unrelated file.
NO_SAMPLING_PARAMS = {"snowflake/claude-opus-4-7"}

REPO_ROOT = Path(__file__).parents[2]
MAIN_PATH = REPO_ROOT / "model_prices_and_context_window.json"
BACKUP_PATH = REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json"


def _load(path):
    with open(path) as f:
        return json.load(f)


@pytest.mark.parametrize("model", sorted(EXPECTED))
def test_model_pricing_metadata(model):
    model_cost = _load(MAIN_PATH)

    info = model_cost.get(model)
    assert info is not None, f"{model} not found in model_prices_and_context_window.json"

    for key, expected in EXPECTED[model].items():
        assert info.get(key) == expected, f"{model}: {key} is {info.get(key)!r}, expected {expected!r}"

    assert info["mode"] == "chat"
    # A present-but-null price resolves to $0 exactly like a missing entry, which is the
    # failure this file exists to prevent - so assert truthiness, not just presence.
    assert info["input_cost_per_token"], f"{model} has a falsy input_cost_per_token"
    assert info["output_cost_per_token"], f"{model} has a falsy output_cost_per_token"

    assert info.get("supports_adaptive_thinking", False) is (model in ADAPTIVE)

    if model in NO_SAMPLING_PARAMS:
        assert info.get("supports_sampling_params") is False, (
            f"{model} must declare supports_sampling_params=false"
        )


@pytest.mark.parametrize("model", sorted(EXPECTED))
def test_long_context_entries_declare_tiered_pricing(model):
    """A >272k window with no `*_above_272k_tokens` rates undercharges long prompts.

    Generalised rather than pinned to gpt-5.5 alone, so adding another long-context route
    here without its tiered rates fails instead of silently under-recording spend.
    """
    info = _load(MAIN_PATH)[model]
    if info["max_input_tokens"] <= 272000:
        pytest.skip(f"{model} caps at {info['max_input_tokens']}; no tier applies")
    for field in (
        "input_cost_per_token_above_272k_tokens",
        "output_cost_per_token_above_272k_tokens",
        "cache_read_input_token_cost_above_272k_tokens",
    ):
        assert info.get(field), f"{model} exceeds 272k tokens but has no {field}"


@pytest.mark.parametrize("model", sorted(EXPECTED))
def test_backup_matches_main(model):
    """The bundled backup map is what ships in the package, so it must carry these too."""
    main_cost = _load(MAIN_PATH)
    backup_cost = _load(BACKUP_PATH)

    assert backup_cost.get(model) == main_cost.get(
        model
    ), f"{model} differs between main and backup model cost maps"
