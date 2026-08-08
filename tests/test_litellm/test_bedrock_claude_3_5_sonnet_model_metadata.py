import json
from pathlib import Path

import pytest

BEDROCK_CLAUDE_3_5_SONNET_MODELS = [
    "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "anthropic.claude-3-5-sonnet-20241022-v2:0",
]

REGIONAL_PREFIXES = ["us.", "eu.", "apac."]

ABOVE_200K_FIELDS = [
    "input_cost_per_token_above_200k_tokens",
    "output_cost_per_token_above_200k_tokens",
    "cache_creation_input_token_cost_above_200k_tokens",
    "cache_read_input_token_cost_above_200k_tokens",
]


def _load(path):
    with open(path) as f:
        return json.load(f)


@pytest.mark.parametrize("model", BEDROCK_CLAUDE_3_5_SONNET_MODELS)
def test_bedrock_claude_3_5_sonnet_context_window(model):
    """Claude 3.5 Sonnet has a 200k context window on Bedrock."""
    json_path = Path(__file__).parents[2] / "model_prices_and_context_window.json"
    model_cost = _load(json_path)

    info = model_cost.get(model)
    assert (
        info is not None
    ), f"{model} not found in model_prices_and_context_window.json"

    assert info["litellm_provider"] == "bedrock"
    assert info["max_input_tokens"] == 200000


@pytest.mark.parametrize("model", BEDROCK_CLAUDE_3_5_SONNET_MODELS)
def test_bedrock_claude_3_5_sonnet_has_no_long_context_tier(model):
    """There is no >200k pricing tier for Claude 3.5 Sonnet."""
    json_path = Path(__file__).parents[2] / "model_prices_and_context_window.json"
    info = _load(json_path)[model]

    present = [field for field in ABOVE_200K_FIELDS if field in info]
    assert not present, f"{model} should not carry >200k pricing fields: {present}"


@pytest.mark.parametrize("model", BEDROCK_CLAUDE_3_5_SONNET_MODELS)
def test_bedrock_claude_3_5_sonnet_matches_regional_profiles(model):
    """The base entry and its regional copies describe the same model."""
    json_path = Path(__file__).parents[2] / "model_prices_and_context_window.json"
    model_cost = _load(json_path)
    info = model_cost[model]

    for prefix in REGIONAL_PREFIXES:
        regional = model_cost.get(prefix + model)
        if regional is None:
            continue
        assert (
            regional["max_input_tokens"] == info["max_input_tokens"]
        ), f"{prefix + model} and {model} disagree on max_input_tokens"


@pytest.mark.parametrize("model", BEDROCK_CLAUDE_3_5_SONNET_MODELS)
def test_bedrock_claude_3_5_sonnet_backup_matches_main(model):
    """Ensure the bundled model cost map stays in sync with the canonical file."""
    repo_root = Path(__file__).parents[2]
    main_cost = _load(repo_root / "model_prices_and_context_window.json")
    backup_cost = _load(
        repo_root / "litellm" / "model_prices_and_context_window_backup.json"
    )

    assert backup_cost.get(model) == main_cost.get(
        model
    ), f"{model} differs between main and backup model cost maps"
