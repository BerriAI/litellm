import json
from pathlib import Path

import pytest

PRICING_FILES = (
    "model_prices_and_context_window.json",
    "litellm/model_prices_and_context_window_backup.json",
)

BEDROCK_BATCH_MODELS = (
    "qwen.qwen3-235b-a22b-2507-v1:0",
    "anthropic.claude-haiku-4-5-20251001-v1:0",
    "apac.anthropic.claude-haiku-4-5-20251001-v1:0",
    "au.anthropic.claude-haiku-4-5-20251001-v1:0",
    "eu.anthropic.claude-haiku-4-5-20251001-v1:0",
    "global.anthropic.claude-haiku-4-5-20251001-v1:0",
    "jp.anthropic.claude-haiku-4-5-20251001-v1:0",
    "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "anthropic.claude-sonnet-4-5-20250929-v1:0",
    "au.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "claude-sonnet-4-5-20250929-v1:0",
    "eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "jp.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
)


@pytest.mark.parametrize("pricing_file", PRICING_FILES)
@pytest.mark.parametrize("model", BEDROCK_BATCH_MODELS)
def test_bedrock_batch_pricing_is_half_of_on_demand(
    pricing_file: str, model: str
) -> None:
    model_cost_map = json.loads((Path(__file__).parents[2] / pricing_file).read_text())
    model_info = model_cost_map[model]

    assert model_info["input_cost_per_token_batches"] == pytest.approx(
        model_info["input_cost_per_token"] / 2
    )
    assert model_info["output_cost_per_token_batches"] == pytest.approx(
        model_info["output_cost_per_token"] / 2
    )
