import json
from pathlib import Path
from typing import Final

import pytest

from litellm import register_model
from litellm.cost_calculator import cost_per_token

REPO_ROOT: Final = Path(__file__).parents[2]
MAIN_PATH: Final = REPO_ROOT / "model_prices_and_context_window.json"
BACKUP_PATH: Final = REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json"

USD_PER_CREDIT: Final = 2.0
TOKENS_PER_MILLION: Final = 1_000_000

# Source: Snowflake Service Consumption Table, Tables 6(b) and 6(c), effective November 17, 2026
# https://www.snowflake.com/legal-files/CreditConsumptionTable.pdf
# Rates are AI Credits per million tokens, converted at the $2.00 per credit global-routing price on
# https://docs.snowflake.com/en/user-guide/snowflake-cortex/pricing
# Tuple order: input, output, cache read, cache write. None means the table lists no such rate.
# The table prints the claude-haiku-5-5 cache write as 0.063, which is 0.0625 rounded (1.25 x its 0.05 input rate).
PUBLISHED_CREDITS: Final = {
    "snowflake/openai-gpt-5-mini": (0.125, 1.0, 0.015, None),
    "snowflake/openai-gpt-5-nano": (0.03, 0.22, 0.005, None),
    "snowflake/llama3.1-8b": (0.11, 0.11, None, None),
    "snowflake/llama3.2-1b": (0.05, 0.05, None, None),
    "snowflake/llama3.2-3b": (0.075, 0.075, None, None),
    "snowflake/mistral-7b": (0.075, 0.1, None, None),
    "snowflake/claude-4-sonnet": (1.5, 7.5, 0.15, 1.875),
    "snowflake/claude-sonnet-4-5": (1.5, 7.5, 0.15, 1.875),
    "snowflake/claude-sonnet-4-6": (1.5, 7.5, 0.15, 1.875),
    "snowflake/claude-haiku-4-5": (0.5, 2.5, 0.05, 0.625),
    "snowflake/claude-opus-5-5": (2.0, 10.0, 0.1, 2.5),
    "snowflake/claude-sonnet-5-5": (1.0, 5.0, 0.05, 1.25),
    "snowflake/claude-haiku-5-5": (0.05, 0.25, 0.005, 0.0625),
    "snowflake/claude-sonnet-5": (1.0, 5.0, 0.1, 1.25),
    "snowflake/claude-opus-5": (2.5, 12.5, 0.25, 3.125),
    "snowflake/claude-opus-4-8": (2.5, 12.5, 0.25, 3.125),
    "snowflake/claude-opus-4-7": (2.5, 12.5, 0.25, 3.125),
    "snowflake/claude-opus-4-6": (2.5, 12.5, 0.25, 3.125),
    "snowflake/claude-opus-4-5": (2.5, 12.5, 0.25, 3.125),
}
RATE_FIELDS: Final = (
    "input_cost_per_token",
    "output_cost_per_token",
    "cache_read_input_token_cost",
    "cache_creation_input_token_cost",
)
CACHE_WRITE_MODELS: Final = tuple(key for key, credits in PUBLISHED_CREDITS.items() if credits[3] is not None)


def _snowflake_entries(path: Path) -> dict[str, dict[str, object]]:
    with open(path) as f:
        return {key: entry for key, entry in json.load(f).items() if key.startswith("snowflake/")}


@pytest.mark.parametrize("model", PUBLISHED_CREDITS)
def test_snowflake_entry_matches_published_rate(model: str) -> None:
    entry: Final = _snowflake_entries(MAIN_PATH)[model]
    for field, credits in zip(RATE_FIELDS, PUBLISHED_CREDITS[model]):
        if credits is None:
            continue
        assert entry[field] == pytest.approx(credits * USD_PER_CREDIT / TOKENS_PER_MILLION), field


def test_snowflake_entries_are_identical_in_the_backup_file() -> None:
    assert _snowflake_entries(MAIN_PATH) == _snowflake_entries(BACKUP_PATH)


@pytest.mark.parametrize("model", CACHE_WRITE_MODELS)
def test_snowflake_cache_writes_are_billed_at_the_cache_write_rate(model: str, local_model_cost_map: None) -> None:
    entry: Final = _snowflake_entries(MAIN_PATH)[model]
    register_model({model: entry})
    prompt_cost, _ = cost_per_token(
        model=model,
        custom_llm_provider="snowflake",
        prompt_tokens=TOKENS_PER_MILLION,
        completion_tokens=0,
        cache_creation_input_tokens=TOKENS_PER_MILLION,
    )
    assert prompt_cost == pytest.approx(entry["cache_creation_input_token_cost"] * TOKENS_PER_MILLION)
    assert prompt_cost > entry["input_cost_per_token"] * TOKENS_PER_MILLION


CLAUDE_MODELS: Final = tuple(key for key in PUBLISHED_CREDITS if key.startswith("snowflake/claude-"))


@pytest.mark.parametrize("model", CLAUDE_MODELS)
def test_snowflake_claude_entry_limits_and_cache_rates_are_consistent(model: str) -> None:
    entry: Final = _snowflake_entries(MAIN_PATH)[model]
    assert 0 < entry["max_output_tokens"] <= entry["max_input_tokens"]
    assert entry["supports_prompt_caching"] is True
    assert (
        entry["cache_read_input_token_cost"] < entry["input_cost_per_token"] < entry["cache_creation_input_token_cost"]
    )
