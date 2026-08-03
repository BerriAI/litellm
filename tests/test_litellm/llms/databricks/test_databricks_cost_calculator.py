"""
Regression tests for Databricks cost calculation.

The previous hand-rolled arithmetic billed every prompt token at the full
input rate and ignored cache / audio / reasoning fields on Usage. These tests
pin the generic_cost_per_token path so that regression cannot return silently.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

import litellm
from litellm.llms.databricks.cost_calculator import cost_per_token
from litellm.types.utils import PromptTokensDetailsWrapper, Usage

MODEL = "databricks/databricks-meta-llama-3-3-70b-instruct"
INPUT_COST = 5.0001e-07
OUTPUT_COST = 1.5000300000000002e-06
# Synthetic cache-read rate used only for this regression; real Databricks
# entries currently leave cache_read_input_token_cost unset.
CACHE_READ_COST = 1.0e-07


def _usage(prompt_tokens: int, cached_tokens: int, completion_tokens: int) -> Usage:
    return Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=cached_tokens),
    )


@pytest.fixture
def databricks_model_with_cache_rate():
    """
    Inject a temporary cache-read rate for the model under test so the
    generic path has a non-None cache_read_input_token_cost to apply.
    """
    original = litellm.model_cost
    try:
        with open("model_prices_and_context_window.json", "r") as f:
            model_cost_map = json.load(f)
    except FileNotFoundError:
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        model_cost_map = litellm.get_model_cost_map(url="")

    model_cost_map = dict(model_cost_map)
    entry = dict(model_cost_map.get(MODEL) or {})
    entry["litellm_provider"] = "databricks"
    entry["input_cost_per_token"] = INPUT_COST
    entry["output_cost_per_token"] = OUTPUT_COST
    entry["cache_read_input_token_cost"] = CACHE_READ_COST
    model_cost_map[MODEL] = entry
    bare = MODEL.split("/", 1)[-1]
    model_cost_map[bare] = entry
    litellm.model_cost = model_cost_map
    try:
        yield
    finally:
        litellm.model_cost = original


def test_cached_prompt_tokens_billed_at_cache_read_rate(databricks_model_with_cache_rate):
    prompt_tokens = 1000
    cached_tokens = 800
    completion_tokens = 50

    prompt_cost, completion_cost = cost_per_token(
        model=MODEL, usage=_usage(prompt_tokens, cached_tokens, completion_tokens)
    )

    expected_prompt_cost = (prompt_tokens - cached_tokens) * INPUT_COST + cached_tokens * CACHE_READ_COST
    assert prompt_cost == pytest.approx(expected_prompt_cost)
    assert completion_cost == pytest.approx(completion_tokens * OUTPUT_COST)

    # Old hand-rolled path would have charged the full input rate for every token.
    full_rate_cost = prompt_tokens * INPUT_COST
    assert prompt_cost < full_rate_cost


def test_no_cached_tokens_matches_full_input_rate(databricks_model_with_cache_rate):
    prompt_tokens = 100
    completion_tokens = 10

    prompt_cost, completion_cost = cost_per_token(
        model=MODEL, usage=_usage(prompt_tokens, 0, completion_tokens)
    )

    assert prompt_cost == pytest.approx(prompt_tokens * INPUT_COST)
    assert completion_cost == pytest.approx(completion_tokens * OUTPUT_COST)


def test_warm_call_cheaper_than_cold_call(databricks_model_with_cache_rate):
    prompt_tokens = 1000
    completion_tokens = 20

    cold_prompt_cost, _ = cost_per_token(
        model=MODEL, usage=_usage(prompt_tokens, 0, completion_tokens)
    )
    warm_prompt_cost, _ = cost_per_token(
        model=MODEL, usage=_usage(prompt_tokens, 900, completion_tokens)
    )

    assert warm_prompt_cost < cold_prompt_cost


def test_legacy_alias_still_resolves_to_pricing_entry(databricks_model_with_cache_rate):
    """Bare foundation-model aliases keep working after the generic-path switch."""
    prompt_tokens = 10
    completion_tokens = 2
    # Register the remapped key the old alias path used.
    litellm.model_cost["databricks-meta-llama-3-1-70b-instruct"] = {
        "litellm_provider": "databricks",
        "input_cost_per_token": INPUT_COST,
        "output_cost_per_token": OUTPUT_COST,
        "cache_read_input_token_cost": CACHE_READ_COST,
    }

    prompt_cost, completion_cost = cost_per_token(
        model="meta-llama-3.1-70b-instruct",
        usage=_usage(prompt_tokens, 0, completion_tokens),
    )
    assert prompt_cost == pytest.approx(prompt_tokens * INPUT_COST)
    assert completion_cost == pytest.approx(completion_tokens * OUTPUT_COST)
