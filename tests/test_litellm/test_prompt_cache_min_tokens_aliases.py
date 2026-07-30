"""Regression: an Anthropic model served through a reseller alias had no
`prompt_cache_min_tokens`, so `get_prompt_cache_min_tokens` fell back
to DEFAULT_MINIMUM_PROMPT_CACHE_TOKEN_COUNT (1024). Where the real minimum
is higher, `is_prompt_caching_valid_prompt` then returned True for a prompt
Anthropic will not cache, and Anthropic returns no error in that case.
"""
import pytest
from litellm.constants import DEFAULT_MINIMUM_PROMPT_CACHE_TOKEN_COUNT
from litellm.utils import get_prompt_cache_min_tokens
import litellm

@pytest.fixture(autouse=True)
def local_model_cost_map(monkeypatch):
    original_model_cost = litellm.model_cost
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm.model_cost = litellm.get_model_cost_map(url="")
    litellm.get_model_info.cache_clear()
    try:
        yield
    finally:
        litellm.model_cost = original_model_cost
        litellm.get_model_info.cache_clear()

# (alias, expected minimum). Every value already appears in the cost map on the
# canonical alias for the same model; these are the reseller entries missing it.
ALIASES_ABOVE_DEFAULT = [
    ("azure_ai/claude-haiku-4-5", 4096),
    ("azure_ai/claude-opus-4-5", 4096),
    ("azure_ai/claude-opus-4-6", 4096),
    ("azure_ai/claude-opus-4-7", 2048),
    ("openrouter/anthropic/claude-haiku-4.5", 4096),
    ("openrouter/anthropic/claude-opus-4.7", 2048),
    ("snowflake/claude-haiku-4-5", 4096),
    ("vercel_ai_gateway/anthropic/claude-haiku-4.5", 4096),
]

ALIASES_BELOW_DEFAULT = [
    ("azure_ai/claude-fable-5", 512),
    ("vertex_ai/claude-fable-5", 512),
]

@pytest.mark.parametrize("model,expected", ALIASES_ABOVE_DEFAULT)
def test_alias_minimum_is_not_the_default_when_the_real_one_is_higher(model, expected):
    """The failure is silent, so it has to be caught here rather than at runtime."""
    assert get_prompt_cache_min_tokens(model=model) == expected
    assert get_prompt_cache_min_tokens(model=model) > DEFAULT_MINIMUM_PROMPT_CACHE_TOKEN_COUNT

@pytest.mark.parametrize("model,expected", ALIASES_BELOW_DEFAULT)
def test_alias_minimum_is_not_the_default_when_the_real_one_is_lower(model, expected):
    """The opposite error: caching skipped for a prompt that would have cached."""
    assert get_prompt_cache_min_tokens(model=model) == expected
    assert get_prompt_cache_min_tokens(model=model) < DEFAULT_MINIMUM_PROMPT_CACHE_TOKEN_COUNT

@pytest.mark.parametrize("alias,canonical", [
    ("azure_ai/claude-haiku-4-5", "claude-haiku-4-5"),
    ("snowflake/claude-sonnet-4-6", "claude-sonnet-4-6"),
    ("openrouter/anthropic/claude-opus-4.5", "claude-opus-4-5"),
    ("vertex_ai/claude-fable-5", "claude-fable-5"),
])
def test_an_alias_agrees_with_its_canonical_model(alias, canonical):
    """The minimum is a property of the model, not of the surface serving it.
    The cost map already assumes this: the Bedrock, us., eu. and apac. variants
    all carry the same minimum as the direct entry.
    """
    assert get_prompt_cache_min_tokens(model=alias) == get_prompt_cache_min_tokens(
        model=canonical)

def test_minimums_are_non_monotonic_across_generations():
    """Newer is not always lower, which is why these cannot be inferred.
    If this ever passes by accident because everything collapsed to one value,
    the resolution is broken rather than the models having converged.
    """
    assert get_prompt_cache_min_tokens(model="claude-opus-5") == 512
    assert get_prompt_cache_min_tokens(model="claude-opus-4-8") == 1024
    assert get_prompt_cache_min_tokens(model="claude-opus-4-7") == 2048
    assert get_prompt_cache_min_tokens(model="claude-opus-4-6") == 4096


def test_an_unknown_model_still_falls_back_rather_than_raising():
    """The fallback is right for a model nobody has recorded.
    The bug was reaching it for models recorded elsewhere in the same file."""
    assert (get_prompt_cache_min_tokens(model="not-a-real-model-xyz") ==
            DEFAULT_MINIMUM_PROMPT_CACHE_TOKEN_COUNT)
