"""
Anthropic stacks its geo and speed pricing modifiers with the prompt-caching
multipliers, so cache reads and cache writes bill off the modified input rate,
not the standard one: a fast-mode cache read on Claude Opus 5 costs
0.1 x $10/MTok, and a ``us`` data-residency cache write costs 1.25 x $5.50/MTok.

Regression for the earlier behavior, which subtracted the cache portion before
applying the multiplier and added it back unscaled, understating spend on every
cache-heavy fast-mode or regionalized request.

https://platform.claude.com/docs/en/build-with-claude/fast-mode#pricing
"""

import pytest

import litellm
from litellm.types.utils import ModelResponse, PromptTokensDetailsWrapper, Usage

INPUT_COST = 5e-06
OUTPUT_COST = 2.5e-05
CACHE_READ_COST = 5e-07
CACHE_WRITE_COST = 6.25e-06

UNCACHED_PROMPT_TOKENS = 1000
CACHE_READ_TOKENS = 2000
CACHE_WRITE_TOKENS = 500
COMPLETION_TOKENS = 100

STANDARD_COST = (
    UNCACHED_PROMPT_TOKENS * INPUT_COST
    + CACHE_READ_TOKENS * CACHE_READ_COST
    + CACHE_WRITE_TOKENS * CACHE_WRITE_COST
    + COMPLETION_TOKENS * OUTPUT_COST
)


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


PROMPT_TOKENS = UNCACHED_PROMPT_TOKENS + CACHE_READ_TOKENS + CACHE_WRITE_TOKENS


def _cost(speed: str | None = None, inference_geo: str | None = None) -> float:
    """Mirrors the Usage that ``AnthropicConfig.calculate_usage`` builds: a
    ``prompt_tokens`` total that already includes the cache tokens, with the
    uncached remainder in ``prompt_tokens_details.text_tokens``."""
    usage = Usage(
        prompt_tokens=PROMPT_TOKENS,
        completion_tokens=COMPLETION_TOKENS,
        total_tokens=PROMPT_TOKENS + COMPLETION_TOKENS,
        cache_read_input_tokens=CACHE_READ_TOKENS,
        cache_creation_input_tokens=CACHE_WRITE_TOKENS,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            cached_tokens=CACHE_READ_TOKENS,
            cache_creation_tokens=CACHE_WRITE_TOKENS,
            text_tokens=UNCACHED_PROMPT_TOKENS,
        ),
        speed=speed,
        inference_geo=inference_geo,
    )
    response = ModelResponse(
        id="test-id",
        created=1234567890,
        model="claude-opus-5",
        object="chat.completion",
        choices=[],
        usage=usage,
    )
    return litellm.completion_cost(
        completion_response=response,
        model="claude-opus-5",
        custom_llm_provider="anthropic",
    )


def test_standard_request_is_unmodified():
    assert _cost() == pytest.approx(STANDARD_COST)


def test_global_geo_is_not_treated_as_a_priced_region():
    assert _cost(inference_geo="global") == pytest.approx(STANDARD_COST)


def test_fast_mode_scales_cache_read_and_cache_write():
    assert _cost(speed="fast") == pytest.approx(STANDARD_COST * 2.0)


def test_us_data_residency_scales_cache_read_and_cache_write():
    assert _cost(inference_geo="us") == pytest.approx(STANDARD_COST * 1.1)


def test_fast_mode_and_data_residency_stack():
    assert _cost(speed="fast", inference_geo="us") == pytest.approx(STANDARD_COST * 2.2)


def test_cache_tokens_carry_the_same_multiplier_as_uncached_tokens():
    """The bug was cache-specific, so pin the cache slice on its own: the delta
    between a fast and a standard request must include the cache portion."""
    cache_portion = CACHE_READ_TOKENS * CACHE_READ_COST + CACHE_WRITE_TOKENS * CACHE_WRITE_COST

    assert _cost(speed="fast") - _cost() == pytest.approx(STANDARD_COST)
    assert _cost(speed="fast") - _cost() > cache_portion


def test_unknown_model_falls_back_to_no_multiplier():
    from litellm.llms.anthropic.cost_calculation import _pricing_modifier_multiplier

    usage = Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20, speed="fast")

    assert _pricing_modifier_multiplier(model="not-a-real-claude-model", usage=usage) == 1.0


def test_model_without_modifier_entry_is_unscaled():
    from litellm.llms.anthropic.cost_calculation import _pricing_modifier_multiplier

    usage = Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20, speed="fast")

    assert _pricing_modifier_multiplier(model="claude-haiku-4-5", usage=usage) == 1.0
