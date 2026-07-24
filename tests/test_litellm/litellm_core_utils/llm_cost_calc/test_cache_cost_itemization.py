"""
Tests for issue #34309:
cache_read_cost / cache_creation_cost in cost_breakdown are null for providers
that report cache tokens via prompt_tokens_details (e.g. OpenAI Responses API)
instead of Anthropic-style top-level usage keys.
"""

import time

import pytest

import litellm
from litellm.cost_calculator import completion_cost
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.types.utils import (
    ModelResponse,
    PromptTokensDetailsWrapper,
    Usage,
)

MODEL = "openai/gpt-5-cacheitemize-test-34309"

# Standard rates used throughout these tests.
INPUT_RATE = 5e-6
OUTPUT_RATE = 30e-6
CACHE_READ_RATE = 5e-7
CACHE_WRITE_RATE = 6.25e-6


@pytest.fixture(autouse=True)
def _register_model():
    litellm.register_model(
        {
            MODEL: {
                "litellm_provider": "openai",
                "mode": "chat",
                "input_cost_per_token": INPUT_RATE,
                "output_cost_per_token": OUTPUT_RATE,
                "cache_read_input_token_cost": CACHE_READ_RATE,
                "cache_creation_input_token_cost": CACHE_WRITE_RATE,
            },
        },
    )
    yield


def _new_logging_obj() -> Logging:
    return Logging(
        model=MODEL,
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        call_type="completion",
        start_time=time.time(),
        litellm_call_id="test-34309",
        function_id="test-34309",
    )


def _openai_style_usage(cached: int, cache_write: int, fresh: int, out: int) -> Usage:
    """Build a Usage object that mimics the OpenAI Responses API:
    cache-read  -> prompt_tokens_details.cached_tokens
    cache-write -> prompt_tokens_details.cache_creation_tokens
    No top-level cache_read_input_tokens / cache_creation_input_tokens.
    """
    ptd = PromptTokensDetailsWrapper(cached_tokens=cached)
    # cache_creation_tokens is the standard field used by _parse_prompt_tokens_details
    setattr(ptd, "cache_creation_tokens", cache_write)
    return Usage(
        prompt_tokens=cached + cache_write + fresh,
        completion_tokens=out,
        total_tokens=cached + cache_write + fresh + out,
        prompt_tokens_details=ptd,
    )


def _response_with_usage(usage: Usage) -> ModelResponse:
    resp = ModelResponse()
    resp.model = MODEL
    resp.usage = usage  # type: ignore[attr-defined]
    return resp


def test_openai_style_cache_read_cost_itemized():
    """cache_read_cost is populated from prompt_tokens_details.cached_tokens."""
    cached = 16000
    fresh = 100
    out = 50
    usage = _openai_style_usage(cached=cached, cache_write=0, fresh=fresh, out=out)
    logging_obj = _new_logging_obj()

    completion_cost(
        completion_response=_response_with_usage(usage),
        model=MODEL,
        custom_llm_provider="openai",
        litellm_logging_obj=logging_obj,
    )

    breakdown = logging_obj.cost_breakdown
    assert breakdown is not None, "cost_breakdown was not populated"
    assert breakdown.get("cache_read_cost") == pytest.approx(
        cached * CACHE_READ_RATE, rel=1e-6
    ), (
        f"cache_read_cost not itemized: got {breakdown.get('cache_read_cost')!r}, "
        f"expected {cached * CACHE_READ_RATE}"
    )
    assert breakdown.get("cache_creation_cost") is None or breakdown.get(
        "cache_creation_cost"
    ) == pytest.approx(0.0, abs=1e-12)


def test_openai_style_cache_creation_cost_itemized():
    """cache_creation_cost is populated from prompt_tokens_details.cache_creation_tokens."""
    cache_write = 26022
    fresh = 100
    out = 50
    usage = _openai_style_usage(cached=0, cache_write=cache_write, fresh=fresh, out=out)
    logging_obj = _new_logging_obj()

    completion_cost(
        completion_response=_response_with_usage(usage),
        model=MODEL,
        custom_llm_provider="openai",
        litellm_logging_obj=logging_obj,
    )

    breakdown = logging_obj.cost_breakdown
    assert breakdown is not None
    assert breakdown.get("cache_creation_cost") == pytest.approx(
        cache_write * CACHE_WRITE_RATE, rel=1e-6
    ), (
        f"cache_creation_cost not itemized: got {breakdown.get('cache_creation_cost')!r}, "
        f"expected {cache_write * CACHE_WRITE_RATE}"
    )


def test_openai_style_both_cache_costs_itemized():
    """Both cache_read_cost and cache_creation_cost are populated when both counts present."""
    cached = 16000
    cache_write = 26022
    fresh = 100
    out = 50
    usage = _openai_style_usage(
        cached=cached, cache_write=cache_write, fresh=fresh, out=out
    )
    logging_obj = _new_logging_obj()

    total = completion_cost(
        completion_response=_response_with_usage(usage),
        model=MODEL,
        custom_llm_provider="openai",
        litellm_logging_obj=logging_obj,
    )

    breakdown = logging_obj.cost_breakdown
    assert breakdown is not None
    assert breakdown.get("cache_read_cost") == pytest.approx(
        cached * CACHE_READ_RATE, rel=1e-6
    )
    assert breakdown.get("cache_creation_cost") == pytest.approx(
        cache_write * CACHE_WRITE_RATE, rel=1e-6
    )
    # Grand total is unaffected by itemization.
    assert total == pytest.approx(breakdown["total_cost"], rel=1e-6)


def test_cache_cost_itemization_semantics_match_anthropic_convention():
    """input_cost stays the full folded prompt-side spend; cache fields are additive annotations.

    This matches the Anthropic-path convention so downstream consumers see consistent
    semantics regardless of provider.
    """
    cached = 16000
    cache_write = 26022
    fresh = 100
    out = 50
    usage = _openai_style_usage(
        cached=cached, cache_write=cache_write, fresh=fresh, out=out
    )
    logging_obj = _new_logging_obj()

    completion_cost(
        completion_response=_response_with_usage(usage),
        model=MODEL,
        custom_llm_provider="openai",
        litellm_logging_obj=logging_obj,
    )

    b = logging_obj.cost_breakdown
    assert b is not None

    # input_cost must be the full prompt-side cost (cache dollars folded in).
    expected_input = (
        fresh * INPUT_RATE + cached * CACHE_READ_RATE + cache_write * CACHE_WRITE_RATE
    )
    assert b.get("input_cost") == pytest.approx(expected_input, rel=1e-6)

    # Cache annotations overlap input_cost (they are NOT disjoint from it).
    assert b.get("cache_read_cost") == pytest.approx(cached * CACHE_READ_RATE, rel=1e-6)
    assert b.get("cache_creation_cost") == pytest.approx(
        cache_write * CACHE_WRITE_RATE, rel=1e-6
    )

    # total_cost == input_cost + output_cost (cache fields are annotations only).
    assert (b["input_cost"] + b.get("output_cost", 0.0)) == pytest.approx(
        b["total_cost"], rel=1e-6
    )


def test_anthropic_style_cache_itemization_still_works():
    """Regression: Anthropic-style top-level cache keys continue to populate the breakdown."""
    cache_read_tokens = 2000
    cache_creation_tokens = 3000
    fresh = 100
    out = 50
    usage = Usage(
        prompt_tokens=cache_read_tokens + cache_creation_tokens + fresh,
        completion_tokens=out,
        total_tokens=cache_read_tokens + cache_creation_tokens + fresh + out,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            cached_tokens=cache_read_tokens,
        ),
    )
    # Anthropic reports these as top-level usage attributes.
    setattr(usage, "cache_read_input_tokens", cache_read_tokens)
    setattr(usage, "cache_creation_input_tokens", cache_creation_tokens)
    logging_obj = _new_logging_obj()

    completion_cost(
        completion_response=_response_with_usage(usage),
        model=MODEL,
        custom_llm_provider="openai",
        litellm_logging_obj=logging_obj,
    )

    b = logging_obj.cost_breakdown
    assert b is not None
    assert b.get("cache_read_cost") == pytest.approx(
        cache_read_tokens * CACHE_READ_RATE, rel=1e-6
    )
    assert b.get("cache_creation_cost") == pytest.approx(
        cache_creation_tokens * CACHE_WRITE_RATE, rel=1e-6
    )


def test_no_cache_tokens_leaves_cache_fields_absent():
    """When there are no cache tokens, cache_read_cost and cache_creation_cost are not set."""
    usage = Usage(
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
    )
    logging_obj = _new_logging_obj()

    completion_cost(
        completion_response=_response_with_usage(usage),
        model=MODEL,
        custom_llm_provider="openai",
        litellm_logging_obj=logging_obj,
    )

    b = logging_obj.cost_breakdown
    assert b is not None
    assert "cache_read_cost" not in b
    assert "cache_creation_cost" not in b
