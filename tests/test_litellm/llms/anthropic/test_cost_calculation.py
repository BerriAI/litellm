"""
A server-side fallback response must be priced one attempt at a time.

With Anthropic's server-side fallback, `usage.iterations` carries one entry per attempt, each with
its own `model`; the top-level `usage` and `model` describe only the attempt that produced the
returned message. `calculate_usage` sums every attempt's tokens (correct), but pricing that sum at
the top-level model charges the declined attempt's tokens at the fallback model's rates, and
charges for attempts Anthropic does not bill at all.

Anthropic's rule, quoted from the public docs on 2026-09-02:
  refusals-and-fallback: "A mid-stream refusal bills the input tokens and the output already
    streamed at normal rates."
  Fable 5 fallback billing cookbook, "Billing changes": "Input tokens are not billed on a direct
    classifier block (i.e. when a request is blocked before any output tokens were returned)."
    "Use usage.iterations if you need exact per-model attribution."

The first usage object below is copied verbatim from a real Claude response (content stripped).
"""

import pytest

from litellm.llms.anthropic.chat.transformation import AnthropicConfig
from litellm.llms.anthropic.cost_calculation import cost_per_token

RESPONSE_MODEL = "claude-opus-4-8"  # top-level model: the attempt that produced the message

# Fable 5 was declined mid-stream after 299 output tokens; Opus 4.8 finished the turn.
MID_STREAM_DECLINE_THEN_FALLBACK = {
    "input_tokens": 2,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 491047,
    "output_tokens": 476,
    "service_tier": "standard",
    "cache_creation": {"ephemeral_1h_input_tokens": 0, "ephemeral_5m_input_tokens": 0},
    "inference_geo": "not_available",
    "iterations": [
        {
            "type": "message",
            "model": "claude-fable-5",
            "input_tokens": 2,
            "cache_creation_input_tokens": 2083,
            "cache_read_input_tokens": 560186,
            "output_tokens": 299,
            "cache_creation": {"ephemeral_5m_input_tokens": 2083, "ephemeral_1h_input_tokens": 0},
        },
        {
            "type": "fallback_message",
            "model": "claude-opus-4-8",
            "input_tokens": 2,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 491047,
            "output_tokens": 476,
            "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
        },
    ],
}

# Same shape, but the first attempt was blocked before any output: reported, not billed.
BLOCKED_BEFORE_OUTPUT_THEN_FALLBACK = {
    "input_tokens": 2,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 491047,
    "output_tokens": 476,
    "iterations": [
        {
            "type": "message",
            "model": "claude-fable-5",
            "input_tokens": 2,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 560186,
            "output_tokens": 0,
        },
        {
            "type": "fallback_message",
            "model": "claude-opus-4-8",
            "input_tokens": 2,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 491047,
            "output_tokens": 476,
        },
    ],
}


def _price_attempt(it: dict) -> float:
    usage = AnthropicConfig().calculate_usage(usage_object=it, reasoning_content=None)
    p, c = cost_per_token(model=it["model"], usage=usage)
    return p + c


def _price_response(usage_object: dict) -> float:
    usage = AnthropicConfig().calculate_usage(usage_object=usage_object, reasoning_content=None)
    p, c = cost_per_token(model=RESPONSE_MODEL, usage=usage)
    return p + c


def test_token_totals_still_sum_every_attempt():
    usage = AnthropicConfig().calculate_usage(usage_object=MID_STREAM_DECLINE_THEN_FALLBACK, reasoning_content=None)
    assert usage.cache_read_input_tokens == 560186 + 491047
    assert usage.completion_tokens == 299 + 476
    assert usage.prompt_tokens_details.cache_creation_tokens == 2083


def test_mid_stream_decline_bills_each_attempt_at_its_own_model():
    its = MID_STREAM_DECLINE_THEN_FALLBACK["iterations"]
    expected = _price_attempt(its[0]) + _price_attempt(its[1])
    assert _price_response(MID_STREAM_DECLINE_THEN_FALLBACK) == pytest.approx(expected, rel=1e-9)

    # and the declined Fable attempt is the larger part of the bill, as it should be at Fable rates
    assert _price_attempt(its[0]) > _price_attempt(its[1])


def test_attempt_blocked_before_output_is_not_billed():
    served = BLOCKED_BEFORE_OUTPUT_THEN_FALLBACK["iterations"][1]
    assert _price_response(BLOCKED_BEFORE_OUTPUT_THEN_FALLBACK) == pytest.approx(_price_attempt(served), rel=1e-9)


def test_single_attempt_responses_are_unchanged():
    plain = {
        "input_tokens": 100,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 1000,
        "output_tokens": 50,
        "iterations": [
            {
                "type": "message",
                "model": RESPONSE_MODEL,
                "input_tokens": 100,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 1000,
                "output_tokens": 50,
            }
        ],
    }
    without = dict(plain)
    del without["iterations"]
    assert _price_response(plain) == pytest.approx(_price_response(without), rel=1e-12)


def test_fast_mode_multiplier_still_applies_across_attempts():
    # speed is a request-level field; both attempts on Opus so the 2x fast multiplier is well-defined
    fast = {
        "input_tokens": 10,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "output_tokens": 100,
        "speed": "fast",
        "iterations": [
            {
                "type": "message",
                "model": "claude-opus-5",
                "input_tokens": 10,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "output_tokens": 40,
            },
            {
                "type": "fallback_message",
                "model": "claude-opus-5",
                "input_tokens": 10,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "output_tokens": 60,
            },
        ],
    }
    standard = {**fast, "speed": "standard"}
    fast_usage = AnthropicConfig().calculate_usage(usage_object=fast, reasoning_content=None)
    std_usage = AnthropicConfig().calculate_usage(usage_object=standard, reasoning_content=None)
    fp, fc = cost_per_token(model="claude-opus-5", usage=fast_usage)
    sp, sc = cost_per_token(model="claude-opus-5", usage=std_usage)
    assert fp + fc == pytest.approx(2 * (sp + sc), rel=1e-9)
