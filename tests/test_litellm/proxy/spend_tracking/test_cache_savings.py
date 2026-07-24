import os
import sys

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.proxy.spend_tracking.cache_savings import (
    extract_cache_creation_tokens,
    extract_cache_read_tokens,
)


def test_extract_cache_read_tokens_anthropic_top_level():
    usage_obj = {
        "prompt_tokens": 100,
        "cache_read_input_tokens": 80,
        "prompt_tokens_details": {"cached_tokens": 80},
    }
    # Anthropic top-level value should win over prompt_tokens_details fallback.
    assert extract_cache_read_tokens(usage_obj) == 80


def test_extract_cache_read_tokens_openai_compatible_fallback():
    # Anthropic field absent — fall back to prompt_tokens_details.cached_tokens.
    usage_obj = {
        "prompt_tokens": 22583,
        "prompt_tokens_details": {"cached_tokens": 22016},
    }
    assert extract_cache_read_tokens(usage_obj) == 22016


def test_extract_cache_read_tokens_zero_explicit_falls_through():
    usage_obj = {
        "cache_read_input_tokens": 0,
        "prompt_tokens_details": {"cached_tokens": 640},
    }
    assert extract_cache_read_tokens(usage_obj) == 640


def test_extract_cache_read_tokens_zero_when_missing():
    assert extract_cache_read_tokens({}) == 0
    assert extract_cache_read_tokens({"cache_read_input_tokens": None}) == 0
    assert extract_cache_read_tokens({"prompt_tokens_details": {"cached_tokens": None}}) == 0


def test_extract_cache_read_tokens_ignores_non_numbers():
    # Matches CACHE_READ_INPUT_TOKENS_SQL, which reads anything that is not a
    # JSON number as absent. Booleans are not token counts either.
    assert extract_cache_read_tokens({"cache_read_input_tokens": "80"}) == 0
    assert extract_cache_read_tokens({"cache_read_input_tokens": True, "prompt_tokens_details": {"cached_tokens": 12}}) == 12
    assert extract_cache_read_tokens({"prompt_tokens_details": "REDACTED"}) == 0


def test_extract_cache_read_tokens_truncates_floats():
    assert extract_cache_read_tokens({"cache_read_input_tokens": 7.9}) == 7


def test_extract_cache_creation_tokens_anthropic_top_level():
    usage_obj = {
        "prompt_tokens": 100,
        "cache_creation_input_tokens": 50,
        "prompt_tokens_details": {"cache_write_tokens": 50},
    }
    # Anthropic top-level should short-circuit the fallback.
    assert extract_cache_creation_tokens(usage_obj) == 50


def test_extract_cache_creation_tokens_openai_cache_write_alias():
    # kimi-k2 emits cache_write_tokens.
    usage_obj = {
        "prompt_tokens": 1000,
        "prompt_tokens_details": {"cache_write_tokens": 200},
    }
    assert extract_cache_creation_tokens(usage_obj) == 200


def test_extract_cache_creation_tokens_openai_cache_creation_alias():
    # Other OpenAI-compatible providers emit cache_creation_tokens.
    usage_obj = {
        "prompt_tokens": 1000,
        "prompt_tokens_details": {"cache_creation_tokens": 300},
    }
    assert extract_cache_creation_tokens(usage_obj) == 300


def test_extract_cache_creation_tokens_zero_when_missing():
    assert extract_cache_creation_tokens({}) == 0
    assert extract_cache_creation_tokens({"cache_creation_input_tokens": None}) == 0
    assert extract_cache_creation_tokens({"prompt_tokens_details": {"cache_write_tokens": None}}) == 0
