# -*- coding: utf-8 -*-
"""
Regression tests for decisions-route responses flowing into the legacy
Langfuse logger (issue #45584).

DecisionsUsage / OpenAIDecisionUsage are Pydantic models without a dict-style
``.get`` and name their token counts input_tokens/output_tokens (and
cached_tokens/cache_write_tokens). The Langfuse logger used to crash on those
(AttributeError: 'DecisionsUsage' object has no attribute 'get') and drop the
whole trace; token counts were also misread as zero.
"""

from litellm.integrations.langfuse.langfuse import (
    _extract_cache_read_input_tokens,
    _usage_token_value,
)
from litellm.types.decisions import DecisionsUsage


class _PlainUsage:
    prompt_tokens = 5
    completion_tokens = 7
    total_tokens = 12
    cache_creation_input_tokens = 2
    cache_read_input_tokens = 3


class _EmptyUsage:
    pass


def test_usage_token_value_reads_standard_attributes() -> None:
    usage = _PlainUsage()
    assert _usage_token_value(usage, "prompt_tokens") == 5
    assert _usage_token_value(usage, "completion_tokens") == 7
    assert _usage_token_value(usage, "total_tokens") == 12
    assert _usage_token_value(usage, "cache_creation_input_tokens") == 2


def test_usage_token_value_reads_dict_payloads() -> None:
    usage = {"prompt_tokens": 5, "completion_tokens": 7, "cache_creation_input_tokens": 2}
    assert _usage_token_value(usage, "prompt_tokens") == 5
    assert _usage_token_value(usage, "completion_tokens") == 7
    assert _usage_token_value(usage, "cache_creation_input_tokens") == 2


def test_usage_token_value_defaults_missing_counts_to_zero() -> None:
    assert _usage_token_value(_EmptyUsage(), "prompt_tokens") == 0
    assert _usage_token_value({}, "prompt_tokens") == 0


def test_usage_token_value_falls_back_to_decisions_field_names() -> None:
    usage = DecisionsUsage(input_tokens=5, output_tokens=7, cached_tokens=3, cache_write_tokens=2)
    assert _usage_token_value(usage, "prompt_tokens", "input_tokens") == 5
    assert _usage_token_value(usage, "completion_tokens", "output_tokens") == 7
    # DecisionsUsage has no total_tokens; must default to 0 rather than crash.
    assert _usage_token_value(usage, "total_tokens") == 0
    assert _usage_token_value(usage, "cache_creation_input_tokens", "cache_write_tokens") == 2


def test_extract_cache_read_accepts_pydantic_decisions_usage() -> None:
    usage = DecisionsUsage(input_tokens=5, output_tokens=7, cached_tokens=3, cache_write_tokens=2)
    assert _extract_cache_read_input_tokens(usage) == 3


def test_extract_cache_read_keeps_dict_semantics() -> None:
    usage = {"cache_read_input_tokens": 4, "prompt_tokens": 10}
    assert _extract_cache_read_input_tokens(usage) == 4
    assert _extract_cache_read_input_tokens({}) == 0


def test_extract_cache_read_keeps_plain_attribute_semantics() -> None:
    usage = _PlainUsage()
    assert _extract_cache_read_input_tokens(usage) == 3
