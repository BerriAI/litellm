"""Asserts that prompt caching information is correctly returned for Anthropic, OpenAI, and Deepseek"""

import io
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

import litellm
import pytest


def _usage_format_tests(usage: litellm.Usage):
    """
    OpenAI prompt caching
    - prompt_tokens = sum of non-cache hit tokens + cache-hit tokens
    - total_tokens = prompt_tokens + completion_tokens

    Example
    ```
    "usage": {
        "prompt_tokens": 2006,
        "completion_tokens": 300,
        "total_tokens": 2306,
        "prompt_tokens_details": {
            "cached_tokens": 1920
        },
        "completion_tokens_details": {
            "reasoning_tokens": 0
        }
        # ANTHROPIC_ONLY #
        "cache_creation_input_tokens": 0
    }
    ```
    """
    assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens

    assert usage.prompt_tokens > usage.prompt_tokens_details.cached_tokens


def test_supports_prompt_caching():
    from litellm.utils import supports_prompt_caching

    supports_pc = supports_prompt_caching(model="anthropic/claude-sonnet-4-5-20250929")

    assert supports_pc
