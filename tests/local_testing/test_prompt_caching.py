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


@pytest.mark.parametrize(
    "model",
    [
        "anthropic/claude-3-5-sonnet-20240620",
        # "openai/gpt-4o",
        # "deepseek/deepseek-chat",
    ],
)
def test_prompt_caching_model(model):
    try:
        for _ in range(2):
            response = litellm.completion(
                model=model,
                messages=[
                    # System Message
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": "Here is the full text of a complex legal agreement"
                                * 400,
                                "cache_control": {"type": "ephemeral"},
                            }
                        ],
                    },
                    # marked for caching with the cache_control parameter, so that this checkpoint can read from the previous cache.
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "What are the key terms and conditions in this agreement?",
                                "cache_control": {"type": "ephemeral"},
                            }
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": "Certainly! the key terms and conditions are the following: the contract is 1 year long for $10/mo",
                    },
                    # The final turn is marked with cache-control, for continuing in followups.
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "What are the key terms and conditions in this agreement?",
                                "cache_control": {"type": "ephemeral"},
                            }
                        ],
                    },
                ],
                temperature=0.2,
                max_tokens=10,
            )

            _usage_format_tests(response.usage)

        print("response=", response)
        print("response.usage=", response.usage)

        _usage_format_tests(response.usage)

        assert "prompt_tokens_details" in response.usage
        assert response.usage.prompt_tokens_details.cached_tokens > 0
    except litellm.InternalServerError:
        pass


def test_supports_prompt_caching():
    from litellm.utils import supports_prompt_caching

    supports_pc = supports_prompt_caching(model="anthropic/claude-3-5-sonnet-20240620")

    assert supports_pc
