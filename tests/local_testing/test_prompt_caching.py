"""Asserts that prompt caching information is correctly returned for Anthropic, OpenAI, and Deepseek"""

import io
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

import litellm
import pytest


@pytest.mark.parametrize(
    "model",
    [
        "anthropic/claude-3-5-sonnet-20240620",
        "openai/gpt-4o",
        "deepseek/deepseek-chat",
    ],
)
def test_prompt_caching_model(model):
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

    print("response=", response)
    print("response.usage=", response.usage)

    assert "prompt_tokens_details" in response.usage
    assert response.usage.prompt_tokens_details.cached_tokens > 0

    # assert "cache_read_input_tokens" in response.usage
    # assert "cache_creation_input_tokens" in response.usage

    # # Assert either a cache entry was created or cache was read - changes depending on the anthropic api ttl
    # assert (response.usage.cache_read_input_tokens > 0) or (
    #     response.usage.cache_creation_input_tokens > 0
    # )
