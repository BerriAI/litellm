import os
import sys
from unittest.mock import patch

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path


import pytest

import litellm
from litellm import ModelResponse
from base_llm_unit_tests import BaseLLMChatTest

class TestOpenAIO1(BaseLLMChatTest):
    def get_base_completion_call_args(self):
        return {
            "model": "o1",
        }

    def get_client(self):
        from openai import OpenAI

        return OpenAI(api_key="fake-api-key")

    def test_prompt_caching(self):
        """Temporary override. o1 prompt caching is not working."""
        pass


class TestOpenAIO3(BaseLLMChatTest):
    def get_base_completion_call_args(self):
        return {
            "model": "o3-mini",
        }

    def get_client(self):
        from openai import OpenAI

        return OpenAI(api_key="fake-api-key")

    def test_prompt_caching(self):
        """Override, as o3 prompt caching is flaky"""
        pass


def test_o3_reasoning_effort():
    resp = litellm.completion(
        model="o3-mini",
        messages=[{"role": "user", "content": "Hello!"}],
        reasoning_effort="high",
    )
    assert resp.choices[0].message.content is not None


@pytest.mark.parametrize("model", ["o1", "o3-mini"])
def test_streaming_response(model):
    """Test that streaming response is returned correctly"""
    from litellm import completion

    response = completion(
        model=model,
        messages=[
            {"role": "system", "content": "Be a good bot!"},
            {"role": "user", "content": "Hello!"},
        ],
        stream=True,
    )

    assert response is not None

    chunks = []
    for chunk in response:
        chunks.append(chunk)

    resp = litellm.stream_chunk_builder(chunks=chunks)
    print(resp)
