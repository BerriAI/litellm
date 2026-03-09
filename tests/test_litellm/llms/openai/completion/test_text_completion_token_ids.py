"""
Unit tests for text_completion with token IDs (list of integers) as prompt.
Tests the fix for https://github.com/BerriAI/litellm/issues/17118
"""

import os
import sys

import pytest
import respx
from httpx import Response

sys.path.insert(0, os.path.abspath("../../../../.."))

import litellm
from litellm import text_completion


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    """Set up test environment variables."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-key")


@pytest.fixture
def text_completion_response():
    """Mock response for text completion API."""
    return {
        "id": "cmpl-test123",
        "object": "text_completion",
        "created": 1677652288,
        "model": "gpt-3.5-turbo-instruct",
        "choices": [
            {
                "text": " is a greeting",
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 2, "completion_tokens": 4, "total_tokens": 6},
    }


class TestTextCompletionTokenIds:
    """Test text_completion with token IDs as prompt."""

    @respx.mock
    def test_completion_prompt_token_ids(
        self, text_completion_response, monkeypatch
    ):
        """
        Test text_completion with a list of token IDs (integers).
        This tests the fix for https://github.com/BerriAI/litellm/issues/17118
        """
        # Token IDs for "Hello world" in GPT tokenizer
        token_ids = [15496, 995]

        # Mock the OpenAI completions endpoint
        respx.post("https://api.openai.com/v1/completions").mock(
            return_value=Response(200, json=text_completion_response)
        )

        response = text_completion(
            model="gpt-3.5-turbo-instruct",
            prompt=token_ids,
            max_tokens=5,
        )

        assert response is not None
        assert response.choices[0].text is not None
        assert response.usage.prompt_tokens == 2

    @respx.mock
    def test_completion_prompt_token_ids_batch(
        self, text_completion_response, monkeypatch
    ):
        """
        Test text_completion with multiple prompts as token IDs.
        """
        # Multiple token ID lists (batch)
        token_ids_batch = [[15496, 995], [9906, 0]]

        # Update mock response for batch
        batch_response = {
            **text_completion_response,
            "choices": [
                {
                    "text": " is a greeting",
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop",
                },
                {
                    "text": " is another",
                    "index": 1,
                    "logprobs": None,
                    "finish_reason": "stop",
                },
            ],
            "usage": {"prompt_tokens": 4, "completion_tokens": 6, "total_tokens": 10},
        }

        respx.post("https://api.openai.com/v1/completions").mock(
            return_value=Response(200, json=batch_response)
        )

        response = text_completion(
            model="gpt-3.5-turbo-instruct",
            prompt=token_ids_batch,
            max_tokens=5,
        )

        assert response is not None
        assert len(response.choices) == 2
