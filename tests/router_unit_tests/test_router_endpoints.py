import os

from litellm import Router

import pytest


@pytest.fixture
def model_list():
    return [
        {
            "model_name": "gpt-5-mini",
            "litellm_params": {
                "model": "gpt-5-mini",
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
        },
        {
            "model_name": "gpt-5.5",
            "litellm_params": {
                "model": "gpt-5.5",
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
        },
        {
            "model_name": "gpt-image-1",
            "litellm_params": {
                "model": "gpt-image-1",
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
        },
        {
            "model_name": "cohere-rerank",
            "litellm_params": {
                "model": "cohere/rerank-english-v3.0",
                "api_key": os.getenv("COHERE_API_KEY"),
            },
        },
        {
            "model_name": "claude-sonnet-4-5-20250929",
            "litellm_params": {
                "model": "gpt-5-mini",
                "mock_response": "hi this is macintosh.",
            },
        },
    ]
