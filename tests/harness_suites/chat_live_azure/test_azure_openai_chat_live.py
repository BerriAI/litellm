import sys
import os

sys.path.insert(
    0, os.path.abspath("../../")
)  # Adds the parent directory to the system path

import pytest
from litellm.llms.azure.common_utils import process_azure_headers
from httpx import Headers
from base_embedding_unit_tests import BaseLLMEmbeddingTest
import litellm
from litellm import completion

def test_azure_safety_result():
    """Bubble up safety result from Azure OpenAI"""
    from litellm import completion

    litellm._turn_on_debug()

    response = completion(
        model="azure/gpt-4.1-mini",
        api_key=os.getenv("AZURE_AI_API_KEY"),
        api_base=os.getenv("AZURE_AI_API_BASE"),
        api_version="2024-12-01-preview",
        messages=[{"role": "user", "content": "Hello world"}],
    )
    print(f"response: {response}")
    assert response.choices[0].message.content is not None
    assert response.choices[0].provider_specific_fields is not None


def test_completion_azure_deployment_id():
    """
    Ensure deployment_id takes precedence over model.
    """
    litellm.set_verbose = True
    response = completion(
        deployment_id="gpt-4.1-mini",
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": "Hello, how are you?",
            }
        ],
    )
    # Add any assertions here to check the response
    print(response)


def test_azure_openai_with_prompt_cache_key():
    """
    E2E test for Azure OpenAI with prompt cache key param on /chat/completions API.
    """
    litellm._turn_on_debug()
    response = litellm.completion(
        model="azure/gpt-4.1-mini",
        api_key=os.getenv("AZURE_AI_API_KEY"),
        api_base=os.getenv("AZURE_AI_API_BASE"),
        api_version="2024-12-01-preview",
        messages=[{"role": "user", "content": "What is the weather in San Francisco?"}],
        prompt_cache_key="test_streaming_azure_openai",
    )
