"""
Test for custom_tokenizer bug fix.
Issue: custom_tokenizer from model_info was not being extracted from deployment,
causing token_counter to always use OpenAI tokenizer instead of the configured custom tokenizer.
"""

import pytest
import litellm
import litellm.proxy.proxy_server
from litellm.proxy.proxy_server import token_counter
from litellm.proxy._types import TokenCountRequest
from litellm import Router


@pytest.mark.asyncio
async def test_custom_tokenizer_from_model_info():
    """
    Test that custom_tokenizer from model_info is correctly used for token counting.

    Real-world scenario: Using intfloat/multilingual-e5-large-instruct tokenizer
    for a custom embedding model (like Groq-hosted llama model used for embeddings).

    This test reproduces the bug where:
    - model_info was declared but never populated from deployment
    - custom_tokenizer was therefore never extracted
    - token_counter always fell back to OpenAI tokenizer

    Expected behavior:
    - When a model has custom_tokenizer in model_info
    - The token_counter should use that custom tokenizer (intfloat/multilingual-e5-large-instruct)
    - tokenizer_type should reflect "huggingface_tokenizer" not "openai_tokenizer"
    """

    # Create a router with a model that has custom_tokenizer for multilingual embeddings
    # This matches the user's real config with intfloat/multilingual-e5-large-instruct
    llm_router = Router(
        model_list=[
            {
                "model_name": "nikro-llama",
                "litellm_params": {
                    "model": "openai/llama-3.1-8b-instant",
                    "api_base": "https://api.groq.com/openai/v1",
                },
                "model_info": {
                    "mode": "embedding",
                    "custom_tokenizer": {
                        "identifier": "intfloat/multilingual-e5-large-instruct",
                        "revision": "main",
                        "auth_token": None,
                    },
                },
            }
        ]
    )

    setattr(litellm.proxy.proxy_server, "llm_router", llm_router)

    # Make a token counting request with a multilingual text sample
    # This is realistic for the multilingual-e5 model
    response = await token_counter(
        request=TokenCountRequest(
            model="nikro-llama",
            messages=[
                {"role": "user", "content": "Hello world! Bonjour le monde! 你好世界!"}
            ],
        )
    )

    print("Response:", response)
    print("Tokenizer type:", response.tokenizer_type)
    print("Model used:", response.model_used)
    print("Total tokens:", response.total_tokens)

    # Verify that custom tokenizer (intfloat/multilingual-e5-large-instruct) was used
    assert response.tokenizer_type == "huggingface_tokenizer", (
        f"Expected 'huggingface_tokenizer' (intfloat/multilingual-e5-large-instruct) "
        f"but got '{response.tokenizer_type}'. "
        "This indicates the custom_tokenizer from model_info was not used."
    )
    assert response.request_model == "nikro-llama"
    assert response.model_used == "llama-3.1-8b-instant"
    assert response.total_tokens > 0


@pytest.mark.asyncio
async def test_custom_tokenizer_with_llamacpp():
    """
    Test custom_tokenizer with llamacpp model (similar to user's setup).

    This simulates the user's Docker environment where:
    - They have a llamacpp model
    - With custom_tokenizer configured
    - In Docker, it was using OpenAI tokenizer (bug)
    - Locally, it was using HuggingFace tokenizer (correct)
    """

    llm_router = Router(
        model_list=[
            {
                "model_name": "my-local-model",
                "litellm_params": {
                    "model": "openai/my-local-llama",
                    "api_base": "http://localhost:8080/v1",
                },
                "model_info": {
                    "custom_tokenizer": {
                        "identifier": "intfloat/multilingual-e5-large-instruct",
                        "revision": "main",
                        "auth_token": None,
                    },
                },
            }
        ]
    )

    setattr(litellm.proxy.proxy_server, "llm_router", llm_router)

    response = await token_counter(
        request=TokenCountRequest(
            model="my-local-model",
            messages=[{"role": "user", "content": "test message"}],
        )
    )

    # The bug would cause this to be "openai_tokenizer"
    assert (
        response.tokenizer_type == "huggingface_tokenizer"
    ), f"Custom tokenizer not used! Got: {response.tokenizer_type}"


@pytest.mark.asyncio
async def test_multilingual_e5_embedding_model():
    """
    Test the exact real-world use case: intfloat/multilingual-e5-large-instruct
    tokenizer with a custom embedding endpoint.

    This is the user's actual production scenario:
    - Custom embedding model endpoint (could be llama.cpp, vLLM, etc.)
    - Using intfloat/multilingual-e5-large-instruct for tokenization
    - Model served via OpenAI-compatible API
    """

    llm_router = Router(
        model_list=[
            {
                "model_name": "my-embedding-model",
                "litellm_params": {
                    "model": "openai/custom-embedding-model",
                    "api_base": "http://localhost:8080/v1",
                },
                "model_info": {
                    "mode": "embedding",
                    "custom_tokenizer": {
                        "identifier": "intfloat/multilingual-e5-large-instruct",
                        "revision": "main",
                        "auth_token": None,
                    },
                },
            }
        ]
    )

    setattr(litellm.proxy.proxy_server, "llm_router", llm_router)

    # Test with multilingual content (what e5-large-instruct is designed for)
    response = await token_counter(
        request=TokenCountRequest(
            model="my-embedding-model",
            messages=[
                {
                    "role": "user",
                    "content": "This is a multilingual test. C'est un test multilingue. 这是一个多语言测试。",
                }
            ],
        )
    )

    print(
        f"Embedding model test - Tokenizer: {response.tokenizer_type}, Tokens: {response.total_tokens}"
    )

    # Must use HuggingFace tokenizer with intfloat/multilingual-e5-large-instruct
    assert response.tokenizer_type == "huggingface_tokenizer", (
        f"The intfloat/multilingual-e5-large-instruct tokenizer was not used! "
        f"Got: {response.tokenizer_type}"
    )
    assert response.total_tokens > 0


@pytest.mark.asyncio
async def test_model_without_custom_tokenizer_uses_default():
    """
    Test that models without custom_tokenizer still work correctly.
    """

    llm_router = Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4",
                },
                "model_info": {},  # No custom_tokenizer
            }
        ]
    )

    setattr(litellm.proxy.proxy_server, "llm_router", llm_router)

    response = await token_counter(
        request=TokenCountRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "hello"}],
        )
    )

    # Should use OpenAI tokenizer for GPT-4
    assert response.tokenizer_type == "openai_tokenizer"
    assert response.model_used == "gpt-4"
