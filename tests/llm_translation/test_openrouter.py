import os
import sys
import pytest
import respx
from httpx import Response

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system paths
import litellm


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    """Set a fake OpenRouter API key so headers are populated without secrets."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-fake-key")


@respx.mock
def test_completion_openrouter_reasoning_content():
    # Mock OpenRouter chat completions endpoint
    respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=Response(
            200,
            json={
                "id": "gen-abc123",
                "object": "chat.completion",
                "created": 1730000000,
                "model": "openrouter/anthropic/claude-3.7-sonnet",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "Hello world",
                            # Providers may send 'reasoning' which we map to reasoning_content
                            "reasoning": "Thinking step by step...",
                        },
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12, "cost": 0.00012},
            },
        )
    )

    litellm._turn_on_debug()
    resp = litellm.completion(
        model="openrouter/anthropic/claude-3.7-sonnet",
        messages=[{"role": "user", "content": "Hello world"}],
        reasoning={"effort": "high"},
    )
    assert resp.choices[0].message.reasoning_content is not None


@respx.mock
def test_completion_openrouter_image_generation():
    # Mock OpenRouter chat completions endpoint for image generation
    # Return an image as either a data URL or a remote URL deterministically here
    respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=Response(
            200,
            json={
                "id": "gen-img-123",
                "object": "chat.completion",
                "created": 1730000001,
                "model": "openrouter/google/gemini-2.5-flash-image",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "Here is your image.",
                            "images": [
                                {
                                    "index": 0,
                                    "type": "image_url",
                                    "image_url": {
                                        # Use a data URL since some providers may return this
                                        "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB"
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 6,
                    "completion_tokens": 1299,
                    "total_tokens": 1305,
                    "completion_tokens_details": {"image_tokens": 1290},
                    "cost": 0.0387243,
                },
            },
        )
    )

    litellm._turn_on_debug()
    resp = litellm.completion(
        model="openrouter/google/gemini-2.5-flash-image",
        messages=[{"role": "user", "content": "Generate an image of a cat"}],
        modalities=["image", "text"],
    )

    url = (
        resp.choices[0]
        .message.images[0]["image_url"]["url"]
    )
    # Accept either data URLs or remote URLs
    assert isinstance(url, str) and len(url) > 0
    assert url.startswith("data:image/") or url.startswith("http")


@respx.mock
def test_openrouter_embedding():
    """Test OpenRouter embeddings support with mocked network."""
    # Mock the OpenRouter embeddings endpoint
    respx.post("https://openrouter.ai/api/v1/embeddings").mock(
        return_value=Response(
            200,
            json={
                "object": "list",
                "data": [
                    {
                        "object": "embedding",
                        "index": 0,
                        "embedding": [0.01, 0.02, 0.03],
                    },
                    {
                        "object": "embedding",
                        "index": 1,
                        "embedding": [0.04, 0.05, 0.06],
                    },
                ],
                "model": "openrouter/openai/text-embedding-3-small",
                "usage": {"prompt_tokens": 6, "total_tokens": 6},
            },
        )
    )

    litellm._turn_on_debug()
    resp = litellm.embedding(
        model="openrouter/openai/text-embedding-3-small",
        input=["Hello world", "How are you?"],
    )
    assert resp is not None
    assert len(resp.data) == 2
    assert resp.data[0]["embedding"] is not None
    assert isinstance(resp.data[0]["embedding"], list)
    assert len(resp.data[0]["embedding"]) > 0
