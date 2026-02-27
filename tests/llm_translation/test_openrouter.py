import os
import sys
import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system paths
import litellm


def test_completion_openrouter_reasoning_content():
    litellm._turn_on_debug()
    resp = litellm.completion(
        model="openrouter/anthropic/claude-3.7-sonnet",
        messages=[{"role": "user", "content": "Hello world"}],
        reasoning={"effort": "high"},
    )
    print(resp)
    assert resp.choices[0].message.reasoning_content is not None


def test_completion_openrouter_image_generation():
    litellm._turn_on_debug()
    resp = litellm.completion(
        model="openrouter/google/gemini-2.5-flash-image",
        messages=[{"role": "user", "content": "Generate an image of a cat"}],
        modalities=["image", "text"],
    )
    print(resp)
    assert (
        resp.choices[0]
        .message.images[0]["image_url"]["url"]
        .startswith("data:image/png;base64,")
    )


def test_openrouter_embedding():
    """Test OpenRouter embeddings support."""
    litellm._turn_on_debug()
    resp = litellm.embedding(
        model="openrouter/openai/text-embedding-3-small",
        input=["Hello world", "How are you?"],
    )
    print(resp)
    assert resp is not None
    assert len(resp.data) == 2
    assert resp.data[0]["embedding"] is not None
    assert isinstance(resp.data[0]["embedding"], list)
    assert len(resp.data[0]["embedding"]) > 0
