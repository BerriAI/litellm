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
        model="openrouter/anthropic/claude-sonnet-4",
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
        resp.choices[0].message.images[0]["image_url"]["url"].startswith("data:image/")
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


def test_openrouter_qwen_cache_control_supported():
    """
    Validates that Qwen models routed through OpenRouter cleanly support 
    cache_control and cache_control_injection_points parameters,
    resolving issue #29322.
    """
    from litellm.llms.openrouter.chat.transformation import OpenrouterConfig

    config = OpenrouterConfig()
    qwen_model = "openrouter/qwen/qwen3.6-flash"
    
    # Retrieve the dynamically mapped parameters for the Qwen endpoint
    supported_params = config.get_supported_openai_params(model=qwen_model)
    
    # Assertions to ensure the translation layer captures the caching parameters
    assert "cache_control" in supported_params, f"cache_control missing from supported parameters for {qwen_model}"
    assert "cache_control_injection_points" in supported_params, f"cache_control_injection_points missing from supported parameters for {qwen_model}"
    print("✅ test_openrouter_qwen_cache_control_supported passed!")


