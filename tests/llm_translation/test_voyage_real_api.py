"""
Real API test for VoyageAI models through LiteLLM

These tests require a valid VOYAGE_API_KEY environment variable.
They are skipped automatically if the API key is not set.
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.abspath("../.."))

import litellm

# Set verbose for debugging
litellm.set_verbose = False

# Skip all tests in this module if VOYAGE_API_KEY is not set
pytestmark = pytest.mark.skipif(
    not os.environ.get("VOYAGE_API_KEY"),
    reason="VOYAGE_API_KEY environment variable not set"
)


def test_voyage_3_5_through_litellm():
    """Test voyage-3.5 model through LiteLLM"""
    print("\n=== Testing voyage-3.5 through LiteLLM ===")

    response = litellm.embedding(
        model="voyage/voyage-3.5",
        input=["Testing voyage-3.5 through LiteLLM integration"],
        api_key=os.environ.get("VOYAGE_API_KEY")
    )

    # Verify response
    assert response is not None
    assert hasattr(response, 'model')
    assert hasattr(response, 'data')
    assert len(response.data) > 0
    assert hasattr(response, 'usage')

    print(f"✓ Model: {response.model}")
    print(f"✓ Embedding dimensions: {len(response.data[0]['embedding'])}")
    print(f"✓ Usage: {response.usage}")


def test_voyage_3_5_lite_through_litellm():
    """Test voyage-3.5-lite model through LiteLLM"""
    print("\n=== Testing voyage-3.5-lite through LiteLLM ===")

    response = litellm.embedding(
        model="voyage/voyage-3.5-lite",
        input=["Testing voyage-3.5-lite through LiteLLM integration"],
        api_key=os.environ.get("VOYAGE_API_KEY")
    )

    # Verify response
    assert response is not None
    assert hasattr(response, 'model')
    assert hasattr(response, 'data')
    assert len(response.data) > 0
    assert hasattr(response, 'usage')

    print(f"✓ Model: {response.model}")
    print(f"✓ Embedding dimensions: {len(response.data[0]['embedding'])}")
    print(f"✓ Usage: {response.usage}")


def test_voyage_context_3_through_litellm():
    """Test voyage-context-3 model through LiteLLM"""
    print("\n=== Testing voyage-context-3 through LiteLLM ===")

    response = litellm.embedding(
        model="voyage/voyage-context-3",
        input=[
            ["Paris is the capital of France.", "What is the capital of France?"],
            ["The sky is blue.", "What color is the sky?"]
        ],
        api_key=os.environ.get("VOYAGE_API_KEY")
    )

    # Verify response
    assert response is not None
    assert hasattr(response, 'model')
    assert hasattr(response, 'data')
    assert len(response.data) > 0
    assert hasattr(response, 'usage')

    print(f"✓ Model: {response.model}")
    print(f"✓ Number of groups: {len(response.data)}")
    print(f"✓ Usage: {response.usage}")


def test_voyage_3_5_with_parameters():
    """Test voyage-3.5 with advanced parameters"""
    print("\n=== Testing voyage-3.5 with parameters ===")

    response = litellm.embedding(
        model="voyage/voyage-3.5",
        input=["Testing with custom dimensions"],
        dimensions=512,
        input_type="document",
        api_key=os.environ.get("VOYAGE_API_KEY")
    )

    # Verify response
    assert response is not None
    assert len(response.data[0]['embedding']) == 512, f"Expected 512 dimensions, got {len(response.data[0]['embedding'])}"

    print(f"✓ Model: {response.model}")
    print(f"✓ Custom dimensions: {len(response.data[0]['embedding'])}")
    print(f"✓ Usage: {response.usage}")


def test_voyage_multimodal_3_5_text_only():
    """Test voyage-multimodal-3.5 with text-only input"""
    print("\n=== Testing voyage-multimodal-3.5 (text-only) ===")

    response = litellm.embedding(
        model="voyage/voyage-multimodal-3.5",
        input=["Testing multimodal embeddings with text only"],
        api_key=os.environ.get("VOYAGE_API_KEY")
    )

    # Verify response
    assert response is not None
    assert hasattr(response, 'model')
    assert hasattr(response, 'data')
    assert len(response.data) > 0
    assert hasattr(response, 'usage')

    print(f"✓ Model: {response.model}")
    print(f"✓ Embedding dimensions: {len(response.data[0]['embedding'])}")
    print(f"✓ Usage: {response.usage}")


def test_voyage_multimodal_3_5_with_image():
    """Test voyage-multimodal-3.5 with text + image input"""
    print("\n=== Testing voyage-multimodal-3.5 (text + image) ===")

    response = litellm.embedding(
        model="voyage/voyage-multimodal-3.5",
        input=[
            {
                "content": [
                    {"type": "text", "text": "A beautiful beach scene"},
                    {"type": "image_url", "image_url": "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=400"}
                ]
            }
        ],
        api_key=os.environ.get("VOYAGE_API_KEY")
    )

    # Verify response
    assert response is not None
    assert hasattr(response, 'model')
    assert hasattr(response, 'data')
    assert len(response.data) > 0
    assert hasattr(response, 'usage')

    print(f"✓ Model: {response.model}")
    print(f"✓ Embedding dimensions: {len(response.data[0]['embedding'])}")
    print(f"✓ Usage: {response.usage}")


def test_voyage_multimodal_3_text_only():
    """Test voyage-multimodal-3 with text-only input"""
    print("\n=== Testing voyage-multimodal-3 (text-only) ===")

    response = litellm.embedding(
        model="voyage/voyage-multimodal-3",
        input=["Testing multimodal-3 model"],
        api_key=os.environ.get("VOYAGE_API_KEY")
    )

    # Verify response
    assert response is not None
    assert hasattr(response, 'model')
    assert hasattr(response, 'data')
    assert len(response.data) > 0
    assert hasattr(response, 'usage')

    print(f"✓ Model: {response.model}")
    print(f"✓ Embedding dimensions: {len(response.data[0]['embedding'])}")
    print(f"✓ Usage: {response.usage}")


def test_voyage_multimodal_with_dimensions():
    """Test voyage-multimodal-3.5 with custom dimensions"""
    print("\n=== Testing voyage-multimodal-3.5 with dimensions ===")

    response = litellm.embedding(
        model="voyage/voyage-multimodal-3.5",
        input=["Testing with custom dimensions"],
        dimensions=512,
        api_key=os.environ.get("VOYAGE_API_KEY")
    )

    # Verify response
    assert response is not None
    assert len(response.data[0]['embedding']) == 512, f"Expected 512 dimensions, got {len(response.data[0]['embedding'])}"

    print(f"✓ Model: {response.model}")
    print(f"✓ Custom dimensions: {len(response.data[0]['embedding'])}")
    print(f"✓ Usage: {response.usage}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
