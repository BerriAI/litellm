import os
import sys

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import pytest

import litellm
from litellm import embedding

litellm.num_retries = 3


def test_cohere_embedding_outout_dimensions():
    litellm._turn_on_debug()
    response = embedding(
        model="cohere/embed-v4.0", input="Hello, world!", dimensions=512
    )
    print(f"response: {response}\n")
    assert len(response.data[0]["embedding"]) == 512


# Comprehensive Cohere Embed v4 tests
@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_cohere_embed_v4_basic_text(sync_mode):
    """Test basic text embedding functionality with Cohere Embed v4."""
    try:
        data = {
            "model": "cohere/embed-v4.0",
            "input": ["Hello world!", "This is a test sentence."],
            "input_type": "search_document",
        }

        if sync_mode:
            response = embedding(**data)
        else:
            response = await litellm.aembedding(**data)

        # Validate response structure
        assert response.model is not None
        assert len(response.data) == 2
        assert response.data[0]["object"] == "embedding"
        assert len(response.data[0]["embedding"]) > 0
        assert response.usage.prompt_tokens > 0
        assert isinstance(response.usage, litellm.Usage)

    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_cohere_embed_v4_with_dimensions(sync_mode):
    """Test Cohere Embed v4 with specific dimension parameter."""
    try:
        data = {
            "model": "cohere/embed-v4.0",
            "input": ["Test with custom dimensions"],
            "dimensions": 512,
            "input_type": "search_query",
        }

        if sync_mode:
            response = embedding(**data)
        else:
            response = await litellm.aembedding(**data)

        # Validate dimension
        assert len(response.data[0]["embedding"]) == 512
        assert isinstance(response.usage, litellm.Usage)

    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_cohere_embed_v4_image_embedding(sync_mode):
    """Test Cohere Embed v4 image embedding functionality (multimodal)."""
    try:
        import base64

        # 1x1 pixel red PNG (base64 encoded)
        test_image_data = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\tpHYs\x00\x00\x0b\x13\x00\x00\x0b\x13\x01\x00\x9a\x9c\x18\x00\x00\x00\x0cIDATx\x9cc\xf8\x00\x00\x00\x01\x00\x01\x00\x00\x00\x00"
        test_image_b64 = base64.b64encode(test_image_data).decode("utf-8")

        data = {
            "model": "cohere/embed-v4.0",
            "input": [test_image_b64],
            "input_type": "image",
        }

        if sync_mode:
            response = embedding(**data)
        else:
            response = await litellm.aembedding(**data)

        # Validate response structure for image embedding
        assert response.model is not None
        assert len(response.data) == 1
        assert response.data[0]["object"] == "embedding"
        assert len(response.data[0]["embedding"]) > 0
        assert isinstance(response.usage, litellm.Usage)

    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.parametrize(
    "input_type", ["search_document", "search_query", "classification", "clustering"]
)
@pytest.mark.asyncio
async def test_cohere_embed_v4_input_types(input_type):
    """Test Cohere Embed v4 with different input types."""
    try:
        response = await litellm.aembedding(
            model="cohere/embed-v4.0",
            input=[f"Test text for {input_type}"],
            input_type=input_type,
        )

        assert response.model is not None
        assert len(response.data) == 1
        assert response.data[0]["object"] == "embedding"
        assert len(response.data[0]["embedding"]) > 0
        assert isinstance(response.usage, litellm.Usage)

    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_cohere_embed_v4_encoding_format():
    """Test Cohere Embed v4 with different encoding formats."""
    try:
        response = embedding(
            model="cohere/embed-v4.0",
            input=["Test encoding format"],
            encoding_format="float",
        )

        assert response.model is not None
        assert len(response.data) == 1
        assert response.data[0]["object"] == "embedding"
        assert len(response.data[0]["embedding"]) > 0
        # Validate that embeddings are floats
        assert all(isinstance(x, float) for x in response.data[0]["embedding"])
        assert isinstance(response.usage, litellm.Usage)

    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_cohere_embed_v4_error_handling():
    """Test error handling for Cohere Embed v4 with invalid inputs."""
    try:
        # Test with empty input - should raise an error
        try:
            response = embedding(model="cohere/embed-v4.0", input=[])  # Empty input
            pytest.fail("Should have failed with empty input")
        except Exception:
            pass  # Expected to fail

        # Test with None input - should raise an error
        try:
            response = embedding(model="cohere/embed-v4.0", input=None)
            pytest.fail("Should have failed with None input")
        except Exception:
            pass  # Expected to fail

    except Exception as e:
        pytest.fail(f"Error in error handling test: {e}")


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_cohere_embed_v4_multiple_texts(sync_mode):
    """Test Cohere Embed v4 with multiple text inputs."""
    try:
        texts = [
            "The quick brown fox jumps over the lazy dog",
            "Machine learning is transforming the world",
            "Python is a versatile programming language",
            "Natural language processing enables human-computer interaction",
        ]

        data = {
            "model": "cohere/embed-v4.0",
            "input": texts,
            "input_type": "search_document",
        }

        if sync_mode:
            response = embedding(**data)
        else:
            response = await litellm.aembedding(**data)

        # Validate response structure
        assert response.model is not None
        assert len(response.data) == len(texts)

        for i, data_item in enumerate(response.data):
            assert data_item["object"] == "embedding"
            assert data_item["index"] == i
            assert len(data_item["embedding"]) > 0
            assert all(isinstance(x, float) for x in data_item["embedding"])

        assert isinstance(response.usage, litellm.Usage)
        assert response.usage.prompt_tokens > 0

    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_cohere_embed_v4_with_optional_params():
    """Test Cohere Embed v4 with various optional parameters."""
    try:
        response = embedding(
            model="cohere/embed-v4.0",
            input=["Test with optional parameters"],
            input_type="search_query",
            dimensions=256,
            encoding_format="float",
        )

        # Validate response
        assert response.model is not None
        assert len(response.data) == 1
        assert response.data[0]["object"] == "embedding"
        assert len(response.data[0]["embedding"]) == 256  # Custom dimensions
        assert all(isinstance(x, float) for x in response.data[0]["embedding"])
        assert isinstance(response.usage, litellm.Usage)

    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# ==================== COHERE V2 API TESTS ====================


