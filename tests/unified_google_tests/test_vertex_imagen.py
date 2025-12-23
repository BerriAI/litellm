"""
E2E test for Vertex AI Imagen API using the JSON-driven endpoint.

Tests both:
1. Direct Vertex AI Imagen calls
2. Model routing to other providers (e.g., gpt-image-1) with Imagen-format response

Run with:
    pytest tests/unified_google_tests/test_vertex_imagen.py -v -s
"""

import base64

import pytest

import litellm
from litellm.experimental.endpoint_definitions.vertex_ai.imagen import generate_image


def test_vertex_imagen_generate():
    """Test generating an image with Vertex AI Imagen API."""
    litellm._turn_on_debug()
    response = generate_image(
        instances=[{"prompt": "A simple red circle on a white background"}],
        parameters={"sampleCount": 1},
    )
    print("RESPONSE: ", response)

    # Verify response structure matches Vertex AI API
    assert "predictions" in response
    assert len(response["predictions"]) == 1

    prediction = response["predictions"][0]
    assert "bytesBase64Encoded" in prediction
    assert "mimeType" in prediction

    # Verify we got actual image data
    image_bytes = base64.b64decode(prediction["bytesBase64Encoded"])
    assert len(image_bytes) > 0
    print(f"Generated image: {prediction['mimeType']}, {len(image_bytes)} bytes")


def test_generate_image_with_gpt_model():
    """Test generating an image using GPT model through Imagen-style interface.
    
    This test verifies that:
    1. Non-Imagen models are routed to litellm.image_generation
    2. The response is transformed to Imagen format (predictions with bytesBase64Encoded)
    """
    litellm._turn_on_debug()
    
    response = generate_image(
        model="gpt-image-1",
        instances=[{"prompt": "A simple blue square on a white background"}],
        parameters={"sampleCount": 1},
    )
    print("GPT RESPONSE: ", response)

    # Verify response is in Imagen format
    assert "predictions" in response
    assert len(response["predictions"]) >= 1

    prediction = response["predictions"][0]
    # Should have either bytesBase64Encoded or url
    assert "bytesBase64Encoded" in prediction or "url" in prediction
    assert "mimeType" in prediction

    # Verify we got actual image data
    if "bytesBase64Encoded" in prediction:
        image_bytes = base64.b64decode(prediction["bytesBase64Encoded"])
        assert len(image_bytes) > 0
        print(f"Generated image via GPT: {prediction['mimeType']}, {len(image_bytes)} bytes")
    else:
        print(f"Generated image via GPT: {prediction['mimeType']}, URL: {prediction['url']}")

