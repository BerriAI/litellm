"""
E2E test for Vertex AI Imagen API using the JSON-driven endpoint.

Run with:
    cd tests/unified_google_tests
    pytest test_vertex_imagen.py -v -s
"""

import base64

import pytest

from litellm.experimental.endpoint_definitions.vertex_ai.imagen import generate_image


def test_vertex_imagen_generate():
    """Test generating an image with Vertex AI Imagen API."""
    response = generate_image(
        instances=[{"prompt": "A simple red circle on a white background"}],
        parameters={"sampleCount": 1},
    )

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

