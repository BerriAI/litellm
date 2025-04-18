import os
import sys
from unittest.mock import MagicMock, call, patch
import pytest
import base64
from PIL import Image
import io


import numpy as np


sys.path.insert(0, os.path.abspath("../../.."))

import litellm
from litellm.llms.diffusers.diffusers import DiffusersImageHandler

API_FUNCTION_PARAMS = [
    (
        "image_generation",
        False,
        {
            "model": "diffusers/runwayml/stable-diffusion-v1-5",
            "prompt": "A cute cat",
            "n": 1,
            "size": "512x512",
        },
    ),
    (
        "image_generation",
        True,
        {
            "model": "diffusers/runwayml/stable-diffusion-v1-5",
            "prompt": "A cute cat",
            "n": 1,
            "size": "512x512",
        },
    ),
]


@pytest.fixture
def mock_diffusers():
    """Fixture that properly mocks the diffusers pipeline"""
    with patch(
        "diffusers.StableDiffusionPipeline.from_pretrained"
    ) as mock_from_pretrained:
        # Create real test images
        def create_test_image():
            arr = np.random.rand(512, 512, 3) * 255
            return Image.fromarray(arr.astype("uint8")).convert("RGB")

        test_images = [create_test_image(), create_test_image()]

        # Create mock pipeline that returns our test images
        mock_pipe = MagicMock()
        mock_pipe.return_value.images = test_images
        mock_from_pretrained.return_value = mock_pipe

        yield {
            "from_pretrained": mock_from_pretrained,
            "pipeline": mock_pipe,
            "test_images": test_images,
        }


def test_diffusers_image_handler(mock_diffusers):
    """Test that the handler properly processes images into base64 responses"""
    from litellm.llms.diffusers.diffusers import DiffusersImageHandler

    handler = DiffusersImageHandler()

    # Test with 2 images
    response = handler.generate_image(
        prompt="test prompt",
        model="runwayml/stable-diffusion-v1-5",
        num_images_per_prompt=2,
    )

    # Verify response structure
    assert hasattr(response, "data")
    assert isinstance(response.data, list)
    assert len(response.data) == 2  # Should return exactly 2 images

    # Verify each image is properly encoded
    for img_data in response.data:
        assert "b64_json" in img_data
        # Test we can decode it back to an image
        try:
            img_bytes = base64.b64decode(img_data["b64_json"])
            img = Image.open(io.BytesIO(img_bytes))
            assert img.size == (512, 512)
        except Exception as e:
            pytest.fail(f"Failed to decode base64 image: {str(e)}")

    # Verify pipeline was called correctly
    mock_diffusers["from_pretrained"].assert_called_once_with(
        "runwayml/stable-diffusion-v1-5"
    )
    mock_diffusers["pipeline"].assert_called_once_with(
        prompt="test prompt", num_images_per_prompt=2
    )


@pytest.mark.parametrize(
    "function_name,is_async,args",
    [
        (
            "image_generation",
            False,
            {
                "model": "diffusers/runwayml/stable-diffusion-v1-5",
                "prompt": "A cat",
                "n": 1,
            },
        ),
        (
            "image_generation",
            True,
            {
                "model": "diffusers/runwayml/stable-diffusion-v1-5",
                "prompt": "A cat",
                "n": 1,
            },
        ),
    ],
)
@pytest.mark.asyncio
async def test_image_generation(function_name, is_async, args, mock_diffusers):
    """Test the image generation API endpoint"""
    # Configure mock
    mock_diffusers["pipeline"].return_value.images = mock_diffusers["test_images"][
        : args["n"]
    ]

    if is_async:
        response = await litellm.aimage_generation(**args)
    else:
        response = litellm.image_generation(**args)

    # Verify response
    assert len(response.data) == args["n"]
    assert "b64_json" in response.data[0]

    # Test base64 decoding
    try:
        img_bytes = base64.b64decode(response.data[0]["b64_json"])
        img = Image.open(io.BytesIO(img_bytes))
        assert img.size == (512, 512)
    except Exception as e:
        pytest.fail(f"Invalid base64 image: {str(e)}")
