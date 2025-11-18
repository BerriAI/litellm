import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm import aimage_generation


@pytest.mark.parametrize(
    "model",
    [
        "fal_ai/fal-ai/flux-pro/v1.1-ultra",
        "fal_ai/fal-ai/flux-pro/v1.1",
        "fal_ai/fal-ai/flux/schnell",
        "fal_ai/fal-ai/bytedance/seedream/v3/text-to-image",
        "fal_ai/fal-ai/bytedance/dreamina/v3.1/text-to-image",
        "fal_ai/fal-ai/recraft/v3/text-to-image",
        "fal_ai/fal-ai/ideogram/v3",
        "fal_ai/bria/text-to-image/3.2",
        "fal_ai/fal-ai/stable-diffusion-v35-medium"
    ],
)
@pytest.mark.asyncio
async def test_fal_ai_image_generation_basic(model):
    """
    Test basic image generation for various Fal AI models.
    
    Tests that each model can:
    - Accept a basic text prompt
    - Return a valid response with image data
    - Handle the response properly through litellm
    """
    try:
        litellm.set_verbose = True
        
        response = await aimage_generation(
            model=model,
            prompt="A cute baby sea otter",
        )
        
        print(f"\nResponse from {model}:")
        print(f"  Number of images: {len(response.data)}")
        print(f"  First image URL: {response.data[0].url if response.data else 'None'}")
        
        # Basic assertions
        assert response is not None, f"Response should not be None for {model}"
        assert hasattr(response, "data"), f"Response should have data attribute for {model}"
        assert len(response.data) > 0, f"Response should have at least one image for {model}"
        
        # Check that we got a URL or b64_json
        first_image = response.data[0]
        assert (
            first_image.url is not None or first_image.b64_json is not None
        ), f"Image should have either url or b64_json for {model}"
        
        print(f"✓ Test passed for {model}")
        
    except litellm.RateLimitError as e:
        pytest.skip(f"Rate limit error for {model}: {str(e)}")
    except litellm.ContentPolicyViolationError as e:
        pytest.skip(f"Content policy violation for {model}: {str(e)}")
    except litellm.InternalServerError as e:
        pytest.skip(f"Internal server error for {model}: {str(e)}")
    except Exception as e:
        if "Your task failed as a result of our safety system" in str(e):
            pytest.skip(f"Safety system rejection for {model}")
        else:
            pytest.fail(f"Test failed for {model}: {str(e)}")

