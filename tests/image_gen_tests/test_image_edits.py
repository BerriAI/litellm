import logging
import os
import sys
import traceback
import pytest
import base64
from io import BytesIO

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.utils import ImageResponse
# Get the current directory of the file being run
pwd = os.path.dirname(os.path.realpath(__file__))

TEST_IMAGES = [
    open(os.path.join(pwd, "ishaan_github.png"), "rb"),
    open(os.path.join(pwd, "litellm_site.png"), "rb"),
]

def get_test_images_as_bytesio():
    """Helper function to get test images as BytesIO objects"""
    bytesio_images = []
    for image_path in ["ishaan_github.png", "litellm_site.png"]:
        with open(os.path.join(pwd, image_path), "rb") as f:
            image_bytes = f.read()
            bytesio_images.append(BytesIO(image_bytes))
    return bytesio_images

@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.flaky(retries=3, delay=2)
@pytest.mark.asyncio
async def test_openai_image_edit_litellm_sdk(sync_mode):
    from litellm import image_edit, aimage_edit
    litellm._turn_on_debug()
    try:
        prompt = """
        Create a studio ghibli style image that combines all the reference images. Make sure the person looks like a CTO.
        """

        if sync_mode:
            result = image_edit(
                prompt=prompt,
                model="gpt-image-1",
                image=TEST_IMAGES,
            )
        else:
            result = await aimage_edit(
                prompt=prompt,
                model="gpt-image-1",
                image=TEST_IMAGES,
            )
        print("result from image edit", result)

        # Validate the response meets expected schema
        ImageResponse.model_validate(result)
        
        if isinstance(result, ImageResponse) and result.data:
            image_base64 = result.data[0].b64_json
            if image_base64:
                image_bytes = base64.b64decode(image_base64)

                # Save the image to a file
                with open("test_image_edit.png", "wb") as f:
                    f.write(image_bytes)
    except litellm.ContentPolicyViolationError as e:
        pass



@pytest.mark.flaky(retries=3, delay=2)
@pytest.mark.asyncio
async def test_openai_image_edit_litellm_router():
    litellm._turn_on_debug()
    try:
        prompt = """
        Create a studio ghibli style image that combines all the reference images. Make sure the person looks like a CTO.
        """
        router = litellm.Router(
            model_list=[
                {
                    "model_name": "gpt-image-1",
                    "litellm_params": {
                        "model": "gpt-image-1",
                    },
                }
            ]
        )
        result = await router.aimage_edit(
            prompt=prompt,
            model="gpt-image-1",
            image=TEST_IMAGES,
        )
        print("result from image edit", result)

        # Validate the response meets expected schema
        ImageResponse.model_validate(result)
        
        if isinstance(result, ImageResponse) and result.data:
            image_base64 = result.data[0].b64_json
            if image_base64:
                image_bytes = base64.b64decode(image_base64)

                # Save the image to a file
                with open("test_image_edit.png", "wb") as f:
                    f.write(image_bytes)
    except litellm.ContentPolicyViolationError as e:
        pass

@pytest.mark.flaky(retries=3, delay=2)
@pytest.mark.asyncio
async def test_openai_image_edit_with_bytesio():
    """Test image editing using BytesIO objects instead of file readers"""
    from litellm import image_edit, aimage_edit
    litellm._turn_on_debug()
    try:
        prompt = """
        Create a studio ghibli style image that combines all the reference images. Make sure the person looks like a CTO.
        """
        
        # Get images as BytesIO objects
        bytesio_images = get_test_images_as_bytesio()

        result = await aimage_edit(
            prompt=prompt,
            model="gpt-image-1",
            image=bytesio_images,
        )
        print("result from image edit with BytesIO", result)

        # Validate the response meets expected schema
        ImageResponse.model_validate(result)
        
        if isinstance(result, ImageResponse) and result.data:
            image_base64 = result.data[0].b64_json
            if image_base64:
                image_bytes = base64.b64decode(image_base64)

                # Save the image to a file
                with open("test_image_edit_bytesio.png", "wb") as f:
                    f.write(image_bytes)
    except litellm.ContentPolicyViolationError as e:
        pass



# @pytest.mark.flaky(retries=3, delay=2)
@pytest.mark.asyncio
async def test_azure_image_edit_litellm_sdk():
    from litellm import image_edit, aimage_edit
    litellm._turn_on_debug()
    try:
        prompt = """
        Create a studio ghibli style image that combines all the reference images. Make sure the person looks like a CTO.
        """
        result = await aimage_edit(
            prompt=prompt,
            model="azure/Dalle3",
            api_base=os.environ.get("AZURE_IMAGE_ENDPOINT"),
            api_key=os.environ.get("AZURE_IMAGE_API_KEY"),
            api_version=os.environ.get("AZURE_IMAGE_API_VERSION"),
            image=TEST_IMAGES,
        )
        print("result from image edit", result)

        # Validate the response meets expected schema
        ImageResponse.model_validate(result)
        
        if isinstance(result, ImageResponse) and result.data:
            image_base64 = result.data[0].b64_json
            if image_base64:
                image_bytes = base64.b64decode(image_base64)

                # Save the image to a file
                with open("test_image_edit.png", "wb") as f:
                    f.write(image_bytes)
    except litellm.ContentPolicyViolationError as e:
        pass

