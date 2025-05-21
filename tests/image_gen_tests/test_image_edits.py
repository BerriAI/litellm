

import logging
import os
import sys
import traceback
import pytest
import base64

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.utils import ImageResponse

@pytest.mark.parametrize("sync_mode", [True])
@pytest.mark.asyncio
async def test_openai_image_edit_litellm_sdk(sync_mode):
    from litellm import image_edit, aimage_edit
    litellm._turn_on_debug()

    prompt = """
    Create a studio ghibli style image that combines all the reference images. Make sure the person looks like a CTO.
    """
    image=[
        open("ishaan_github.png", "rb"),
        open("litellm_site.png", "rb"),
    ]

    if sync_mode:
        result = image_edit(
            prompt=prompt,
            model="gpt-image-1",
            image=image,
        )
    else:
        result = await aimage_edit(
            prompt=prompt,
            model="gpt-image-1",
            image=image,
        )
    print("result from image edit", result)
    
    if isinstance(result, ImageResponse):
        image_base64 = result.data[0].b64_json
        image_bytes = base64.b64decode(image_base64)

        # Save the image to a file
        with open("test_image_edit.png", "wb") as f:
            f.write(image_bytes)
