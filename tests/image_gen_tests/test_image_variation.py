# What this tests?
## This tests the litellm support for the openai /generations endpoint

import logging
import os
import sys
import traceback


sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

from dotenv import load_dotenv
from openai.types.image import Image
from litellm.caching import InMemoryCache

logging.basicConfig(level=logging.DEBUG)
load_dotenv()
import asyncio
import os
import pytest

import litellm
import json
import tempfile
from base_image_generation_test import BaseImageGenTest
import logging
from litellm._logging import verbose_logger
from io import BytesIO
from PIL import Image as PILImage

verbose_logger.setLevel(logging.DEBUG)


@pytest.fixture
def image_url():
    # DALL-E 2 image variations require a square PNG (less than 4MB)
    # Generate a 1024x1024 square PNG programmatically to avoid network dependency
    # and the non-square aspect ratio of the old LiteLLM logo URL
    img = PILImage.new("RGBA", (1024, 1024), color=(128, 128, 128, 255))
    image_file = BytesIO()
    img.save(image_file, format="PNG")
    image_file.seek(0)
    # openai>=2.24.0 requires BytesIO to have .name for MIME type detection in multipart uploads
    image_file.name = "litellm_logo.png"

    return image_file


# Commented out: OpenAI /images/variations endpoint deprecated (DALL-E 2 shutdown May 12, 2026)
# def test_openai_image_variation_openai_sdk(image_url):
#     from openai import OpenAI
#
#     client = OpenAI(timeout=60.0, max_retries=3)
#     response = client.images.create_variation(image=image_url, n=2, size="1024x1024")
#     print(response)
#
#
# @pytest.mark.parametrize("sync_mode", [True, False])
# @pytest.mark.asyncio
# async def test_openai_image_variation_litellm_sdk(image_url, sync_mode):
#     from litellm import image_variation, aimage_variation
#
#     if sync_mode:
#         image_variation(image=image_url, n=2, size="1024x1024")
#     else:
#         await aimage_variation(image=image_url, n=2, size="1024x1024")
#
#
# def test_topaz_image_variation(image_url):
#     from litellm import image_variation, aimage_variation
#     from litellm.llms.custom_httpx.http_handler import HTTPHandler
#     from unittest.mock import patch
#
#     client = HTTPHandler()
#     with patch.object(client, "post") as mock_post:
#         try:
#             image_variation(
#                 model="topaz/Standard V2",
#                 image=image_url,
#                 n=2,
#                 size="1024x1024",
#                 client=client,
#             )
#         except Exception as e:
#             print(e)
#         mock_post.assert_called_once()


def test_image_variation_placeholder():
    """Placeholder: variation tests commented out - OpenAI /images/variations deprecated (DALL-E 2 shutdown May 12, 2026)."""
    pass
