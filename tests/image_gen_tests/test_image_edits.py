import asyncio
import base64
import json
import logging
import os
import traceback
from abc import ABC, abstractmethod
from io import BytesIO
from typing import Optional
from unittest.mock import AsyncMock, patch

import pytest

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import StandardLoggingPayload
from litellm.utils import ImageResponse

# Configure pytest marks to avoid warnings
pytestmark = pytest.mark.asyncio


class TestCustomLogger(CustomLogger):
    def __init__(self):
        self.standard_logging_payload: Optional[StandardLoggingPayload] = None

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self.standard_logging_payload = kwargs.get("standard_logging_object", None)
        pass


class BaseLLMImageEditTest(ABC):
    """
    Abstract base test class that enforces a common test across all image edit test classes.
    """

    @property
    def image_edit_function(self):
        return litellm.image_edit

    @property
    def async_image_edit_function(self):
        return litellm.aimage_edit

    @abstractmethod
    def get_base_image_edit_call_args(self) -> dict:
        """Must return the base image edit call args"""
        pass

    @pytest.fixture(autouse=True)
    def _handle_rate_limits(self):
        """Fixture to handle rate limit errors for all test methods"""
        try:
            yield
        except litellm.RateLimitError:
            pytest.skip("Rate limit exceeded")
        except litellm.InternalServerError:
            pytest.skip("Model is overloaded")

    @pytest.mark.parametrize("sync_mode", [True, False])
    @pytest.mark.flaky(retries=3, delay=2)
    @pytest.mark.asyncio
    async def test_openai_image_edit_litellm_sdk(self, sync_mode):
        """
        Test image edit functionality with both sync and async modes.
        """
        litellm.turn_on_debug()
        try:
            prompt = """
            Create a studio ghibli style image that combines all the reference images. Make sure the person looks like a CTO.
            """

            call_args = self.get_base_image_edit_call_args()
            call_args["prompt"] = prompt

            if sync_mode:
                result = self.image_edit_function(**call_args)
            else:
                result = await self.async_image_edit_function(**call_args)

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


# Get the current directory of the file being run
pwd = os.path.dirname(os.path.realpath(__file__))


def _read_image_bytes(filename: str) -> bytes:
    with open(os.path.join(pwd, filename), "rb") as f:
        return f.read()


_ISHAAN_GITHUB_BYTES = _read_image_bytes("ishaan_github.png")
_LITELLM_SITE_BYTES = _read_image_bytes("litellm_site.png")


def _make_test_images() -> list:
    return [_ISHAAN_GITHUB_BYTES, _LITELLM_SITE_BYTES]


def _make_single_test_image() -> bytes:
    return _ISHAAN_GITHUB_BYTES


def get_test_images_as_bytesio():
    return [
        BytesIO(_ISHAAN_GITHUB_BYTES),
        BytesIO(_LITELLM_SITE_BYTES),
    ]


class TestOpenAIImageEditGPTImage1(BaseLLMImageEditTest):
    """
    Concrete implementation of BaseLLMImageEditTest for OpenAI image edits.
    """

    test_openai_image_edit_litellm_sdk = None

    def get_base_image_edit_call_args(self) -> dict:
        """Return base call args for OpenAI image edit"""
        return {
            "model": "gpt-image-1",
            "image": _make_test_images(),
        }


class TestAzureAIFlux2ImageEdit(BaseLLMImageEditTest):
    """
    Concrete implementation of BaseLLMImageEditTest for Azure AI FLUX 2 image edits.
    FLUX 2 uses JSON with base64 image instead of multipart/form-data.
    """

    def get_base_image_edit_call_args(self) -> dict:
        """Return base call args for Azure AI FLUX 2 image edit"""
        return {
            "model": "azure_ai/flux.2-pro",
            "image": _make_single_test_image(),
            "api_base": os.getenv("AZURE_AI_API_BASE"),
            "api_key": os.getenv("AZURE_AI_API_KEY"),
            "api_version": "preview",
        }


@pytest.mark.flaky(retries=3, delay=2)
@pytest.mark.asyncio
async def test_openai_image_edit_litellm_router():
    litellm.turn_on_debug()
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
            image=_make_test_images(),
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
