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


@pytest.mark.flaky(retries=3, delay=2)
@pytest.mark.asyncio
async def test_openai_image_edit_with_bytesio():
    """Test image editing using BytesIO objects instead of file readers"""
    from litellm import aimage_edit, image_edit

    litellm.turn_on_debug()
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






@pytest.mark.asyncio
async def test_azure_image_edit_cost_tracking():
    """Test Azure image edit cost tracking with custom logger"""
    from litellm import aimage_edit, image_edit

    test_custom_logger = TestCustomLogger()
    litellm.logging_callback_manager._reset_all_callbacks()
    litellm.callbacks = [test_custom_logger]

    # Mock response for Azure image edit with usage data for cost tracking
    mock_response = {
        "created": 1589478378,
        "data": [
            {
                "b64_json": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
            }
        ],
        "usage": {
            "total_tokens": 1100,
            "input_tokens": 100,
            "input_tokens_details": {"image_tokens": 50, "text_tokens": 50},
            "output_tokens": 1000,
        },
    }

    class MockResponse:
        def __init__(self, json_data, status_code):
            self._json_data = json_data
            self.status_code = status_code
            self.text = json.dumps(json_data)
            self.headers = {}

        def json(self):
            return self._json_data

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
    ) as mock_post:
        # Configure the mock to return our response
        mock_post.return_value = MockResponse(mock_response, 200)

        litellm.turn_on_debug()

        prompt = """
        Create a studio ghibli style image that combines all the reference images. Make sure the person looks like a CTO.
        """

        # Set up test environment variables

        result = await aimage_edit(
            prompt=prompt,
            model="azure/CUSTOM_AZURE_DEPLOYMENT_NAME",
            base_model="azure/gpt-image-1",
            image=_make_test_images(),
        )

        # Verify the request was made correctly
        mock_post.assert_called_once()

        # Validate the response meets expected schema
        ImageResponse.model_validate(result)

        if isinstance(result, ImageResponse) and result.data:
            image_base64 = result.data[0].b64_json
            if image_base64:
                image_bytes = base64.b64decode(image_base64)

                # Save the image to a file
                with open("test_image_edit.png", "wb") as f:
                    f.write(image_bytes)

        await asyncio.sleep(5)
        print(
            "standard logging payload",
            json.dumps(
                test_custom_logger.standard_logging_payload, indent=4, default=str
            ),
        )

        # check model
        assert (
            test_custom_logger.standard_logging_payload["model"]
            == "CUSTOM_AZURE_DEPLOYMENT_NAME"
        )
        assert (
            test_custom_logger.standard_logging_payload["custom_llm_provider"]
            == "azure"
        )

        # check response_cost
        assert test_custom_logger.standard_logging_payload["response_cost"] is not None
        assert test_custom_logger.standard_logging_payload["response_cost"] > 0






@pytest.mark.flaky(retries=3, delay=2)
@pytest.mark.asyncio
async def test_multiple_image_edit_with_different_formats():
    """Test multiple images editing with different file formats and types"""
    from litellm import aimage_edit

    litellm.turn_on_debug()

    try:
        prompt = "Create a cohesive artistic style across all images"

        mixed_images = [
            _make_single_test_image(),
            get_test_images_as_bytesio()[1],
        ]

        result = await aimage_edit(
            prompt=prompt,
            model="gpt-image-1",
            image=mixed_images,
        )

        print("Mixed format images result:", result)
        ImageResponse.model_validate(result)

        assert result is not None
        assert result.data is not None
        assert len(result.data) > 0

        # Save result if available
        if result.data and result.data[0].b64_json:
            image_bytes = base64.b64decode(result.data[0].b64_json)
            with open("test_multiple_image_edit_mixed.png", "wb") as f:
                f.write(image_bytes)

    except litellm.ContentPolicyViolationError as e:
        pytest.skip(f"Content policy violation: {e}")
