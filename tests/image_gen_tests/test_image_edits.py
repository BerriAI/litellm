import logging
import os
import sys
import traceback
import asyncio
from typing import Optional
import pytest
import base64
from io import BytesIO
from unittest.mock import patch, AsyncMock
import json
from abc import ABC, abstractmethod

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.utils import ImageResponse
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import StandardLoggingPayload

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
        litellm._turn_on_debug()
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

TEST_IMAGES = [
    open(os.path.join(pwd, "ishaan_github.png"), "rb"),
    open(os.path.join(pwd, "litellm_site.png"), "rb"),
]

SINGLE_TEST_IMAGE = open(os.path.join(pwd, "ishaan_github.png"), "rb")

def get_test_images_as_bytesio():
    """Helper function to get test images as BytesIO objects"""
    bytesio_images = []
    for image_path in ["ishaan_github.png", "litellm_site.png"]:
        with open(os.path.join(pwd, image_path), "rb") as f:
            image_bytes = f.read()
            bytesio_images.append(BytesIO(image_bytes))
    return bytesio_images


class TestOpenAIImageEditGPTImage1(BaseLLMImageEditTest):
    """
    Concrete implementation of BaseLLMImageEditTest for OpenAI image edits.
    """

    def get_base_image_edit_call_args(self) -> dict:
        """Return base call args for OpenAI image edit"""
        return {
            "model": "gpt-image-1",
            "image": TEST_IMAGES,
        }

class TestOpenAIImageEditDallE2(BaseLLMImageEditTest):
    """
    Concrete implementation of BaseLLMImageEditTest for OpenAI DALL-E-2 image edits.
    DALL-E-2 only supports a single image (not an array).
    """

    def get_base_image_edit_call_args(self) -> dict:
        """Return base call args for OpenAI DALL-E-2 image edit (single image only)"""
        return {
            "model": "dall-e-2",
            "image": SINGLE_TEST_IMAGE,
        }


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


@pytest.mark.asyncio
async def test_azure_image_edit_litellm_sdk():
    """Test Azure image edit with mocked httpx request to validate request body and URL"""
    from litellm import image_edit, aimage_edit
    
    # Mock response for Azure image edit
    mock_response = {
        "created": 1589478378,
        "data": [
            {
                "b64_json": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
            }
        ]
    }

    class MockResponse:
        def __init__(self, json_data, status_code):
            self._json_data = json_data
            self.status_code = status_code
            self.text = json.dumps(json_data)

        def json(self):
            return self._json_data

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
    ) as mock_post:
        # Configure the mock to return our response
        mock_post.return_value = MockResponse(mock_response, 200)

        litellm._turn_on_debug()
        
        prompt = """
        Create a studio ghibli style image that combines all the reference images. Make sure the person looks like a CTO.
        """
        
        # Set up test environment variables
        test_api_base = "https://ai-api-gw-uae-north.openai.azure.com"
        test_api_key = "test-api-key"
        test_api_version = "2025-04-01-preview"
        
        result = await aimage_edit(
            prompt=prompt,
            model="azure/gpt-image-1",
            api_base=test_api_base,
            api_key=test_api_key,
            api_version=test_api_version,
            image=TEST_IMAGES,
        )
        
        # Verify the request was made correctly
        mock_post.assert_called_once()
        
        # Check the URL
        call_args = mock_post.call_args
        expected_url = f"{test_api_base}/openai/deployments/gpt-image-1/images/edits?api-version={test_api_version}"
        actual_url = call_args.args[0] if call_args.args else call_args.kwargs.get('url')
        print(f"Expected URL: {expected_url}")
        print(f"Actual URL: {actual_url}")
        assert actual_url == expected_url, f"URL mismatch. Expected: {expected_url}, Got: {actual_url}"
        
        # Check the request body
        if 'data' in call_args.kwargs:
            # For multipart form data, check the data parameter
            form_data = call_args.kwargs['data']
            print("Form data keys:", list(form_data.keys()) if hasattr(form_data, 'keys') else "Not a dict")
            
            # Validate that model and prompt are in the form data
            assert 'model' in form_data, "model should be in form data"
            assert 'prompt' in form_data, "prompt should be in form data"
            assert form_data['model'] == 'gpt-image-1', f"Expected model 'gpt-image-1', got {form_data['model']}"
            assert prompt.strip() in form_data['prompt'], f"Expected prompt to contain '{prompt.strip()}'"
            
        # Check headers
        headers = call_args.kwargs.get('headers', {})
        print("Request headers:", headers)
        assert 'Authorization' in headers, "Authorization header should be present"
        assert headers['Authorization'].startswith('Bearer '), "Authorization should be Bearer token"
        
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



@pytest.mark.asyncio
async def test_openai_image_edit_cost_tracking():
    """Test OpenAI image edit cost tracking with custom logger"""
    from litellm import image_edit, aimage_edit
    test_custom_logger = TestCustomLogger()
    litellm.logging_callback_manager._reset_all_callbacks()
    litellm.callbacks = [test_custom_logger]
    
    # Mock response for Azure image edit
    mock_response = {
        "created": 1589478378,
        "data": [
            {
                "b64_json": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
            }
        ]
    }

    class MockResponse:
        def __init__(self, json_data, status_code):
            self._json_data = json_data
            self.status_code = status_code
            self.text = json.dumps(json_data)

        def json(self):
            return self._json_data

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
    ) as mock_post:
        # Configure the mock to return our response
        mock_post.return_value = MockResponse(mock_response, 200)

        litellm._turn_on_debug()
        
        prompt = """
        Create a studio ghibli style image that combines all the reference images. Make sure the person looks like a CTO.
        """
        
        # Set up test environment variables
        
        result = await aimage_edit(
            prompt=prompt,
            model="openai/gpt-image-1",
            image=TEST_IMAGES,
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
        print("standard logging payload", json.dumps(test_custom_logger.standard_logging_payload, indent=4, default=str))

        # check model
        assert test_custom_logger.standard_logging_payload["model"] == "gpt-image-1"
        assert test_custom_logger.standard_logging_payload["custom_llm_provider"] == "openai"

        # check response_cost
        assert test_custom_logger.standard_logging_payload["response_cost"] is not None
        assert test_custom_logger.standard_logging_payload["response_cost"] > 0




@pytest.mark.asyncio
async def test_azure_image_edit_cost_tracking():
    """Test Azure image edit cost tracking with custom logger"""
    from litellm import image_edit, aimage_edit
    test_custom_logger = TestCustomLogger()
    litellm.logging_callback_manager._reset_all_callbacks()
    litellm.callbacks = [test_custom_logger]
    
    # Mock response for Azure image edit
    mock_response = {
        "created": 1589478378,
        "data": [
            {
                "b64_json": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
            }
        ]
    }

    class MockResponse:
        def __init__(self, json_data, status_code):
            self._json_data = json_data
            self.status_code = status_code
            self.text = json.dumps(json_data)

        def json(self):
            return self._json_data

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
    ) as mock_post:
        # Configure the mock to return our response
        mock_post.return_value = MockResponse(mock_response, 200)

        litellm._turn_on_debug()
        
        prompt = """
        Create a studio ghibli style image that combines all the reference images. Make sure the person looks like a CTO.
        """
        
        # Set up test environment variables
        
        result = await aimage_edit(
            prompt=prompt,
            model="azure/CUSTOM_AZURE_DEPLOYMENT_NAME",
            base_model="azure/gpt-image-1",
            image=TEST_IMAGES,
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
        print("standard logging payload", json.dumps(test_custom_logger.standard_logging_payload, indent=4, default=str))

        # check model
        assert test_custom_logger.standard_logging_payload["model"] == "CUSTOM_AZURE_DEPLOYMENT_NAME"
        assert test_custom_logger.standard_logging_payload["custom_llm_provider"] == "azure"

        # check response_cost
        assert test_custom_logger.standard_logging_payload["response_cost"] is not None
        assert test_custom_logger.standard_logging_payload["response_cost"] > 0


@pytest.mark.asyncio
async def test_recraft_image_edit_api():
    from litellm import aimage_edit
    import requests
    litellm._turn_on_debug()
    global TEST_IMAGES
    try:
        prompt = """
        Create a studio ghibli style image that combines all the reference images. Make sure the person looks like a CTO.
        """
        result = await aimage_edit(
            prompt=prompt,
            model="recraft/recraftv3",
            image=TEST_IMAGES,
        )
        print("result from image edit", result)

        # Validate the response meets expected schema
        ImageResponse.model_validate(result)
        
        if isinstance(result, ImageResponse) and result.data:
            image_url = result.data[0].url
            
            # download the image
            image_bytes = requests.get(image_url).content
            with open("test_image_edit.png", "wb") as f:
                f.write(image_bytes)
    except litellm.ContentPolicyViolationError as e:
        pass


def test_recraft_image_edit_config():
    """
    Test Recraft image edit configuration parameter mapping and request transformation.
    """
    from litellm.llms.recraft.image_edit.transformation import RecraftImageEditConfig
    from litellm.types.images.main import ImageEditOptionalRequestParams
    from litellm.types.router import GenericLiteLLMParams
    
    config = RecraftImageEditConfig()
    
    # Test supported OpenAI params
    supported_params = config.get_supported_openai_params("recraftv3")
    expected_params = ["n", "response_format", "style"]
    assert supported_params == expected_params
    
    # Test parameter mapping (reuses OpenAI logic with filtering)
    image_edit_params = ImageEditOptionalRequestParams({
        "n": 2,
        "response_format": "b64_json",
        "style": "realistic_image",
        "size": "1024x1024",  # Should be dropped
        "quality": "high"     # Should be dropped
    })
    
    mapped_params = config.map_openai_params(image_edit_params, "recraftv3", drop_params=True)
    
    # Should only contain supported params
    assert mapped_params["n"] == 2
    assert mapped_params["response_format"] == "b64_json"
    assert mapped_params["style"] == "realistic_image"
    assert "size" not in mapped_params  # Should be dropped
    assert "quality" not in mapped_params  # Should be dropped
    
    # Test request transformation (reuses OpenAI file handling)
    mock_image = b"fake_image_data"
    prompt = "winter landscape"
    litellm_params = GenericLiteLLMParams(api_key="test_key")
    
    data, files = config.transform_image_edit_request(
        model="recraftv3",
        prompt=prompt,
        image=mock_image,
        image_edit_optional_request_params={"strength": 0.7, "n": 1},
        litellm_params=litellm_params,
        headers={}
    )
    
    # Check data structure (like OpenAI but with Recraft additions)
    assert data["prompt"] == prompt
    assert data["strength"] == 0.7  # Recraft-specific parameter
    assert data["model"] == "recraftv3"
    
    # Check file structure (reuses OpenAI logic)
    assert len(files) == 1
    assert files[0][0] == "image"  # Field name (not image[] like OpenAI)
    assert files[0][1][1] == mock_image  # Image data
    assert files[0][1][2] == "image/png"  # Content type


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.flaky(retries=3, delay=2)
@pytest.mark.asyncio
async def test_multiple_vs_single_image_edit(sync_mode):
    """Test that both single and multiple image editing work correctly"""
    from litellm import image_edit, aimage_edit
    litellm._turn_on_debug()
    
    try:
        prompt = "Add a soft blue tint to the image(s)"
        
        # Test single image
        if sync_mode:
            single_result = image_edit(
                prompt=prompt,
                model="gpt-image-1",
                image=SINGLE_TEST_IMAGE,
            )
        else:
            single_result = await aimage_edit(
                prompt=prompt,
                model="gpt-image-1",
                image=SINGLE_TEST_IMAGE,
            )
        
        print("Single image result:", single_result)
        ImageResponse.model_validate(single_result)
        
        # Test multiple images
        if sync_mode:
            multiple_result = image_edit(
                prompt=prompt,
                model="gpt-image-1",
                image=TEST_IMAGES,
            )
        else:
            multiple_result = await aimage_edit(
                prompt=prompt,
                model="gpt-image-1",
                image=TEST_IMAGES,
            )
        
        print("Multiple images result:", multiple_result)
        ImageResponse.model_validate(multiple_result)
        
        # Both should return valid responses
        assert single_result is not None
        assert multiple_result is not None
        assert single_result.data is not None
        assert multiple_result.data is not None
        assert len(single_result.data) > 0
        assert len(multiple_result.data) > 0
        
    except litellm.ContentPolicyViolationError as e:
        pytest.skip(f"Content policy violation: {e}")


@pytest.mark.flaky(retries=3, delay=2)
@pytest.mark.asyncio
async def test_multiple_image_edit_with_different_formats():
    """Test multiple images editing with different file formats and types"""
    from litellm import aimage_edit
    litellm._turn_on_debug()
    
    try:
        prompt = "Create a cohesive artistic style across all images"
        
        # Test with mixed BytesIO and file objects
        mixed_images = [
            SINGLE_TEST_IMAGE,  # File object
            get_test_images_as_bytesio()[1]  # BytesIO object
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


@pytest.mark.flaky(retries=3, delay=2)
@pytest.mark.asyncio
async def test_image_edit_array_handling():
    """Test that the image parameter correctly handles both single items and arrays"""
    from litellm import aimage_edit
    
    # Mock response
    mock_response = {
        "created": 1589478378,
        "data": [
            {
                "b64_json": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
            }
        ]
    }

    class MockResponse:
        def __init__(self, json_data, status_code):
            self._json_data = json_data
            self.status_code = status_code
            self.text = json.dumps(json_data)

        def json(self):
            return self._json_data

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
    ) as mock_post:
        mock_post.return_value = MockResponse(mock_response, 200)
        
        prompt = "Test prompt"
        
        # Test 1: Single image (should be converted to list internally)
        result1 = await aimage_edit(
            prompt=prompt,
            model="gpt-image-1",
            image=SINGLE_TEST_IMAGE,
        )
        
        # Test 2: Multiple images (already a list)
        result2 = await aimage_edit(
            prompt=prompt,
            model="gpt-image-1",
            image=TEST_IMAGES,
        )
        

        # Both valid calls should succeed
        ImageResponse.model_validate(result1)
        ImageResponse.model_validate(result2)
        
        # Verify that both calls were made to the API
        assert mock_post.call_count == 2


