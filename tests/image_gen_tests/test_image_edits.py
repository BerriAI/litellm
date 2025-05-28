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

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.utils import ImageResponse
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import StandardLoggingPayload

class TestCustomLogger(CustomLogger):
    def __init__(self):
        self.standard_logging_payload: Optional[StandardLoggingPayload] = None
    
    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self.standard_logging_payload = kwargs.get("standard_logging_object", None)
        pass

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
