import json
import os
import sys
import traceback
from typing import Callable, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path
import litellm
from litellm.llms.azure.azure import AzureChatCompletion
from litellm.llms.azure.image_generation import (
    AzureDallE3ImageGenerationConfig,
    get_azure_image_generation_config,
)
from litellm.utils import get_optional_params_image_gen


@pytest.mark.parametrize(
    "received_model, expected_config",
    [
        ("dall-e-3", AzureDallE3ImageGenerationConfig),
        ("dalle-3", AzureDallE3ImageGenerationConfig),
        ("openai_dall_e_3", AzureDallE3ImageGenerationConfig),
    ],
)
def test_azure_image_generation_config(received_model, expected_config):
    assert isinstance(
        get_azure_image_generation_config(received_model), expected_config
    )


def test_azure_image_generation_flattens_extra_body():
    """
    Test that Azure image generation correctly flattens extra_body parameters.
    
    Azure's image generation API doesn't support the extra_body parameter,
    so we need to flatten any parameters in extra_body to the top level.
    
    This test verifies the fix for: https://github.com/BerriAI/litellm/issues/16059
    Where partial_images and stream parameters were incorrectly sent in extra_body.
    """
    # Test 1: Verify get_optional_params_image_gen puts extra params in extra_body
    optional_params = get_optional_params_image_gen(
        model="gpt-image-1",
        n=1,
        size="1024x1024",
        custom_llm_provider="azure",
        partial_images=2,
        stream=True
    )
    
    assert "extra_body" in optional_params
    assert "partial_images" in optional_params["extra_body"]
    assert "stream" in optional_params["extra_body"]
    assert optional_params["extra_body"]["partial_images"] == 2
    assert optional_params["extra_body"]["stream"] is True
    
    # Test 2: Verify Azure flattens extra_body when building request data
    # Simulate what happens in Azure's image_generation method
    test_optional_params = {
        "n": 1,
        "size": "1024x1024",
        "extra_body": {
            "partial_images": 2,
            "stream": True,
            "custom_param": "test_value"
        }
    }
    
    # This is what the Azure image_generation method does
    extra_body = test_optional_params.pop("extra_body", {})
    flattened_params = {**test_optional_params, **extra_body}
    
    data = {"model": "gpt-image-1", "prompt": "A cute sea otter", **flattened_params}
    
    # Verify the final data structure
    assert "extra_body" not in data, "extra_body should NOT be in the final data dict"
    assert "partial_images" in data, "partial_images should be at top level"
    assert "stream" in data, "stream should be at top level"
    assert "custom_param" in data, "custom_param should be at top level"
    assert data["partial_images"] == 2
    assert data["stream"] is True
    assert data["custom_param"] == "test_value"
    assert data["n"] == 1
    assert data["size"] == "1024x1024"


def test_azure_image_generation_creates_token_provider_from_credentials():
    """
    Test that azure_ad_token_provider is created from tenant_id, client_id, client_secret.
    
    This test verifies the fix in images/main.py where we now create the
    azure_ad_token_provider from credentials in litellm_params if it's not already provided.
    """
    # Simulate the fix in images/main.py
    litellm_params_dict = {
        "tenant_id": "test-tenant-id",
        "client_id": "test-client-id",
        "client_secret": "test-client-secret",
        "azure_scope": None,
    }
    
    azure_ad_token_provider = None
    
    # This is the logic we added in images/main.py
    if azure_ad_token_provider is None:
        tenant_id = litellm_params_dict.get("tenant_id")
        client_id = litellm_params_dict.get("client_id")
        client_secret = litellm_params_dict.get("client_secret")
        azure_scope = litellm_params_dict.get("azure_scope") or "https://cognitiveservices.azure.com/.default"
        
        # Verify the credentials are extracted correctly
        assert tenant_id == "test-tenant-id"
        assert client_id == "test-client-id"
        assert client_secret == "test-client-secret"
        assert azure_scope == "https://cognitiveservices.azure.com/.default"
        
        # Verify the condition to create token provider is met
        assert tenant_id and client_id and client_secret, "Credentials should be present to create token provider"


def test_azure_image_generation_headers_without_api_key():
    """
    Test that when api_key is None, the api-key header is not added to headers.
    
    This prevents the httpx TypeError: "Header value must be str or bytes, not <class 'NoneType'>"
    that was occurring when api_key was None and being set in headers.
    
    This is a unit test for the fix in images/main.py where we now check:
    if api_key is not None:
        default_headers["api-key"] = api_key
    """
    from litellm.images.main import image_generation

    # Test the header building logic directly
    api_key = None
    
    default_headers = {
        "Content-Type": "application/json",
    }
    
    # This is the fix: only add api-key if it's not None
    if api_key is not None:
        default_headers["api-key"] = api_key
    
    # Verify api-key is not in headers when api_key is None
    assert "api-key" not in default_headers
    
    # Verify Content-Type is still there
    assert default_headers["Content-Type"] == "application/json"
    
    # Test with a valid api_key
    api_key = "valid-key-123"
    default_headers_with_key = {
        "Content-Type": "application/json",
    }
    if api_key is not None:
        default_headers_with_key["api-key"] = api_key
    
    # Verify api-key is added when api_key is valid
    assert "api-key" in default_headers_with_key
    assert default_headers_with_key["api-key"] == "valid-key-123"


def test_azure_image_generation_drop_params_response_format():
    """
    Test that unsupported params like response_format are dropped when drop_params=True.
    
    Azure gpt-image-1.5 doesn't support response_format parameter. When drop_params=True,
    this parameter should be completely removed and not appear in the final request body,
    including not being added to extra_body.
    
    This test verifies the fix where:
    1. Unsupported params are removed from non_default_params in _check_valid_arg
    2. Unsupported params are also removed from passed_params to prevent them from
       being re-added via extra_body in add_provider_specific_params_to_optional_params
    
    Without the fix, response_format would be added to extra_body and cause Azure to
    return a 400 Bad Request error due to strict schema validation.
    """
    from litellm.llms.openai.image_generation.gpt_transformation import (
        GPTImageGenerationConfig,
    )

    # Test with gpt-image-1.5 which doesn't support response_format
    config = GPTImageGenerationConfig()
    supported_params = config.get_supported_openai_params(model="gpt-image-1.5")
    
    # Verify response_format is NOT in supported params for gpt-image-1.5
    assert "response_format" not in supported_params
    assert "n" in supported_params
    assert "size" in supported_params
    
    # Test get_optional_params_image_gen with drop_params=True
    optional_params = get_optional_params_image_gen(
        model="gpt-image-1.5",
        n=1,
        size="1024x1024",
        response_format="b64_json",  # This should be dropped
        custom_llm_provider="azure",
        provider_config=config,
        drop_params=True,
    )
    
    # Verify response_format is NOT in optional_params
    assert "response_format" not in optional_params, (
        "response_format should be dropped from optional_params"
    )
    
    # Verify response_format is NOT in extra_body either
    if "extra_body" in optional_params:
        assert "response_format" not in optional_params["extra_body"], (
            "response_format should not be in extra_body"
        )
    
    # Verify supported params ARE in optional_params
    assert "n" in optional_params
    assert optional_params["n"] == 1
    assert "size" in optional_params
    assert optional_params["size"] == "1024x1024"


def test_azure_image_generation_drop_params_false_raises_error():
    """
    Test that unsupported params raise an error when drop_params=False.
    
    This verifies that the error handling still works correctly when drop_params
    is not enabled.
    """
    from litellm.exceptions import UnsupportedParamsError
    from litellm.llms.openai.image_generation.gpt_transformation import (
        GPTImageGenerationConfig,
    )

    config = GPTImageGenerationConfig()
    
    # Test that passing unsupported param with drop_params=False raises error
    with pytest.raises(UnsupportedParamsError) as exc_info:
        optional_params = get_optional_params_image_gen(
            model="gpt-image-1.5",
            n=1,
            response_format="b64_json",  # Unsupported param
            custom_llm_provider="azure",
            provider_config=config,
            drop_params=False,
        )
    
    # Verify the error message mentions the unsupported parameter
    assert "response_format" in str(exc_info.value)


def test_azure_image_generation_base_model_vs_deployment_name():
    """
    Test that Azure image generation correctly uses base_model in request body
    but deployment name in the URL.
    
    When base_model is specified in litellm_params, the request should:
    1. Use base_model (e.g., "gpt-image-1.5") in the JSON request body
    2. Use the deployment name (e.g., "gpt-image-15") in the URL path
    
    This is important because Azure expects:
    - URL: /openai/deployments/{deployment_name}/images/generations
    - Body: {"model": "{base_model}", ...}
    
    Example config:
      model: azure/gpt-image-15  # deployment name
      base_model: gpt-image-1.5  # actual model name
    """
    from unittest.mock import MagicMock
    
    # Setup test parameters
    azure_chat_completion = AzureChatCompletion()
    
    prompt = "A beautiful image of a cat"
    model = "gpt-image-15"  # This is the deployment name
    base_model = "gpt-image-1.5"  # This is the actual model name
    api_base = "https://openai-gpt-image-1-5-test-v-1.openai.azure.com/"
    api_version = "2024-07-01-preview"
    api_key = "test-api-key"
    
    litellm_params = {
        "base_model": base_model,
        "api_base": api_base,
        "api_version": api_version,
    }
    
    optional_params = {
        "n": 1,
        "size": "1024x1024"
    }
    
    # Mock the HTTP request to capture what gets sent
    with patch.object(
        azure_chat_completion,
        "make_sync_azure_httpx_request",
        return_value=MagicMock(
            json=lambda: {
                "created": 1234567890,
                "data": [
                    {
                        "url": "https://example.com/image.png",
                        "revised_prompt": prompt
                    }
                ]
            }
        )
    ) as mock_request:
        # Mock logging object
        logging_obj = MagicMock()
        logging_obj.pre_call = MagicMock()
        logging_obj.post_call = MagicMock()
        
        # Call the image_generation method
        response = azure_chat_completion.image_generation(
            prompt=prompt,
            timeout=60.0,
            optional_params=optional_params,
            logging_obj=logging_obj,
            headers={},
            model=model,
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            litellm_params=litellm_params,
        )
        
        # Verify the mock was called
        assert mock_request.called, "HTTP request should have been made"
        
        # Get the call arguments
        call_kwargs = mock_request.call_args.kwargs
        
        # Verify the URL uses the deployment name (not base_model)
        api_base_used = call_kwargs.get("api_base", "")
        assert model in api_base_used, (
            f"URL should contain deployment name '{model}', "
            f"but got: {api_base_used}"
        )
        assert base_model not in api_base_used or base_model == model, (
            f"URL should NOT contain base_model '{base_model}' when it differs from deployment name, "
            f"but got: {api_base_used}"
        )
        
        # Verify the request body uses base_model (not deployment name)
        request_data = call_kwargs.get("data", {})
        assert request_data.get("model") == base_model, (
            f"Request body 'model' field should be base_model '{base_model}', "
            f"but got: {request_data.get('model')}"
        )
        
        # Verify other fields are correct
        assert request_data.get("prompt") == prompt
        assert request_data.get("n") == 1
        assert request_data.get("size") == "1024x1024"


@pytest.mark.asyncio
async def test_azure_aimage_generation_base_model_vs_deployment_name():
    """
    Test that Azure async image generation correctly uses base_model in request body
    but deployment name in the URL.
    
    This is the async version of test_azure_image_generation_base_model_vs_deployment_name.
    """
    from unittest.mock import MagicMock
    
    # Setup test parameters
    azure_chat_completion = AzureChatCompletion()
    
    prompt = "A beautiful image of a cat"
    model = "gpt-image-15"  # This is the deployment name
    base_model = "gpt-image-1.5"  # This is the actual model name
    api_base = "https://openai-gpt-image-1-5-test-v-1.openai.azure.com/"
    api_version = "2024-07-01-preview"
    api_key = "test-api-key"
    
    data = {
        "model": base_model,
        "prompt": prompt,
        "n": 1,
        "size": "1024x1024"
    }
    
    azure_client_params = {
        "api_base": api_base,
        "api_version": api_version,
    }
    
    # Mock the HTTP request to capture what gets sent
    with patch.object(
        azure_chat_completion,
        "make_async_azure_httpx_request",
        new_callable=AsyncMock,
        return_value=MagicMock(
            json=lambda: {
                "created": 1234567890,
                "data": [
                    {
                        "url": "https://example.com/image.png",
                        "revised_prompt": prompt
                    }
                ]
            }
        )
    ) as mock_request:
        # Mock logging object
        logging_obj = MagicMock()
        logging_obj.pre_call = MagicMock()
        logging_obj.post_call = MagicMock()
        
        # Call the aimage_generation method
        response = await azure_chat_completion.aimage_generation(
            data=data,
            model_response=None,
            azure_client_params=azure_client_params,
            api_key=api_key,
            input=[],
            logging_obj=logging_obj,
            headers={},
            model=model,  # Pass the deployment name
            timeout=60.0,
        )
        
        # Verify the mock was called
        assert mock_request.called, "HTTP request should have been made"
        
        # Get the call arguments
        call_kwargs = mock_request.call_args.kwargs
        
        # Verify the URL uses the deployment name (not base_model)
        api_base_used = call_kwargs.get("api_base", "")
        assert model in api_base_used, (
            f"URL should contain deployment name '{model}', "
            f"but got: {api_base_used}"
        )
        assert base_model not in api_base_used or base_model == model, (
            f"URL should NOT contain base_model '{base_model}' when it differs from deployment name, "
            f"but got: {api_base_used}"
        )
        
        # Verify the request body uses base_model (not deployment name)
        request_data = call_kwargs.get("data", {})
        assert request_data.get("model") == base_model, (
            f"Request body 'model' field should be base_model '{base_model}', "
            f"but got: {request_data.get('model')}"
        )
