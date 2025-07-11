import sys
import os
import io, asyncio
import pytest
import json
from unittest.mock import MagicMock, AsyncMock, patch, Mock

sys.path.insert(0, os.path.abspath("../.."))

import litellm
import litellm.types.utils
from litellm.proxy.guardrails.guardrail_hooks.model_armor import ModelArmorGuardrail
from litellm.proxy._types import UserAPIKeyAuth
from litellm.caching import DualCache
from litellm.types.guardrails import GuardrailEventHooks
from fastapi import HTTPException


@pytest.mark.asyncio
async def test_model_armor_pre_call_hook_sanitization():
    """Test Model Armor pre-call hook with content sanitization"""
    mock_user_api_key_dict = UserAPIKeyAuth()
    mock_cache = MagicMock(spec=DualCache)
    
    guardrail = ModelArmorGuardrail(
        template_id="test-template",
        project_id="test-project",
        location="us-central1",
        guardrail_name="model-armor-test",
        mask_request_content=True,
    )
    
    # Mock the Model Armor API response
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json = AsyncMock(return_value={
        "sanitized_text": "Hello, my phone number is [REDACTED]",
        "action": "SANITIZE"
    })
    
    # Mock the access token method
    guardrail._ensure_access_token_async = AsyncMock(return_value=("test-token", "test-project"))
    
    # Mock the async handler
    guardrail.async_handler = AsyncMock()
    guardrail.async_handler.post = AsyncMock(return_value=mock_response)
    
    request_data = {
        "model": "gpt-4",
        "messages": [
            {"role": "user", "content": "Hello, my phone number is +1 412 555 1212"}
        ],
        "metadata": {"guardrails": ["model-armor-test"]}
    }
    
    result = await guardrail.async_pre_call_hook(
        user_api_key_dict=mock_user_api_key_dict,
        cache=mock_cache,
        data=request_data,
        call_type="completion"
    )
    
    # Assert the message was sanitized
    assert result["messages"][0]["content"] == "Hello, my phone number is [REDACTED]"
    
    # Verify API was called correctly
    guardrail.async_handler.post.assert_called_once()
    call_args = guardrail.async_handler.post.call_args
    assert "sanitizeUserPrompt" in call_args[1]["url"]
    assert call_args[1]["json"]["user_prompt_data"]["text"] == "Hello, my phone number is +1 412 555 1212"


@pytest.mark.asyncio
async def test_model_armor_pre_call_hook_blocked():
    """Test Model Armor pre-call hook when content is blocked"""
    mock_user_api_key_dict = UserAPIKeyAuth()
    mock_cache = MagicMock(spec=DualCache)
    
    guardrail = ModelArmorGuardrail(
        template_id="test-template",
        project_id="test-project",
        location="us-central1",
        guardrail_name="model-armor-test",
    )
    
    # Mock the Model Armor API response for blocked content
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json = AsyncMock(return_value={
        "action": "BLOCK",
        "blocked": True,
        "reason": "Prohibited content detected"
    })
    
    # Mock the access token method
    guardrail._ensure_access_token_async = AsyncMock(return_value=("test-token", "test-project"))
    
    # Mock the async handler
    guardrail.async_handler = AsyncMock()
    guardrail.async_handler.post = AsyncMock(return_value=mock_response)
    
    request_data = {
        "model": "gpt-4",
        "messages": [
            {"role": "user", "content": "Some harmful content"}
        ],
        "metadata": {"guardrails": ["model-armor-test"]}
    }
    
    # Should raise HTTPException for blocked content
    with pytest.raises(HTTPException) as exc_info:
        await guardrail.async_pre_call_hook(
            user_api_key_dict=mock_user_api_key_dict,
            cache=mock_cache,
            data=request_data,
            call_type="completion"
        )
    
    assert exc_info.value.status_code == 400
    assert "Content blocked by Model Armor" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_model_armor_post_call_hook_sanitization():
    """Test Model Armor post-call hook with response sanitization"""
    mock_user_api_key_dict = UserAPIKeyAuth()
    
    guardrail = ModelArmorGuardrail(
        template_id="test-template",
        project_id="test-project",
        location="us-central1",
        guardrail_name="model-armor-test",
        mask_response_content=True,
    )
    
    # Mock the Model Armor API response
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json = AsyncMock(return_value={
        "sanitized_text": "Here is the information: [REDACTED]",
        "action": "SANITIZE"
    })
    
    # Mock the access token method
    guardrail._ensure_access_token_async = AsyncMock(return_value=("test-token", "test-project"))
    
    # Mock the async handler
    guardrail.async_handler = AsyncMock()
    guardrail.async_handler.post = AsyncMock(return_value=mock_response)
    
    # Create a mock response
    mock_llm_response = litellm.ModelResponse()
    mock_llm_response.choices = [
        litellm.Choices(
            message=litellm.Message(
                content="Here is the information: Credit card 1234-5678-9012-3456"
            )
        )
    ]
    
    request_data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "What's my credit card?"}],
        "metadata": {"guardrails": ["model-armor-test"]}
    }
    
    await guardrail.async_post_call_success_hook(
        data=request_data,
        user_api_key_dict=mock_user_api_key_dict,
        response=mock_llm_response
    )
    
    # Assert the response was sanitized
    assert mock_llm_response.choices[0].message.content == "Here is the information: [REDACTED]"
    
    # Verify API was called correctly
    guardrail.async_handler.post.assert_called_once()
    call_args = guardrail.async_handler.post.call_args
    assert "sanitizeModelResponse" in call_args[1]["url"]


@pytest.mark.asyncio
async def test_model_armor_with_list_content():
    """Test Model Armor with messages containing list content"""
    mock_user_api_key_dict = UserAPIKeyAuth()
    mock_cache = MagicMock(spec=DualCache)
    
    guardrail = ModelArmorGuardrail(
        template_id="test-template",
        project_id="test-project",
        location="us-central1",
        guardrail_name="model-armor-test",
    )
    
    # Mock the Model Armor API response
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json = AsyncMock(return_value={
        "action": "NONE"
    })
    
    # Mock the access token method
    guardrail._ensure_access_token_async = AsyncMock(return_value=("test-token", "test-project"))
    
    # Mock the async handler
    guardrail.async_handler = AsyncMock()
    guardrail.async_handler.post = AsyncMock(return_value=mock_response)
    
    request_data = {
        "model": "gpt-4",
        "messages": [
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": "Hello world"},
                    {"type": "text", "text": "How are you?"}
                ]
            }
        ],
        "metadata": {"guardrails": ["model-armor-test"]}
    }
    
    result = await guardrail.async_pre_call_hook(
        user_api_key_dict=mock_user_api_key_dict,
        cache=mock_cache,
        data=request_data,
        call_type="completion"
    )
    
    # Verify the content was extracted correctly
    guardrail.async_handler.post.assert_called_once()
    call_args = guardrail.async_handler.post.call_args
    assert call_args[1]["json"]["user_prompt_data"]["text"] == "Hello world\nHow are you?"


@pytest.mark.asyncio
async def test_model_armor_api_error_handling():
    """Test Model Armor error handling when API returns error"""
    mock_user_api_key_dict = UserAPIKeyAuth()
    mock_cache = MagicMock(spec=DualCache)
    
    guardrail = ModelArmorGuardrail(
        template_id="test-template",
        project_id="test-project",
        location="us-central1",
        guardrail_name="model-armor-test",
        fail_on_error=True,
    )
    
    # Mock the Model Armor API error response
    mock_response = AsyncMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"
    
    # Mock the access token method
    guardrail._ensure_access_token_async = AsyncMock(return_value=("test-token", "test-project"))
    
    # Mock the async handler
    guardrail.async_handler = AsyncMock()
    guardrail.async_handler.post = AsyncMock(return_value=mock_response)
    
    request_data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}],
        "metadata": {"guardrails": ["model-armor-test"]}
    }
    
    # Should raise HTTPException for API error
    with pytest.raises(HTTPException) as exc_info:
        await guardrail.async_pre_call_hook(
            user_api_key_dict=mock_user_api_key_dict,
            cache=mock_cache,
            data=request_data,
            call_type="completion"
        )
    
    assert exc_info.value.status_code == 500
    assert "Model Armor API error" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_model_armor_credentials_handling():
    """Test Model Armor handling of different credential types"""
    try:
        from google.auth.credentials import Credentials
    except ImportError:
        # If google.auth is not installed, skip this test
        pytest.skip("google.auth not installed")
        return
    
    # Test with string credentials (file path)
    with patch('os.path.exists', return_value=True):
        with patch('builtins.open', mock_open(read_data='{"type": "service_account", "project_id": "test-project"}')):
            with patch.object(ModelArmorGuardrail, '_credentials_from_service_account') as mock_creds:
                mock_creds_obj = Mock()
                mock_creds_obj.token = "test-token"
                mock_creds_obj.expired = False
                mock_creds_obj.project_id = "test-project"  # Add project_id
                mock_creds.return_value = mock_creds_obj
                
                guardrail = ModelArmorGuardrail(
                    template_id="test-template",
                    credentials="/path/to/creds.json",
                    project_id="test-project",  # Provide project_id
                )
                
                # Force credential loading
                creds, project_id = guardrail.load_auth(credentials="/path/to/creds.json", project_id="test-project")
                
                assert mock_creds.called
                assert project_id == "test-project"


@pytest.mark.asyncio
async def test_model_armor_streaming_response():
    """Test Model Armor with streaming responses"""
    mock_user_api_key_dict = UserAPIKeyAuth()
    
    guardrail = ModelArmorGuardrail(
        template_id="test-template",
        project_id="test-project",
        location="us-central1",
        guardrail_name="model-armor-test",
        mask_response_content=True,
    )
    
    # Mock the Model Armor API response
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json = AsyncMock(return_value={
        "sanitized_text": "Sanitized response",
        "action": "SANITIZE"
    })
    
    # Mock the access token method
    guardrail._ensure_access_token_async = AsyncMock(return_value=("test-token", "test-project"))
    
    # Mock the async handler
    guardrail.async_handler = AsyncMock()
    guardrail.async_handler.post = AsyncMock(return_value=mock_response)
    
    # Create mock streaming chunks
    async def mock_stream():
        chunks = [
            litellm.ModelResponseStream(
                choices=[
                    litellm.types.utils.StreamingChoices(
                        delta=litellm.types.utils.Delta(content="Sensitive ")
                    )
                ]
            ),
            litellm.ModelResponseStream(
                choices=[
                    litellm.types.utils.StreamingChoices(
                        delta=litellm.types.utils.Delta(content="information")
                    )
                ]
            ),
        ]
        for chunk in chunks:
            yield chunk
    
    request_data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Tell me secrets"}],
        "metadata": {"guardrails": ["model-armor-test"]}
    }
    
    # Process streaming response
    result_chunks = []
    async for chunk in guardrail.async_post_call_streaming_iterator_hook(
        user_api_key_dict=mock_user_api_key_dict,
        response=mock_stream(),
        request_data=request_data
    ):
        result_chunks.append(chunk)
    
    # Should have processed the chunks through Model Armor
    assert len(result_chunks) > 0
    guardrail.async_handler.post.assert_called()


def mock_open(read_data=''):
    """Helper to create a mock file object"""
    import io
    from unittest.mock import MagicMock
    
    file_object = io.StringIO(read_data)
    file_object.__enter__ = lambda self: self
    file_object.__exit__ = lambda self, *args: None
    
    mock_file = MagicMock(return_value=file_object)
    return mock_file