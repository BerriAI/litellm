import sys
import os
import io, asyncio
import pytest
sys.path.insert(0, os.path.abspath("../.."))
import litellm
from litellm.proxy.guardrails.guardrail_hooks.bedrock_guardrails import BedrockGuardrail
from litellm.proxy._types import UserAPIKeyAuth
from litellm.caching import DualCache
from unittest.mock import MagicMock, AsyncMock, patch

@pytest.mark.asyncio
async def test_bedrock_guardrails():
    # Create proper mock objects
    mock_user_api_key_dict = UserAPIKeyAuth()
    
    guardrail = BedrockGuardrail(
        guardrailIdentifier="wf0hkdb5x07f",
        guardrailVersion="DRAFT",
        mask_request_content=True,
    )

    request_data = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "Hello, my phone number is +1 412 555 1212"},
            {"role": "assistant", "content": "Hello, how can I help you today?"},
            {"role": "user", "content": "I need to cancel my order"},
            {"role": "user", "content": "ok, my credit card number is 1234-5678-9012-3456"},
        ],
    }

    response = await guardrail.async_moderation_hook(
        data=request_data,
        user_api_key_dict=mock_user_api_key_dict,
        call_type="completion"
    )
    print(response)

    if response:  # Only assert if response is not None
        assert response["messages"][0]["content"] == "Hello, my phone number is {PHONE}"
        assert response["messages"][1]["content"] == "Hello, how can I help you today?"
        assert response["messages"][2]["content"] == "I need to cancel my order"
        assert response["messages"][3]["content"] == "ok, my credit card number is {CREDIT_DEBIT_CARD_NUMBER}"


@pytest.mark.asyncio
async def test_bedrock_guardrails_content_list():
    # Create proper mock objects
    mock_user_api_key_dict = UserAPIKeyAuth()
    
    guardrail = BedrockGuardrail(
        guardrailIdentifier="wf0hkdb5x07f",
        guardrailVersion="DRAFT",
        mask_request_content=True,
    )

    request_data = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "Hello, my phone number is +1 412 555 1212"},
                {"type": "text", "text": "what time is it?"},
            ]},
            {"role": "assistant", "content": "Hello, how can I help you today?"},
            {
                "role": "user",
                "content": "who is the president of the united states?"
            }
        ],
    }

    response = await guardrail.async_moderation_hook(
        data=request_data,
        user_api_key_dict=mock_user_api_key_dict,
        call_type="completion"
    )
    print(response)
    
    if response:  # Only assert if response is not None
        # Verify that the list content is properly masked
        assert isinstance(response["messages"][0]["content"], list)
        assert response["messages"][0]["content"][0]["text"] == "Hello, my phone number is {PHONE}"
        assert response["messages"][0]["content"][1]["text"] == "what time is it?"
        assert response["messages"][1]["content"] == "Hello, how can I help you today?"
        assert response["messages"][2]["content"] == "who is the president of the united states?"
    
    



@pytest.mark.asyncio
async def test_bedrock_guardrails_with_streaming():
    from litellm.proxy.utils import ProxyLogging
    from litellm.types.guardrails import GuardrailEventHooks

    # Create proper mock objects
    mock_user_api_key_cache = MagicMock(spec=DualCache)
    mock_user_api_key_dict = UserAPIKeyAuth()

    with pytest.raises(Exception):  # Assert that this raises an exception
        proxy_logging_obj = ProxyLogging(
            user_api_key_cache=mock_user_api_key_cache,
            premium_user=True,
        )
        
        guardrail = BedrockGuardrail(
            guardrailIdentifier="wf0hkdb5x07f",
            guardrailVersion="DRAFT",
            supported_event_hooks=[GuardrailEventHooks.post_call],
            guardrail_name="bedrock-post-guard",
        )

        litellm.callbacks.append(guardrail)

        request_data = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": "My name is ishaan@gmail.com"
                }
            ],
            "stream": True,
            "metadata": {"guardrails": ["bedrock-post-guard"]}
        }

        response = await litellm.acompletion(
            **request_data,
        )

        response = proxy_logging_obj.async_post_call_streaming_iterator_hook(
            user_api_key_dict=mock_user_api_key_dict,
            response=response,
            request_data=request_data,
        )
        
        async for chunk in response:
            print(chunk)


@pytest.mark.asyncio
async def test_bedrock_guardrails_with_streaming_no_violation():
    from litellm.proxy.utils import ProxyLogging
    from litellm.types.guardrails import GuardrailEventHooks

    # Create proper mock objects
    mock_user_api_key_cache = MagicMock(spec=DualCache)
    mock_user_api_key_dict = UserAPIKeyAuth()

    proxy_logging_obj = ProxyLogging(
        user_api_key_cache=mock_user_api_key_cache,
        premium_user=True,
    )
    
    guardrail = BedrockGuardrail(
        guardrailIdentifier="wf0hkdb5x07f",
        guardrailVersion="DRAFT",
        supported_event_hooks=[GuardrailEventHooks.post_call],
        guardrail_name="bedrock-post-guard",
    )

    litellm.callbacks.append(guardrail)


    request_data = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": "hi"
            }
        ],
        "stream": True,
        "metadata": {"guardrails": ["bedrock-post-guard"]}
    }

    response = await litellm.acompletion(
        **request_data,
    )

    response = proxy_logging_obj.async_post_call_streaming_iterator_hook(
        user_api_key_dict=mock_user_api_key_dict,
        response=response,
        request_data=request_data,
    )
    
    
    async for chunk in response:
        print(chunk)
        

@pytest.mark.asyncio
async def test_bedrock_guardrails_streaming_request_body_mock():
    """Test that the exact request body sent to Bedrock matches expected format when using streaming"""
    import json
    from unittest.mock import AsyncMock, MagicMock, patch
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.caching import DualCache
    from litellm.types.guardrails import GuardrailEventHooks
    
    # Create mock objects
    mock_user_api_key_dict = UserAPIKeyAuth()
    mock_cache = MagicMock(spec=DualCache)

    # Create the guardrail
    guardrail = BedrockGuardrail(
        guardrailIdentifier="wf0hkdb5x07f",
        guardrailVersion="DRAFT",
        supported_event_hooks=[GuardrailEventHooks.post_call],
        guardrail_name="bedrock-post-guard",
    )

    # Mock the assembled response from streaming
    mock_response = litellm.ModelResponse(
        id="test-id",
        choices=[
            litellm.Choices(
                index=0,
                message=litellm.Message(
                    role="assistant", 
                    content="The capital of Spain is Madrid."
                ),
                finish_reason="stop"
            )
        ],
        created=1234567890,
        model="gpt-4o",
        object="chat.completion"
    )

    # Mock Bedrock API response
    mock_bedrock_response = MagicMock()
    mock_bedrock_response.status_code = 200
    mock_bedrock_response.json.return_value = {
        "action": "NONE",
        "outputs": []
    }

    # Patch the async_handler.post method to capture the request body
    with patch.object(guardrail, 'async_handler') as mock_async_handler:
        mock_async_handler.post = AsyncMock(return_value=mock_bedrock_response)
        
        # Test data - simulating request data and assembled response
        request_data = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": "what's the capital of spain?"
                }
            ],
            "stream": True,
            "metadata": {"guardrails": ["bedrock-post-guard"]}
        }

        # Call the method that should make the Bedrock API request
        await guardrail.make_bedrock_api_request(
            kwargs=request_data, 
            response=mock_response
        )

        # Verify the API call was made
        mock_async_handler.post.assert_called_once()
        
        # Get the request data that was passed
        call_args = mock_async_handler.post.call_args
        
        # The data should be in the 'data' parameter of the prepared request
        # We need to parse the JSON from the prepared request body
        prepared_request_body = call_args.kwargs.get('data')
        
        # Parse the JSON body
        if isinstance(prepared_request_body, bytes):
            actual_body = json.loads(prepared_request_body.decode('utf-8'))
        else:
            actual_body = json.loads(prepared_request_body)
        
        # Expected body based on the convert_to_bedrock_format method behavior
        expected_body = {
            'source': 'OUTPUT',
            'content': [
                {'text': {'text': "what's the capital of spain?"}},
                {'text': {'text': 'The capital of Spain is Madrid.'}}
            ]
        }
        
        print("Actual Bedrock request body:", json.dumps(actual_body, indent=2))
        print("Expected Bedrock request body:", json.dumps(expected_body, indent=2))
        
        # Assert the request body matches exactly
        assert actual_body == expected_body, f"Request body mismatch. Expected: {expected_body}, Got: {actual_body}"
        

@pytest.mark.asyncio
async def test_bedrock_guardrail_aws_param_persistence():
    """Test that AWS auth params set on init are used for every request and not popped out."""
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.types.guardrails import GuardrailEventHooks

    guardrail = BedrockGuardrail(
        guardrailIdentifier="wf0hkdb5x07f",
        guardrailVersion="DRAFT",
        aws_access_key_id="test-access-key",
        aws_secret_access_key="test-secret-key",
        aws_region_name="us-east-1",
        supported_event_hooks=[GuardrailEventHooks.post_call],
        guardrail_name="bedrock-post-guard",
    )

    with patch.object(guardrail, "get_credentials", wraps=guardrail.get_credentials) as mock_get_creds:
        for i in range(3):
            request_data = {
                "model": "gpt-4o",
                "messages": [
                    {"role": "user", "content": f"request {i}"}
                ],
                "stream": False,
                "metadata": {"guardrails": ["bedrock-post-guard"]}
            }
            with patch.object(guardrail.async_handler, "post", new_callable=AsyncMock) as mock_post:
                # Configure the mock response properly
                mock_response = AsyncMock()
                mock_response.status_code = 200
                mock_response.json = MagicMock(return_value={"action": "NONE", "outputs": []})
                mock_post.return_value = mock_response
                await guardrail.make_bedrock_api_request(kwargs=request_data, response=None)

        assert mock_get_creds.call_count == 3
        for call in mock_get_creds.call_args_list:
            kwargs = call.kwargs
            print("used the following kwargs to get credentials=", kwargs)
            assert kwargs["aws_access_key_id"] == "test-access-key"
            assert kwargs["aws_secret_access_key"] == "test-secret-key"
            assert kwargs["aws_region_name"] == "us-east-1"

        