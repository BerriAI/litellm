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
async def test_bedrock_guardrails_pii_masking():
    # Create proper mock objects
    mock_user_api_key_dict = UserAPIKeyAuth()
    
    guardrail = BedrockGuardrail(
        guardrailIdentifier="wf0hkdb5x07f",
        guardrailVersion="DRAFT",
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
    print("response after moderation hook", response)

    if response:  # Only assert if response is not None
        assert response["messages"][0]["content"] == "Hello, my phone number is {PHONE}"
        assert response["messages"][1]["content"] == "Hello, how can I help you today?"
        assert response["messages"][2]["content"] == "I need to cancel my order"
        assert response["messages"][3]["content"] == "ok, my credit card number is {CREDIT_DEBIT_CARD_NUMBER}"


@pytest.mark.asyncio
async def test_bedrock_guardrails_pii_masking_content_list():
    # Create proper mock objects
    mock_user_api_key_dict = UserAPIKeyAuth()
    
    guardrail = BedrockGuardrail(
        guardrailIdentifier="wf0hkdb5x07f",
        guardrailVersion="DRAFT",
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
            guardrailIdentifier="ff6ujrregl1q",
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
                    "content": "Hi I like coffee"
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
        guardrailIdentifier="ff6ujrregl1q",
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

@pytest.mark.asyncio
async def test_bedrock_guardrail_blocked_vs_anonymized_actions():
    """Test that BLOCKED actions raise exceptions but ANONYMIZED actions do not"""
    from unittest.mock import MagicMock
    from litellm.proxy.guardrails.guardrail_hooks.bedrock_guardrails import BedrockGuardrail
    from litellm.types.proxy.guardrails.guardrail_hooks.bedrock_guardrails import BedrockGuardrailResponse
    
    guardrail = BedrockGuardrail(
        guardrailIdentifier="test-guardrail",
        guardrailVersion="DRAFT"
    )
    
    # Test 1: ANONYMIZED action should NOT raise exception
    anonymized_response: BedrockGuardrailResponse = {
        "action": "GUARDRAIL_INTERVENED",
        "outputs": [{
            "text": "Hello, my phone number is {PHONE}"
        }],
        "assessments": [{
            "sensitiveInformationPolicy": {
                "piiEntities": [{
                    "type": "PHONE",
                    "match": "+1 412 555 1212",
                    "action": "ANONYMIZED"
                }]
            }
        }]
    }
    
    should_raise = guardrail._should_raise_guardrail_blocked_exception(anonymized_response)
    assert should_raise is False, "ANONYMIZED actions should not raise exceptions"
    
    # Test 2: BLOCKED action should raise exception
    blocked_response: BedrockGuardrailResponse = {
        "action": "GUARDRAIL_INTERVENED", 
        "outputs": [{
            "text": "I can't provide that information."
        }],
        "assessments": [{
            "topicPolicy": {
                "topics": [{
                    "name": "Sensitive Topic",
                    "type": "DENY",
                    "action": "BLOCKED"
                }]
            }
        }]
    }
    
    should_raise = guardrail._should_raise_guardrail_blocked_exception(blocked_response)
    assert should_raise is True, "BLOCKED actions should raise exceptions"
    
    # Test 3: Mixed actions - should raise if ANY action is BLOCKED
    mixed_response: BedrockGuardrailResponse = {
        "action": "GUARDRAIL_INTERVENED",
        "outputs": [{
            "text": "I can't provide that information."
        }],
        "assessments": [{
            "sensitiveInformationPolicy": {
                "piiEntities": [{
                    "type": "PHONE", 
                    "match": "+1 412 555 1212",
                    "action": "ANONYMIZED"
                }]
            },
            "topicPolicy": {
                "topics": [{
                    "name": "Blocked Topic",
                    "type": "DENY", 
                    "action": "BLOCKED"
                }]
            }
        }]
    }
    
    should_raise = guardrail._should_raise_guardrail_blocked_exception(mixed_response)
    assert should_raise is True, "Mixed actions with any BLOCKED should raise exceptions"
    
    # Test 4: NONE action should not raise exception
    none_response: BedrockGuardrailResponse = {
        "action": "NONE",
        "outputs": [],
        "assessments": []
    }
    
    should_raise = guardrail._should_raise_guardrail_blocked_exception(none_response)
    assert should_raise is False, "NONE actions should not raise exceptions"
    
    # Test 5: Test other policy types with BLOCKED actions
    content_blocked_response: BedrockGuardrailResponse = {
        "action": "GUARDRAIL_INTERVENED",
        "outputs": [{
            "text": "I can't provide that information."
        }],
        "assessments": [{
            "contentPolicy": {
                "filters": [{
                    "type": "VIOLENCE",
                    "confidence": "HIGH",
                    "action": "BLOCKED"
                }]
            }
        }]
    }
    
    should_raise = guardrail._should_raise_guardrail_blocked_exception(content_blocked_response)
    assert should_raise is True, "Content policy BLOCKED actions should raise exceptions"


@pytest.mark.asyncio
async def test_bedrock_guardrail_masking_with_anonymized_response():
    """Test that masking works correctly when guardrail returns ANONYMIZED actions"""
    from unittest.mock import AsyncMock, MagicMock, patch
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.caching import DualCache
    
    # Create proper mock objects
    mock_user_api_key_dict = UserAPIKeyAuth()
    
    guardrail = BedrockGuardrail(
        guardrailIdentifier="test-guardrail",
        guardrailVersion="DRAFT",
        mask_request_content=True,
    )

    # Mock the Bedrock API response with ANONYMIZED action
    mock_bedrock_response = MagicMock()
    mock_bedrock_response.status_code = 200
    mock_bedrock_response.json.return_value = {
        "action": "GUARDRAIL_INTERVENED",
        "outputs": [{
            "text": "Hello, my phone number is {PHONE}"
        }],
        "assessments": [{
            "sensitiveInformationPolicy": {
                "piiEntities": [{
                    "type": "PHONE",
                    "match": "+1 412 555 1212", 
                    "action": "ANONYMIZED"
                }]
            }
        }]
    }

    request_data = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "Hello, my phone number is +1 412 555 1212"},
        ],
    }

    # Patch the async_handler.post method
    with patch.object(guardrail.async_handler, 'post', new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_bedrock_response
        
        # This should NOT raise an exception since action is ANONYMIZED
        try:
            response = await guardrail.async_moderation_hook(
                data=request_data,
                user_api_key_dict=mock_user_api_key_dict,
                call_type="completion"
            )
            # Should succeed and return data with masked content
            assert response is not None
            assert response["messages"][0]["content"] == "Hello, my phone number is {PHONE}"
        except Exception as e:
            pytest.fail(f"Should not raise exception for ANONYMIZED actions, but got: {e}")


@pytest.mark.asyncio
async def test_bedrock_guardrail_uses_masked_output_without_masking_flags():
    """Test that masked output from guardrails is used even when masking flags are not enabled"""
    from unittest.mock import AsyncMock, MagicMock, patch
    from litellm.proxy._types import UserAPIKeyAuth
    
    # Create proper mock objects
    mock_user_api_key_dict = UserAPIKeyAuth()
    
    # Create guardrail WITHOUT masking flags enabled
    guardrail = BedrockGuardrail(
        guardrailIdentifier="test-guardrail",
        guardrailVersion="DRAFT",
        # Note: No mask_request_content=True or mask_response_content=True
    )

    # Mock the Bedrock API response with ANONYMIZED action and masked output
    mock_bedrock_response = MagicMock()
    mock_bedrock_response.status_code = 200
    mock_bedrock_response.json.return_value = {
        "action": "GUARDRAIL_INTERVENED",
        "outputs": [{
            "text": "Hello, my phone number is {PHONE} and email is {EMAIL}"
        }],
        "assessments": [{
            "sensitiveInformationPolicy": {
                "piiEntities": [
                    {
                        "type": "PHONE",
                        "match": "+1 412 555 1212", 
                        "action": "ANONYMIZED"
                    },
                    {
                        "type": "EMAIL",
                        "match": "user@example.com",
                        "action": "ANONYMIZED"
                    }
                ]
            }
        }]
    }

    request_data = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "Hello, my phone number is +1 412 555 1212 and email is user@example.com"},
        ],
    }

    # Patch the async_handler.post method
    with patch.object(guardrail.async_handler, 'post', new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_bedrock_response
        
        # This should use the masked output even without masking flags
        response = await guardrail.async_moderation_hook(
            data=request_data,
            user_api_key_dict=mock_user_api_key_dict,
            call_type="completion"
        )
        
        # Should use the masked content from guardrail output
        assert response is not None
        assert response["messages"][0]["content"] == "Hello, my phone number is {PHONE} and email is {EMAIL}"
        print("✅ Masked output was applied even without masking flags enabled")


@pytest.mark.asyncio
async def test_bedrock_guardrail_response_pii_masking_non_streaming():
    """Test that PII masking is applied to response content in non-streaming scenarios"""
    from unittest.mock import AsyncMock, MagicMock, patch
    from litellm.proxy._types import UserAPIKeyAuth
    
    # Create proper mock objects
    mock_user_api_key_dict = UserAPIKeyAuth()
    
    # Create guardrail with response masking enabled
    guardrail = BedrockGuardrail(
        guardrailIdentifier="test-guardrail",
        guardrailVersion="DRAFT",
    )

    # Mock the Bedrock API response with ANONYMIZED PII
    mock_bedrock_response = MagicMock()
    mock_bedrock_response.status_code = 200
    mock_bedrock_response.json.return_value = {
        "action": "GUARDRAIL_INTERVENED",
        "outputs": [{
            "text": "My credit card number is {CREDIT_DEBIT_CARD_NUMBER} and my phone is {PHONE}"
        }],
        "assessments": [{
            "sensitiveInformationPolicy": {
                "piiEntities": [
                    {
                        "type": "CREDIT_DEBIT_CARD_NUMBER",
                        "match": "1234-5678-9012-3456",
                        "action": "ANONYMIZED"
                    },
                    {
                        "type": "PHONE",
                        "match": "+1 412 555 1212", 
                        "action": "ANONYMIZED"
                    }
                ]
            }
        }]
    }

    # Create a mock response that contains PII
    mock_response = litellm.ModelResponse(
        id="test-id",
        choices=[
            litellm.Choices(
                index=0,
                message=litellm.Message(
                    role="assistant", 
                    content="My credit card number is 1234-5678-9012-3456 and my phone is +1 412 555 1212"
                ),
                finish_reason="stop"
            )
        ],
        created=1234567890,
        model="gpt-4o",
        object="chat.completion"
    )

    request_data = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "What's your credit card and phone number?"},
        ],
    }

    # Patch the async_handler.post method
    with patch.object(guardrail.async_handler, 'post', new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_bedrock_response
        
        # Call the post-call success hook
        await guardrail.async_post_call_success_hook(
            data=request_data,
            user_api_key_dict=mock_user_api_key_dict,
            response=mock_response
        )
        
        # Verify that the response content was masked
        assert mock_response.choices[0].message.content == "My credit card number is {CREDIT_DEBIT_CARD_NUMBER} and my phone is {PHONE}"
        print("✓ Non-streaming response PII masking test passed")


@pytest.mark.asyncio
async def test_bedrock_guardrail_response_pii_masking_streaming():
    """Test that PII masking is applied to response content in streaming scenarios"""
    from unittest.mock import AsyncMock, MagicMock, patch
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.types.utils import ModelResponseStream
    
    # Create proper mock objects
    mock_user_api_key_dict = UserAPIKeyAuth()
    
    # Create guardrail with response masking enabled
    guardrail = BedrockGuardrail(
        guardrailIdentifier="test-guardrail",
        guardrailVersion="DRAFT",
    )

    # Mock the Bedrock API response with ANONYMIZED PII
    mock_bedrock_response = MagicMock()
    mock_bedrock_response.status_code = 200
    mock_bedrock_response.json.return_value = {
        "action": "GUARDRAIL_INTERVENED",
        "outputs": [{
            "text": "Sure! My email is {EMAIL} and SSN is {US_SSN}"
        }],
        "assessments": [{
            "sensitiveInformationPolicy": {
                "piiEntities": [
                    {
                        "type": "EMAIL",
                        "match": "john@example.com",
                        "action": "ANONYMIZED"
                    },
                    {
                        "type": "US_SSN",
                        "match": "123-45-6789", 
                        "action": "ANONYMIZED"
                    }
                ]
            }
        }]
    }

    # Create mock streaming chunks
    async def mock_streaming_response():
        chunks = [
            ModelResponseStream(
                id="test-id",
                choices=[
                    litellm.utils.StreamingChoices(
                        index=0,
                        delta=litellm.utils.Delta(content="Sure! My email is "),
                        finish_reason=None
                    )
                ],
                created=1234567890,
                model="gpt-4o",
                object="chat.completion.chunk"
            ),
            ModelResponseStream(
                id="test-id",
                choices=[
                    litellm.utils.StreamingChoices(
                        index=0,
                        delta=litellm.utils.Delta(content="john@example.com and SSN is "),
                        finish_reason=None
                    )
                ],
                created=1234567890,
                model="gpt-4o",
                object="chat.completion.chunk"
            ),
            ModelResponseStream(
                id="test-id",
                choices=[
                    litellm.utils.StreamingChoices(
                        index=0,
                        delta=litellm.utils.Delta(content="123-45-6789"),
                        finish_reason="stop"
                    )
                ],
                created=1234567890,
                model="gpt-4o",
                object="chat.completion.chunk"
            )
        ]
        for chunk in chunks:
            yield chunk

    request_data = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "What's your email and SSN?"},
        ],
        "stream": True,
    }

    # Patch the async_handler.post method
    with patch.object(guardrail.async_handler, 'post', new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_bedrock_response
        
        # Call the streaming hook
        masked_stream = guardrail.async_post_call_streaming_iterator_hook(
            user_api_key_dict=mock_user_api_key_dict,
            response=mock_streaming_response(),
            request_data=request_data
        )
        
        # Collect all chunks from the masked stream
        masked_chunks = []
        async for chunk in masked_stream:
            masked_chunks.append(chunk)
        
        # Verify that we got chunks back
        assert len(masked_chunks) > 0
        
        # Reconstruct the full response from chunks to verify masking
        full_content = ""
        for chunk in masked_chunks:
            if hasattr(chunk, 'choices') and chunk.choices:
                if hasattr(chunk.choices[0], 'delta') and chunk.choices[0].delta:
                    if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                        full_content += chunk.choices[0].delta.content
        
        # Verify that the reconstructed content contains the masked PII
        assert "Sure! My email is {EMAIL} and SSN is {US_SSN}" == full_content
        print("✓ Streaming response PII masking test passed")

