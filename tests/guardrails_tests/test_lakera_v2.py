import sys
import os
import io, asyncio
import pytest
import time
from litellm import mock_completion
from unittest.mock import MagicMock, AsyncMock, patch
sys.path.insert(0, os.path.abspath("../.."))
import litellm
from litellm.proxy.guardrails.guardrail_hooks.lakera_ai_v2 import LakeraAIGuardrail
from litellm.types.guardrails import PiiEntityType, PiiAction
from litellm.proxy._types import UserAPIKeyAuth
from litellm.caching.caching import DualCache
from litellm.exceptions import BlockedPiiEntityError, GuardrailRaisedException
from fastapi import HTTPException
from litellm.types.utils import CallTypes as LitellmCallTypes


@pytest.mark.asyncio
async def test_lakera_pre_call_hook_for_pii_masking():
    """Test for Lakera guardrail pre-call hook for PII masking"""
    # Setup the guardrail with specific entities config
    litellm._turn_on_debug()
    lakera_guardrail = LakeraAIGuardrail(
        api_key=os.environ.get("LAKERA_API_KEY"),
    )
    
    # Create a sample request with PII data
    data = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "My credit card is 4111-1111-1111-1111 and my email is test@example.com. My phone number is 555-123-4567"}
        ],
        "model": "gpt-3.5-turbo",
        "metadata": {}
    }
    
    # Mock objects needed for the pre-call hook
    user_api_key_dict = UserAPIKeyAuth(api_key="test_key")
    cache = DualCache()
    
    # Call the pre-call hook with the specified call type
    modified_data = await lakera_guardrail.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=cache,
        data=data,
        call_type="completion"
    )
    print(modified_data)
    
    # Verify the messages have been modified to mask PII
    assert modified_data["messages"][0]["content"] == "You are a helpful assistant."  # System prompt should be unchanged
    
    user_message = modified_data["messages"][1]["content"]
    assert "4111-1111-1111-1111" not in user_message
    assert "test@example.com" not in user_message


@pytest.mark.asyncio
async def test_lakera_blocks_non_pii_violations():
    """Test that Lakera guardrail blocks requests with non-PII violations like hate speech, violence, etc."""
    
    lakera_guardrail = LakeraAIGuardrail(
        api_key="test_key",
    )
    
    # Mock the call_v2_guard method to return a response similar to the user's example
    mock_response = {
        'payload': [],
        'flagged': True,
        'dev_info': {'git_revision': 'f0bc093a', 'git_timestamp': '2025-09-23T15:28:06+00:00', 'model_version': 'lakera-guard-1', 'version': '2.0.281'},
        'metadata': {'request_uuid': 'b7cd4c8a-28aa-4285-a245-2befee514dbf'},
        'breakdown': [
            {'project_id': 'project-9770817088', 'policy_id': 'policy-lakera-default', 'detector_id': 'detector-lakera-default-moderated-content', 'detector_type': 'moderated_content/crime', 'detected': True, 'message_id': 0},
            {'project_id': 'project-9770817088', 'policy_id': 'policy-lakera-default', 'detector_id': 'detector-lakera-default-moderated-content', 'detector_type': 'moderated_content/hate', 'detected': True, 'message_id': 0},
            {'project_id': 'project-9770817088', 'policy_id': 'policy-lakera-default', 'detector_id': 'detector-lakera-default-moderated-content', 'detector_type': 'moderated_content/violence', 'detected': True, 'message_id': 0},
            {'project_id': 'project-9770817088', 'policy_id': 'policy-lakera-default', 'detector_id': 'detector-lakera-default-prompt-attack', 'detector_type': 'prompt_attack', 'detected': True, 'message_id': 0},
            {'project_id': 'project-9770817088', 'policy_id': 'policy-lakera-default', 'detector_id': 'detector-lakera-default-pii', 'detector_type': 'pii/email', 'detected': False, 'message_id': 0},
        ]
    }
    
    with patch.object(lakera_guardrail, 'call_v2_guard', new_callable=AsyncMock) as mock_call:
        mock_call.return_value = (mock_response, {})
        
        # Create a sample request that would trigger violations
        data = {
            "messages": [
                {"role": "user", "content": "Some harmful content that triggers violations"}
            ],
            "model": "gpt-3.5-turbo",
            "metadata": {}
        }
        
        # Mock objects needed for the pre-call hook
        user_api_key_dict = UserAPIKeyAuth(api_key="test_key")
        cache = DualCache()
        
        # The guardrail should raise an HTTPException for non-PII violations
        with pytest.raises(HTTPException) as exc_info:
            await lakera_guardrail.async_pre_call_hook(
                user_api_key_dict=user_api_key_dict,
                cache=cache,
                data=data,
                call_type="completion"
            )
        
        # Verify the exception details include the Lakera response
        assert exc_info.value.status_code == 400
        assert "Violated guardrail policy" in str(exc_info.value.detail)
        assert "lakera_guardrail_response" in exc_info.value.detail


@pytest.mark.asyncio
async def test_lakera_only_pii_violations_are_masked():
    """Test that Lakera guardrail only masks PII violations and doesn't block the request."""
    
    lakera_guardrail = LakeraAIGuardrail(
        api_key="test_key",
    )
    
    # Mock response with only PII violations
    mock_response = {
        'payload': [
            {'detector_type': 'pii/email', 'start': 10, 'end': 25, 'message_id': 0}
        ],
        'flagged': True,
        'breakdown': [
            {'project_id': 'project-9770817088', 'detector_type': 'pii/email', 'detected': True, 'message_id': 0},
            {'project_id': 'project-9770817088', 'detector_type': 'moderated_content/hate', 'detected': False, 'message_id': 0},
            {'project_id': 'project-9770817088', 'detector_type': 'prompt_attack', 'detected': False, 'message_id': 0},
        ]
    }
    
    with patch.object(lakera_guardrail, 'call_v2_guard', new_callable=AsyncMock) as mock_call:
        mock_call.return_value = (mock_response, {})
        
        data = {
            "messages": [
                {"role": "user", "content": "My email test@example.com here"}
            ],
            "model": "gpt-3.5-turbo",
            "metadata": {}
        }
        
        user_api_key_dict = UserAPIKeyAuth(api_key="test_key")
        cache = DualCache()
        
        # Should not raise an exception, just mask the PII
        result = await lakera_guardrail.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=cache,
            data=data,
            call_type="completion"
        )
        
        # Verify the request was not blocked
        assert result is not None
        assert "messages" in result


@pytest.mark.asyncio
async def test_lakera_blocks_flagged_content_with_user_scenario():
    """
    Test the exact user scenario where Lakera flagged content but request went through.
    This should now be blocked with the fix to check breakdown field instead of payload.
    """
    
    lakera_guardrail = LakeraAIGuardrail(
        api_key="test_key",
    )
    
    # Mock response matching the exact user scenario
    mock_response = {
        'payload': [],  # Empty payload like in user's case
        'flagged': True,
        'dev_info': {'git_revision': 'f0bc093a', 'git_timestamp': '2025-09-23T15:28:06+00:00', 'model_version': 'lakera-guard-1', 'version': '2.0.281'},
        'metadata': {'request_uuid': 'b7cd4c8a-28aa-4285-a245-2befee514dbf'},
        'breakdown': [
            {'project_id': 'project-9770817088', 'policy_id': 'policy-lakera-default', 'detector_id': 'detector-lakera-default-moderated-content', 'detector_type': 'moderated_content/crime', 'detected': True, 'message_id': 0},
            {'project_id': 'project-9770817088', 'policy_id': 'policy-lakera-default', 'detector_id': 'detector-lakera-default-moderated-content', 'detector_type': 'moderated_content/hate', 'detected': True, 'message_id': 0},
            {'project_id': 'project-9770817088', 'policy_id': 'policy-lakera-default', 'detector_id': 'detector-lakera-default-moderated-content', 'detector_type': 'moderated_content/profanity', 'detected': False, 'message_id': 0},
            {'project_id': 'project-9770817088', 'policy_id': 'policy-lakera-default', 'detector_id': 'detector-lakera-default-moderated-content', 'detector_type': 'moderated_content/sexual', 'detected': False, 'message_id': 0},
            {'project_id': 'project-9770817088', 'policy_id': 'policy-lakera-default', 'detector_id': 'detector-lakera-default-moderated-content', 'detector_type': 'moderated_content/violence', 'detected': True, 'message_id': 0},
            {'project_id': 'project-9770817088', 'policy_id': 'policy-lakera-default', 'detector_id': 'detector-lakera-default-moderated-content', 'detector_type': 'moderated_content/weapons', 'detected': True, 'message_id': 0},
            {'project_id': 'project-9770817088', 'policy_id': 'policy-lakera-default', 'detector_id': 'detector-lakera-default-pii', 'detector_type': 'pii/address', 'detected': False, 'message_id': 0},
            {'project_id': 'project-9770817088', 'policy_id': 'policy-lakera-default', 'detector_id': 'detector-lakera-default-pii', 'detector_type': 'pii/credit_card', 'detected': False, 'message_id': 0},
            {'project_id': 'project-9770817088', 'policy_id': 'policy-lakera-default', 'detector_id': 'detector-lakera-default-pii', 'detector_type': 'pii/email', 'detected': False, 'message_id': 0},
            {'project_id': 'project-9770817088', 'policy_id': 'policy-lakera-default', 'detector_id': 'detector-lakera-default-pii', 'detector_type': 'pii/iban_code', 'detected': False, 'message_id': 0},
            {'project_id': 'project-9770817088', 'policy_id': 'policy-lakera-default', 'detector_id': 'detector-lakera-default-pii', 'detector_type': 'pii/ip_address', 'detected': False, 'message_id': 0},
            {'project_id': 'project-9770817088', 'policy_id': 'policy-lakera-default', 'detector_id': 'detector-lakera-default-pii', 'detector_type': 'pii/name', 'detected': False, 'message_id': 0},
            {'project_id': 'project-9770817088', 'policy_id': 'policy-lakera-default', 'detector_id': 'detector-lakera-default-pii', 'detector_type': 'pii/phone_number', 'detected': False, 'message_id': 0},
            {'project_id': 'project-9770817088', 'policy_id': 'policy-lakera-default', 'detector_id': 'detector-lakera-default-pii', 'detector_type': 'pii/us_social_security_number', 'detected': False, 'message_id': 0},
            {'project_id': 'project-9770817088', 'policy_id': 'policy-lakera-default', 'detector_id': 'detector-lakera-default-prompt-attack', 'detector_type': 'prompt_attack', 'detected': True, 'message_id': 0},
            {'project_id': 'project-9770817088', 'policy_id': 'policy-lakera-default', 'detector_id': 'detector-lakera-default-unknown-links', 'detector_type': 'unknown_links', 'detected': False, 'message_id': 0}
        ]
    }
    
    with patch.object(lakera_guardrail, 'call_v2_guard', new_callable=AsyncMock) as mock_call:
        mock_call.return_value = (mock_response, {})
        
        # Create a sample request that would trigger violations
        data = {
            "messages": [
                {"role": "user", "content": "Some harmful content that should be blocked"}
            ],
            "model": "gpt-3.5-turbo",
            "metadata": {}
        }
        
        # Mock objects needed for the pre-call hook
        user_api_key_dict = UserAPIKeyAuth(api_key="test_key")
        cache = DualCache()
        
        # With the fix, this should now raise an HTTPException instead of letting the request through
        with pytest.raises(HTTPException) as exc_info:
            await lakera_guardrail.async_pre_call_hook(
                user_api_key_dict=user_api_key_dict,
                cache=cache,
                data=data,
                call_type="completion"
            )
        
        # Verify the exception details
        assert exc_info.value.status_code == 400
        assert "Violated guardrail policy" in str(exc_info.value.detail)
        assert "lakera_guardrail_response" in exc_info.value.detail
        
        # Verify the full response is included in the exception
        lakera_response = exc_info.value.detail["lakera_guardrail_response"]
        assert lakera_response["flagged"] is True
        assert lakera_response["metadata"]["request_uuid"] == "b7cd4c8a-28aa-4285-a245-2befee514dbf"
        assert len(lakera_response["breakdown"]) == 16  # All the breakdown items from the user's scenario

