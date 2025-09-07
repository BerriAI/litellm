from unittest.mock import AsyncMock

import pytest

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy._types import CallTypes, UserAPIKeyAuth


class TestCustomGuardrailDeploymentHook:

    @pytest.mark.asyncio
    async def test_async_pre_call_deployment_hook_no_guardrails(self):
        """Test that method returns kwargs unchanged when no guardrails are present"""
        custom_guardrail = CustomGuardrail()

        # Test with guardrails as None
        kwargs = {
            "messages": [{"role": "user", "content": "test message"}],
            "model": "gpt-3.5-turbo",
            "guardrails": None,
        }

        result = await custom_guardrail.async_pre_call_deployment_hook(
            kwargs=kwargs, call_type=CallTypes.completion
        )

        assert result == kwargs

        # Test with guardrails as non-list
        kwargs["guardrails"] = "not_a_list"

        result = await custom_guardrail.async_pre_call_deployment_hook(
            kwargs=kwargs, call_type=CallTypes.completion
        )

        assert result == kwargs

    @pytest.mark.asyncio
    async def test_async_pre_call_deployment_hook_with_guardrails_and_message_update(
        self,
    ):
        """Test that method processes guardrails and updates messages when result contains messages"""
        custom_guardrail = CustomGuardrail()

        # Mock the async_pre_call_hook method
        mock_result = {"messages": [{"role": "user", "content": "filtered message"}]}
        custom_guardrail.async_pre_call_hook = AsyncMock(return_value=mock_result)

        original_messages = [{"role": "user", "content": "original message"}]
        kwargs = {
            "messages": original_messages,
            "model": "gpt-3.5-turbo",
            "guardrails": ["some_guardrail"],
            "user_api_key_user_id": "test_user",
            "user_api_key_team_id": "test_team",
            "user_api_key_end_user_id": "test_end_user",
            "user_api_key_hash": "test_hash",
            "user_api_key_request_route": "test_route",
        }

        result = await custom_guardrail.async_pre_call_deployment_hook(
            kwargs=kwargs, call_type=CallTypes.completion
        )

        # Verify async_pre_call_hook was called with correct parameters
        custom_guardrail.async_pre_call_hook.assert_called_once()
        call_args = custom_guardrail.async_pre_call_hook.call_args

        # Check that UserAPIKeyAuth was created properly
        user_api_key_dict = call_args[1]["user_api_key_dict"]
        assert isinstance(user_api_key_dict, UserAPIKeyAuth)
        assert user_api_key_dict.user_id == "test_user"
        assert user_api_key_dict.team_id == "test_team"
        assert user_api_key_dict.end_user_id == "test_end_user"
        assert user_api_key_dict.api_key == "test_hash"
        assert user_api_key_dict.request_route == "test_route"

        # Check other parameters
        assert call_args[1]["data"] == kwargs
        assert call_args[1]["call_type"] == "completion"

        # Verify messages were updated in result
        assert result["messages"] == mock_result["messages"]
        assert result["messages"] != original_messages


class TestCustomGuardrailShouldRunGuardrail:

    def test_should_run_guardrail_with_litellm_metadata(self):
        """Test that should_run_guardrail works with litellm_metadata pattern"""
        from litellm.types.guardrails import GuardrailEventHooks
        
        custom_guardrail = CustomGuardrail(
            guardrail_name="test_guardrail", 
            default_on=False,
            event_hook=GuardrailEventHooks.pre_call
        )
        
        # Test with guardrails in litellm_metadata
        data = {
            "model": "gpt-3.5-turbo",
            "litellm_metadata": {
                "guardrails": ["test_guardrail"]
            }
        }
        
        result = custom_guardrail.should_run_guardrail(
            data=data, event_type=GuardrailEventHooks.pre_call
        )
        
        assert result is True
        
    def test_should_run_guardrail_with_metadata(self):
        """Test that should_run_guardrail works with metadata pattern"""
        from litellm.types.guardrails import GuardrailEventHooks
        
        custom_guardrail = CustomGuardrail(
            guardrail_name="test_guardrail", 
            default_on=False,
            event_hook=GuardrailEventHooks.pre_call
        )
        
        # Test with guardrails in metadata
        data = {
            "model": "gpt-3.5-turbo",
            "metadata": {
                "guardrails": ["test_guardrail"]
            }
        }
        
        result = custom_guardrail.should_run_guardrail(
            data=data, event_type=GuardrailEventHooks.pre_call
        )
        
        assert result is True
        
    def test_should_run_guardrail_with_root_level_guardrails(self):
        """Test that should_run_guardrail works with root level guardrails"""
        from litellm.types.guardrails import GuardrailEventHooks
        
        custom_guardrail = CustomGuardrail(
            guardrail_name="test_guardrail", 
            default_on=False,
            event_hook=GuardrailEventHooks.pre_call
        )
        
        # Test with guardrails at root level
        data = {
            "model": "gpt-3.5-turbo",
            "guardrails": ["test_guardrail"]
        }
        
        result = custom_guardrail.should_run_guardrail(
            data=data, event_type=GuardrailEventHooks.pre_call
        )
        
        assert result is True
        
        
    def test_should_run_guardrail_no_matching_guardrail(self):
        """Test that should_run_guardrail returns False when guardrail name doesn't match"""
        from litellm.types.guardrails import GuardrailEventHooks
        
        custom_guardrail = CustomGuardrail(
            guardrail_name="test_guardrail", 
            default_on=False,
            event_hook=GuardrailEventHooks.pre_call
        )
        
        # Test with different guardrail name
        data = {
            "model": "gpt-3.5-turbo",
            "litellm_metadata": {
                "guardrails": ["different_guardrail"]
            }
        }
        
        result = custom_guardrail.should_run_guardrail(
            data=data, event_type=GuardrailEventHooks.pre_call
        )
        
        assert result is False
