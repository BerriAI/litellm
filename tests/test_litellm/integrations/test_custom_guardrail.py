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
            event_hook=GuardrailEventHooks.pre_call,
        )

        # Test with guardrails in litellm_metadata
        data = {
            "model": "gpt-3.5-turbo",
            "litellm_metadata": {"guardrails": ["test_guardrail"]},
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
            event_hook=GuardrailEventHooks.pre_call,
        )

        # Test with guardrails in metadata
        data = {
            "model": "gpt-3.5-turbo",
            "metadata": {"guardrails": ["test_guardrail"]},
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
            event_hook=GuardrailEventHooks.pre_call,
        )

        # Test with guardrails at root level
        data = {"model": "gpt-3.5-turbo", "guardrails": ["test_guardrail"]}

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
            event_hook=GuardrailEventHooks.pre_call,
        )

        # Test with different guardrail name
        data = {
            "model": "gpt-3.5-turbo",
            "litellm_metadata": {"guardrails": ["different_guardrail"]},
        }

        result = custom_guardrail.should_run_guardrail(
            data=data, event_type=GuardrailEventHooks.pre_call
        )

        assert result is False

    def test_should_run_guardrail_with_disable_global_guardrail(self):
        """Test that disable_global_guardrail disables a global guardrail when set to True"""
        from litellm.types.guardrails import GuardrailEventHooks

        # Create a guardrail with default_on=True (global guardrail)
        custom_guardrail = CustomGuardrail(
            guardrail_name="global_guardrail",
            default_on=True,
            event_hook=GuardrailEventHooks.pre_call,
        )

        # Test 1: Global guardrail runs by default when default_on=True
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "test"}],
        }
        result = custom_guardrail.should_run_guardrail(
            data=data, event_type=GuardrailEventHooks.pre_call
        )
        assert result is True, "Global guardrail should run when default_on=True"

        # Test 2: Global guardrail is disabled when disable_global_guardrail=True at root level
        data_with_disable_root = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "test"}],
            "disable_global_guardrail": True,
        }
        result = custom_guardrail.should_run_guardrail(
            data=data_with_disable_root, event_type=GuardrailEventHooks.pre_call
        )
        assert (
            result is False
        ), "Global guardrail should be disabled when disable_global_guardrail=True"

        # Test 3: Global guardrail is disabled when disable_global_guardrail=True in litellm_metadata
        data_with_disable_litellm = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "test"}],
            "litellm_metadata": {"disable_global_guardrail": True},
        }
        result = custom_guardrail.should_run_guardrail(
            data=data_with_disable_litellm, event_type=GuardrailEventHooks.pre_call
        )
        assert (
            result is False
        ), "Global guardrail should be disabled when disable_global_guardrail=True in litellm_metadata"

        # Test 4: Global guardrail is disabled when disable_global_guardrail=True in metadata
        data_with_disable_metadata = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "test"}],
            "metadata": {"disable_global_guardrail": True},
        }
        result = custom_guardrail.should_run_guardrail(
            data=data_with_disable_metadata, event_type=GuardrailEventHooks.pre_call
        )
        assert (
            result is False
        ), "Global guardrail should be disabled when disable_global_guardrail=True in metadata"

        # Test 5: Global guardrail runs when disable_global_guardrail=False
        data_with_disable_false = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "test"}],
            "disable_global_guardrail": False,
        }
        result = custom_guardrail.should_run_guardrail(
            data=data_with_disable_false, event_type=GuardrailEventHooks.pre_call
        )
        assert (
            result is True
        ), "Global guardrail should still run when disable_global_guardrail=False"


class TestApplyGuardrailCheck:
    def test_apply_guardrail_check_only_on_direct_implementation(self):
        """
        Test that "apply_guardrail" in type(callback).__dict__ only returns True
        when the object's own class implements the method, not when it's inherited
        from a parent class.

        This is critical for properly routing guardrail handling to the unified
        guardrail handler vs the guardrail's own implementation.
        """

        # Parent class with apply_guardrail (CustomGuardrail already has it)
        class ParentGuardrail(CustomGuardrail):
            """Parent that inherits apply_guardrail from CustomGuardrail"""

            pass

        # Child class that only inherits apply_guardrail (doesn't override)
        class ChildGuardrailWithoutOverride(ParentGuardrail):
            """Child that only inherits apply_guardrail"""

            pass

        # Child class that overrides apply_guardrail
        class ChildGuardrailWithOverride(ParentGuardrail):
            """Child that overrides apply_guardrail"""

            async def apply_guardrail(self, text, language=None, entities=None):
                return f"modified: {text}"

        # Instantiate the classes
        parent_instance = ParentGuardrail()
        child_without_override = ChildGuardrailWithoutOverride()
        child_with_override = ChildGuardrailWithOverride()

        # Test: CustomGuardrail itself has apply_guardrail in its __dict__
        assert (
            "apply_guardrail" in type(CustomGuardrail()).__dict__
        ), "CustomGuardrail should have apply_guardrail in its own __dict__"

        # Test: ParentGuardrail inherits but doesn't override, so it should NOT be in __dict__
        assert (
            "apply_guardrail" not in type(parent_instance).__dict__
        ), "ParentGuardrail should NOT have apply_guardrail in its own __dict__ (only inherited)"

        # Test: ChildGuardrailWithoutOverride only inherits, should NOT be in __dict__
        assert (
            "apply_guardrail" not in type(child_without_override).__dict__
        ), "ChildGuardrailWithoutOverride should NOT have apply_guardrail in its own __dict__ (only inherited)"

        # Test: ChildGuardrailWithOverride overrides the method, SHOULD be in __dict__
        assert (
            "apply_guardrail" in type(child_with_override).__dict__
        ), "ChildGuardrailWithOverride SHOULD have apply_guardrail in its own __dict__ (overridden)"

        # Verify that all instances still have the method via inheritance (hasattr)
        assert hasattr(
            parent_instance, "apply_guardrail"
        ), "All instances should have apply_guardrail via inheritance"
        assert hasattr(
            child_without_override, "apply_guardrail"
        ), "All instances should have apply_guardrail via inheritance"
        assert hasattr(
            child_with_override, "apply_guardrail"
        ), "All instances should have apply_guardrail via inheritance"


class TestGuardrailLoggingAggregation:
    def _make_guardrail(self):
        from litellm.types.guardrails import GuardrailEventHooks

        return CustomGuardrail(
            guardrail_name="test_guardrail",
            event_hook=GuardrailEventHooks.pre_call,
        )

    def _invoke_add_log(self, request_data: dict) -> None:
        guardrail = self._make_guardrail()
        guardrail.add_standard_logging_guardrail_information_to_request_data(
            guardrail_json_response={"result": "ok"},
            request_data=request_data,
            guardrail_status="success",
            start_time=1.0,
            end_time=2.0,
            duration=1.0,
            masked_entity_count={"EMAIL": 1},
            guardrail_provider="presidio",
        )

    def test_appends_to_existing_metadata_list(self):
        request_data = {
            "metadata": {
                "standard_logging_guardrail_information": [
                    {"guardrail_name": "existing_guardrail"}
                ]
            }
        }

        self._invoke_add_log(request_data)

        info = request_data["metadata"]["standard_logging_guardrail_information"]
        assert isinstance(info, list)
        assert len(info) == 2
        assert info[0]["guardrail_name"] == "existing_guardrail"
        assert info[1]["guardrail_name"] == "test_guardrail"

    def test_converts_existing_metadata_dict_to_list(self):
        request_data = {
            "metadata": {
                "standard_logging_guardrail_information": {"guardrail_name": "legacy"}
            }
        }

        self._invoke_add_log(request_data)

        info = request_data["metadata"]["standard_logging_guardrail_information"]
        assert isinstance(info, list)
        assert len(info) == 2
        assert info[0]["guardrail_name"] == "legacy"
        assert info[1]["guardrail_name"] == "test_guardrail"

    def test_appends_to_litellm_metadata(self):
        request_data = {
            "litellm_metadata": {
                "standard_logging_guardrail_information": [
                    {"guardrail_name": "litellm_existing"}
                ]
            }
        }

        self._invoke_add_log(request_data)

        info = request_data["litellm_metadata"][
            "standard_logging_guardrail_information"
        ]
        assert isinstance(info, list)
        assert len(info) == 2
        assert info[1]["guardrail_name"] == "test_guardrail"
