from unittest.mock import AsyncMock

import pytest

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy._types import CallTypes, UserAPIKeyAuth
from litellm.types.utils import GuardrailTracingDetail


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


class TestCustomGuardrailPassthroughSupport:
    """Tests for passthrough endpoint guardrail support - Issue fixes."""

    @pytest.mark.asyncio
    async def test_async_post_call_success_deployment_hook_with_httpx_response(self):
        """
        Test that async_post_call_success_deployment_hook handles raw httpx.Response objects
        from passthrough endpoints without crashing with TypeError.
        
        This tests Fix #3: TypeError: TypedDict does not support instance and class checks
        """
        import httpx

        custom_guardrail = CustomGuardrail()
        
        # Mock the async_post_call_success_hook to return None (guardrail didn't modify response)
        custom_guardrail.async_post_call_success_hook = AsyncMock(return_value=None)
        
        # Create a mock httpx.Response object (typical passthrough response)
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.text = "Mock response"
        
        request_data = {
            "guardrails": ["test_guardrail"],
            "user_api_key_user_id": "test_user",
            "user_api_key_team_id": "test_team",
            "user_api_key_end_user_id": "test_end_user",
            "user_api_key_hash": "test_hash",
            "user_api_key_request_route": "passthrough_route",
        }
        
        # This should not raise TypeError: TypedDict does not support instance and class checks
        result = await custom_guardrail.async_post_call_success_deployment_hook(
            request_data=request_data,
            response=mock_response,
            call_type=CallTypes.allm_passthrough_route,
        )
        
        # When result is None, should return the original response
        assert result == mock_response

    @pytest.mark.asyncio
    async def test_async_post_call_success_deployment_hook_with_none_call_type(self):
        """
        Test that async_post_call_success_deployment_hook handles None call_type gracefully.
        
        This ensures that even if call_type is None (before fix #1), the guardrail doesn't crash.
        """
        custom_guardrail = CustomGuardrail()
        
        # Mock the async_post_call_success_hook to return None
        custom_guardrail.async_post_call_success_hook = AsyncMock(return_value=None)
        
        mock_response = AsyncMock()
        
        request_data = {
            "guardrails": ["test_guardrail"],
            "user_api_key_user_id": "test_user",
        }
        
        # Call with None call_type - should not crash
        result = await custom_guardrail.async_post_call_success_deployment_hook(
            request_data=request_data,
            response=mock_response,
            call_type=None,
        )
        
        # Should return the original response when result is None
        assert result == mock_response

    def test_is_valid_response_type_with_none(self):
        """
        Test _is_valid_response_type helper method correctly identifies None as invalid.
        
        This is part of Fix #3: Safely handling TypedDict types that don't support isinstance checks.
        """
        custom_guardrail = CustomGuardrail()
        
        # None should be invalid
        assert custom_guardrail._is_valid_response_type(None) is False

    def test_is_valid_response_type_with_typeddict_error(self):
        """
        Test _is_valid_response_type gracefully handles TypeError from TypedDict.
        
        This tests Fix #3: When isinstance() is called with TypedDict types, it raises TypeError.
        The method should catch this and allow the response through.
        """
        from litellm.types.utils import ModelResponse
        
        custom_guardrail = CustomGuardrail()
        
        # Create a valid LiteLLM response object
        response = ModelResponse(
            id="test-id",
            choices=[],
            created=0,
            model="test-model",
            object="chat.completion",
        )
        
        # This should return True (it's a valid response type or TypeError is caught)
        result = custom_guardrail._is_valid_response_type(response)
        assert result is True



class TestEventTypeLogging:
    """Tests for event_type logging in guardrail information."""

    @pytest.mark.asyncio
    async def test_log_guardrail_information_infers_event_type_from_async_pre_call_hook(
        self,
    ):
        """
        Test that log_guardrail_information decorator correctly infers GuardrailEventHooks.pre_call
        from async_pre_call_hook function name.
        """
        from litellm.integrations.custom_guardrail import log_guardrail_information
        from litellm.types.guardrails import GuardrailEventHooks

        class TestGuardrail(CustomGuardrail):
            def __init__(self):
                super().__init__(
                    guardrail_name="test_event_type_guardrail",
                    event_hook=[
                        GuardrailEventHooks.pre_call,
                        GuardrailEventHooks.post_call,
                    ],
                )

            @log_guardrail_information
            async def async_pre_call_hook(self, data: dict, **kwargs):
                return {"result": "pre_call_executed"}

        guardrail = TestGuardrail()
        request_data = {"metadata": {}}

        await guardrail.async_pre_call_hook(data=request_data)

        # Check that the guardrail_mode was set to pre_call (not the full list)
        logged_info = request_data["metadata"]["standard_logging_guardrail_information"]
        assert len(logged_info) == 1
        assert logged_info[0]["guardrail_mode"] == GuardrailEventHooks.pre_call

    @pytest.mark.asyncio
    async def test_log_guardrail_information_infers_event_type_from_async_post_call_success_hook(
        self,
    ):
        """
        Test that log_guardrail_information decorator correctly infers GuardrailEventHooks.post_call
        from async_post_call_success_hook function name.
        """
        from litellm.integrations.custom_guardrail import log_guardrail_information
        from litellm.types.guardrails import GuardrailEventHooks

        class TestGuardrail(CustomGuardrail):
            def __init__(self):
                super().__init__(
                    guardrail_name="test_event_type_guardrail",
                    event_hook=[
                        GuardrailEventHooks.pre_call,
                        GuardrailEventHooks.post_call,
                    ],
                )

            @log_guardrail_information
            async def async_post_call_success_hook(self, data: dict, **kwargs):
                return {"result": "post_call_executed"}

        guardrail = TestGuardrail()
        request_data = {"metadata": {}}

        await guardrail.async_post_call_success_hook(data=request_data)

        # Check that the guardrail_mode was set to post_call (not the full list)
        logged_info = request_data["metadata"]["standard_logging_guardrail_information"]
        assert len(logged_info) == 1
        assert logged_info[0]["guardrail_mode"] == GuardrailEventHooks.post_call

    @pytest.mark.asyncio
    async def test_log_guardrail_information_infers_event_type_from_async_moderation_hook(
        self,
    ):
        """
        Test that log_guardrail_information decorator correctly infers GuardrailEventHooks.during_call
        from async_moderation_hook function name.
        """
        from litellm.integrations.custom_guardrail import log_guardrail_information
        from litellm.types.guardrails import GuardrailEventHooks

        class TestGuardrail(CustomGuardrail):
            def __init__(self):
                super().__init__(
                    guardrail_name="test_event_type_guardrail",
                    event_hook=[
                        GuardrailEventHooks.during_call,
                        GuardrailEventHooks.post_call,
                    ],
                )

            @log_guardrail_information
            async def async_moderation_hook(self, data: dict, **kwargs):
                return {"result": "moderation_executed"}

        guardrail = TestGuardrail()
        request_data = {"metadata": {}}

        await guardrail.async_moderation_hook(data=request_data)

        # Check that the guardrail_mode was set to during_call (not the full list)
        logged_info = request_data["metadata"]["standard_logging_guardrail_information"]
        assert len(logged_info) == 1
        assert logged_info[0]["guardrail_mode"] == GuardrailEventHooks.during_call

    @pytest.mark.asyncio
    async def test_log_guardrail_information_infers_event_type_from_async_post_call_streaming_hook(
        self,
    ):
        """
        Test that log_guardrail_information decorator correctly infers GuardrailEventHooks.post_call
        from async_post_call_streaming_hook function name.
        """
        from litellm.integrations.custom_guardrail import log_guardrail_information
        from litellm.types.guardrails import GuardrailEventHooks

        class TestGuardrail(CustomGuardrail):
            def __init__(self):
                super().__init__(
                    guardrail_name="test_event_type_guardrail",
                    event_hook=[
                        GuardrailEventHooks.pre_call,
                        GuardrailEventHooks.post_call,
                    ],
                )

            @log_guardrail_information
            async def async_post_call_streaming_hook(self, data: dict, **kwargs):
                return {"result": "streaming_executed"}

        guardrail = TestGuardrail()
        request_data = {"metadata": {}}

        await guardrail.async_post_call_streaming_hook(data=request_data)

        # Check that the guardrail_mode was set to post_call (not the full list)
        logged_info = request_data["metadata"]["standard_logging_guardrail_information"]
        assert len(logged_info) == 1
        assert logged_info[0]["guardrail_mode"] == GuardrailEventHooks.post_call

    @pytest.mark.asyncio
    async def test_log_guardrail_information_returns_none_for_unknown_function_name(
        self,
    ):
        """
        Test that log_guardrail_information decorator returns None for event_type
        when function name doesn't match known patterns, and falls back to self.event_hook.
        """
        from litellm.integrations.custom_guardrail import log_guardrail_information
        from litellm.types.guardrails import GuardrailEventHooks

        class TestGuardrail(CustomGuardrail):
            def __init__(self):
                super().__init__(
                    guardrail_name="test_event_type_guardrail",
                    event_hook=GuardrailEventHooks.pre_call,
                )

            @log_guardrail_information
            async def some_other_hook(self, data: dict, **kwargs):
                return {"result": "other_hook_executed"}

        guardrail = TestGuardrail()
        request_data = {"metadata": {}}

        await guardrail.some_other_hook(data=request_data)

        # Check that the guardrail_mode falls back to self.event_hook
        logged_info = request_data["metadata"]["standard_logging_guardrail_information"]
        assert len(logged_info) == 1
        assert logged_info[0]["guardrail_mode"] == GuardrailEventHooks.pre_call

    def test_add_standard_logging_uses_event_type_over_event_hook(self):
        """
        Test that add_standard_logging_guardrail_information_to_request_data
        prioritizes event_type parameter over self.event_hook.
        """
        from litellm.types.guardrails import GuardrailEventHooks

        guardrail = CustomGuardrail(
            guardrail_name="test_guardrail",
            event_hook=[GuardrailEventHooks.pre_call, GuardrailEventHooks.post_call],
        )

        request_data = {"metadata": {}}

        # Call with explicit event_type
        guardrail.add_standard_logging_guardrail_information_to_request_data(
            guardrail_json_response={"result": "ok"},
            request_data=request_data,
            guardrail_status="success",
            event_type=GuardrailEventHooks.post_call,
        )

        # Should use the provided event_type (post_call), not the full event_hook list
        logged_info = request_data["metadata"]["standard_logging_guardrail_information"]
        assert len(logged_info) == 1
        assert logged_info[0]["guardrail_mode"] == GuardrailEventHooks.post_call

    def test_add_standard_logging_falls_back_to_event_hook_when_event_type_is_none(
        self,
    ):
        """
        Test that add_standard_logging_guardrail_information_to_request_data
        falls back to self.event_hook when event_type is None.
        """
        from litellm.types.guardrails import GuardrailEventHooks

        guardrail = CustomGuardrail(
            guardrail_name="test_guardrail",
            event_hook=GuardrailEventHooks.pre_call,
        )

        request_data = {"metadata": {}}

        # Call with event_type=None
        guardrail.add_standard_logging_guardrail_information_to_request_data(
            guardrail_json_response={"result": "ok"},
            request_data=request_data,
            guardrail_status="success",
            event_type=None,
        )

        # Should fall back to self.event_hook
        logged_info = request_data["metadata"]["standard_logging_guardrail_information"]
        assert len(logged_info) == 1
        assert logged_info[0]["guardrail_mode"] == GuardrailEventHooks.pre_call


class TestTracingFieldsPopulation:
    """Verify add_standard_logging_guardrail_information_to_request_data passes tracing_detail fields."""

    def test_new_fields_set_on_slg(self):
        cg = CustomGuardrail(guardrail_name="test-rail")
        request_data = {"metadata": {}}
        cg.add_standard_logging_guardrail_information_to_request_data(
            guardrail_json_response={"result": "ok"},
            request_data=request_data,
            guardrail_status="success",
            tracing_detail=GuardrailTracingDetail(
                guardrail_id="rail-123",
                policy_template="EU AI Act Article 5",
                detection_method="regex",
                confidence_score=0.95,
                match_details=[{"type": "pattern", "action_taken": "BLOCK"}],
                patterns_checked=12,
                alert_recipients=["admin@example.com"],
            ),
        )
        slg_list = request_data["metadata"]["standard_logging_guardrail_information"]
        assert len(slg_list) == 1
        slg = slg_list[0]
        assert slg["guardrail_id"] == "rail-123"
        assert slg["policy_template"] == "EU AI Act Article 5"
        assert slg["detection_method"] == "regex"
        assert slg["confidence_score"] == 0.95
        assert slg["patterns_checked"] == 12
        assert slg["alert_recipients"] == ["admin@example.com"]
        assert len(slg["match_details"]) == 1

    def test_new_fields_default_to_absent(self):
        """When tracing_detail is not passed, new fields are absent from the SLG dict."""
        cg = CustomGuardrail(guardrail_name="test-rail")
        request_data = {"metadata": {}}
        cg.add_standard_logging_guardrail_information_to_request_data(
            guardrail_json_response="ok",
            request_data=request_data,
            guardrail_status="success",
        )
        slg = request_data["metadata"]["standard_logging_guardrail_information"][0]
        assert slg.get("guardrail_id") is None
        assert slg.get("policy_template") is None
        assert slg.get("confidence_score") is None

    def test_multiple_guardrails_with_different_policies(self):
        """One request, multiple guardrails each with own policy_template."""
        cg1 = CustomGuardrail(guardrail_name="rail-1")
        cg2 = CustomGuardrail(guardrail_name="rail-2")
        request_data = {"metadata": {}}

        cg1.add_standard_logging_guardrail_information_to_request_data(
            guardrail_json_response="ok",
            request_data=request_data,
            guardrail_status="success",
            tracing_detail=GuardrailTracingDetail(policy_template="GDPR"),
        )
        cg2.add_standard_logging_guardrail_information_to_request_data(
            guardrail_json_response="blocked",
            request_data=request_data,
            guardrail_status="guardrail_intervened",
            tracing_detail=GuardrailTracingDetail(policy_template="EU AI Act Article 5"),
        )

        slg_list = request_data["metadata"]["standard_logging_guardrail_information"]
        assert len(slg_list) == 2
        assert slg_list[0]["policy_template"] == "GDPR"
        assert slg_list[1]["policy_template"] == "EU AI Act Article 5"

    def test_classification_field_passed_through(self):
        """Classification dict for LLM-judge guardrails is passed through."""
        cg = CustomGuardrail(guardrail_name="judge-rail")
        request_data = {"metadata": {}}
        classification = {
            "flagged": True,
            "category": "workplace_emotion_recognition",
            "article_reference": "Article 5(1)(f)",
            "confidence": 0.94,
            "reason": "Request asks to analyze employee sentiment",
        }
        cg.add_standard_logging_guardrail_information_to_request_data(
            guardrail_json_response="blocked",
            request_data=request_data,
            guardrail_status="guardrail_intervened",
            tracing_detail=GuardrailTracingDetail(
                classification=classification,
                detection_method="llm-judge",
                confidence_score=0.94,
            ),
        )
        slg = request_data["metadata"]["standard_logging_guardrail_information"][0]
        assert slg["classification"] == classification
        assert slg["detection_method"] == "llm-judge"
        assert slg["confidence_score"] == 0.94
